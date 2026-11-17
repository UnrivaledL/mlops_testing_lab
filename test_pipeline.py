"""
Автотесты пайплайна данных Taxi Trip Pricing:
1) валидация схемы (Pandera),
2) аудит качества (Great Expectations),
3) бизнес-правило формирования цены,
4) обнаружение дрейфа данных (KS-тест и хи-квадрат).
"""
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, ks_2samp

from data_validation import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    SchemaError,
    clean_data,
    load_data,
    price_formula_match_share,
    run_great_expectations,
    validate_data,
    validate_with_great_expectations,
)

ALPHA = 0.01          # уровень значимости для теста "дрейфа нет"
RANDOM_STATE = 42
MIN_ROWS_FOR_STATS = 100


# --- ФИКСТУРЫ ---
@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    df = load_data()
    assert not df.empty, "Файл с данными пуст"
    return df


@pytest.fixture(scope="module")
def split(raw_df):
    """Эталонная (reference) и текущая (current) выборки — случайное деление 50/50."""
    if len(raw_df) < MIN_ROWS_FOR_STATS:
        pytest.skip(f"Для статистических тестов нужно >= {MIN_ROWS_FOR_STATS} строк")
    reference = raw_df.sample(frac=0.5, random_state=RANDOM_STATE)
    current = raw_df.drop(reference.index)
    return reference, current


def _corrupt(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    bad = df.head(5).copy()
    bad.loc[bad.index[0], column] = value
    return bad


# --- 1. ТЕСТЫ ВАЛИДАЦИИ СХЕМЫ (PANDERA) ---
def test_valid_data_pandera(raw_df):
    validated_df = validate_data(raw_df)
    assert not validated_df.empty
    assert len(validated_df) == len(raw_df)


def test_cleaned_data_pandera(raw_df):
    cleaned = clean_data(raw_df)
    validate_data(cleaned)
    assert cleaned.isna().sum().sum() == 0


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("Trip_Distance_km", -5.0),          # отрицательное расстояние
        ("Trip_Duration_Minutes", 10_000.0),  # нереальная длительность
        ("Base_Fare", -1.0),                  # отрицательный тариф
        ("Passenger_Count", 2.5),             # нецелое число пассажиров
        ("Weather", "Tornado"),               # неизвестная категория
        ("Time_of_Day", "Midnight"),          # неизвестная категория
    ],
)
def test_invalid_values_pandera(raw_df, column, bad_value):
    with pytest.raises(SchemaError):
        validate_data(_corrupt(raw_df, column, bad_value))


def test_missing_column_pandera(raw_df):
    with pytest.raises(SchemaError):
        validate_data(raw_df.drop(columns="Trip_Price"))


def test_extra_column_pandera(raw_df):
    df = raw_df.copy()
    df["Driver_Name"] = "Ivan"
    with pytest.raises(SchemaError):
        validate_data(df)


# --- 2. ТЕСТЫ GREAT EXPECTATIONS ---
def test_great_expectations_raw_data(raw_df):
    results = run_great_expectations(raw_df)
    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"Не выполнены ожидания: {failed}"


def test_great_expectations_strict_nulls_on_clean_data(raw_df):
    assert validate_with_great_expectations(clean_data(raw_df), strict_nulls=True) is True


def test_great_expectations_detects_nulls_in_raw_data(raw_df):
    # В исходном датасете есть пропуски — строгая проверка должна их поймать
    assert validate_with_great_expectations(raw_df, strict_nulls=True) is False


def test_great_expectations_detects_bad_values(raw_df):
    bad = raw_df.copy()
    bad["Trip_Duration_Minutes"] = -1.0
    bad.loc[bad.index[:3], "Weather"] = "Tornado"
    results = run_great_expectations(bad)
    assert results["expect_column_values_to_be_between[Trip_Duration_Minutes]"] is False
    assert results["expect_column_values_to_be_in_set[Weather]"] is False


# --- 3. БИЗНЕС-ПРАВИЛО: ЦЕНА = ПОСАДКА + КМ * ТАРИФ + МИН * ТАРИФ ---
def test_price_formula_consistency(raw_df):
    share = price_formula_match_share(raw_df)
    assert share >= 0.9, f"Формуле тарифа соответствует только {share:.1%} строк"


# --- 4. ТЕСТЫ НА ОБНАРУЖЕНИЕ ДРЕЙФА ДАННЫХ ---
def test_data_drift_detection_ks(split):
    """KS-тест по всем числовым признакам: дрейфа быть не должно.
    Поправка Бонферрони — так как проверяем сразу несколько колонок."""
    reference, current = split
    alpha = ALPHA / len(NUMERIC_COLUMNS)
    p_values = {
        col: ks_2samp(reference[col].dropna(), current[col].dropna()).pvalue
        for col in NUMERIC_COLUMNS
    }
    drifted = {c: round(p, 5) for c, p in p_values.items() if p <= alpha}
    assert not drifted, f"Обнаружен дрейф (p-value): {drifted}"


def test_categorical_drift_detection_chi2(split):
    """Хи-квадрат по категориальным признакам: дрейфа быть не должно."""
    reference, current = split
    alpha = ALPHA / len(CATEGORICAL_COLUMNS)
    drifted = {}
    for col in CATEGORICAL_COLUMNS:
        table = pd.concat(
            [reference[col].value_counts(), current[col].value_counts()], axis=1
        ).fillna(0)
        p_value = chi2_contingency(table.values)[1]
        if p_value <= alpha:
            drifted[col] = round(p_value, 5)
    assert not drifted, f"Обнаружен дрейф категорий (p-value): {drifted}"


def test_simulated_drift_failure(split):
    """Имитация дрейфа: расстояния поездок выросли в 3 раза — KS должен это заметить."""
    reference, current = split
    current_drifted = current.copy()
    current_drifted["Trip_Distance_km"] *= 3.0

    p_value = ks_2samp(
        reference["Trip_Distance_km"].dropna(),
        current_drifted["Trip_Distance_km"].dropna(),
    ).pvalue
    assert p_value < 0.05  # Дрейф успешно обнаружен


def test_simulated_categorical_drift_failure(split):
    """Имитация дрейфа: в текущих данных везде снег — хи-квадрат должен это заметить."""
    reference, current = split
    current_drifted = current.copy()
    current_drifted["Weather"] = "Snow"

    table = pd.concat(
        [reference["Weather"].value_counts(), current_drifted["Weather"].value_counts()],
        axis=1,
    ).fillna(0)
    p_value = chi2_contingency(table.values)[1]
    assert p_value < 0.05  # Дрейф успешно обнаружен
