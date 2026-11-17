"""
Валидация данных датасета Taxi Trip Pricing.

Часть 1 — Pandera: структура, типы, допустимые диапазоны и категории.
Часть 2 — Great Expectations: аудит качества (наличие колонок, пропуски, диапазоны, множества значений).
Дополнительно — очистка данных и проверка бизнес-правила формирования цены.

Запуск отчёта вручную:  python data_validation.py
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

try:  # pandera >= 0.24 рекомендует импорт через pandera.pandas
    import pandera.pandas as pa
except ImportError:  # более старые версии pandera
    import pandera as pa
from pandera.errors import SchemaError, SchemaErrors  # noqa: F401 (используются в тестах)

# Отключаем отправку анонимной статистики Great Expectations
os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")

# ---------------------------------------------------------------------------
# Описание датасета
# ---------------------------------------------------------------------------
DATA_PATH = Path(
    os.getenv(
        "TAXI_DATA_PATH",
        Path(__file__).resolve().parent / "data" / "taxi_trip_pricing.csv",
    )
)

COLUMN_ORDER = [
    "Trip_Distance_km",
    "Time_of_Day",
    "Day_of_Week",
    "Passenger_Count",
    "Traffic_Conditions",
    "Weather",
    "Base_Fare",
    "Per_Km_Rate",
    "Per_Minute_Rate",
    "Trip_Duration_Minutes",
    "Trip_Price",
]

# Физически допустимые интервалы числовых признаков (с запасом)
NUMERIC_RANGES = {
    "Trip_Distance_km": (0.0, 300.0),
    "Passenger_Count": (1.0, 8.0),
    "Base_Fare": (0.0, 20.0),
    "Per_Km_Rate": (0.0, 10.0),
    "Per_Minute_Rate": (0.0, 5.0),
    "Trip_Duration_Minutes": (0.0, 300.0),
    "Trip_Price": (0.0, 1000.0),
}

# Допустимые значения категориальных признаков
CATEGORICAL_VALUES = {
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
    "Day_of_Week": ["Weekday", "Weekend"],
    "Traffic_Conditions": ["Low", "Medium", "High"],
    "Weather": ["Clear", "Rain", "Snow"],
}

NUMERIC_COLUMNS = list(NUMERIC_RANGES)
CATEGORICAL_COLUMNS = list(CATEGORICAL_VALUES)

# В исходном датасете есть пропуски (~5% в каждой колонке).
# Для «сырых» данных допускаем не более 20% пропусков в колонке.
MIN_NON_NULL_SHARE = 0.8


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Загружает CSV с данными поездок."""
    return pd.read_csv(path or DATA_PATH)


# ---------------------------------------------------------------------------
# Часть 1. Pandera
# ---------------------------------------------------------------------------
def _numeric_column(low: float, high: float, *extra_checks) -> pa.Column:
    return pa.Column(
        float,
        checks=[pa.Check.in_range(low, high), *extra_checks],
        nullable=True,  # пропуски в сырых данных допустимы
        coerce=True,    # int -> float, если в колонке нет NaN
    )


_numeric_columns = {col: _numeric_column(lo, hi) for col, (lo, hi) in NUMERIC_RANGES.items()}
_numeric_columns["Passenger_Count"] = _numeric_column(
    *NUMERIC_RANGES["Passenger_Count"],
    pa.Check(lambda s: s % 1 == 0, error="Passenger_Count должен быть целым числом"),
)

_categorical_columns = {
    col: pa.Column(str, checks=pa.Check.isin(values), nullable=True)
    for col, values in CATEGORICAL_VALUES.items()
}

# Декларативная схема валидации Pandera
taxi_schema = pa.DataFrameSchema(
    columns={**_numeric_columns, **_categorical_columns},
    strict=True,  # лишние колонки запрещены, отсутствующие — ошибка
    name="TaxiTripPricingSchema",
)


def validate_data(df: pd.DataFrame, lazy: bool = False) -> pd.DataFrame:
    """Проверяет DataFrame по схеме.

    lazy=False -> выбрасывает SchemaError при первой ошибке;
    lazy=True  -> собирает все ошибки и выбрасывает SchemaErrors.
    """
    return taxi_schema.validate(df, lazy=lazy)


# ---------------------------------------------------------------------------
# Очистка данных и бизнес-правило цены
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет строки без целевой переменной, заполняет пропуски
    медианой (числовые) и модой (категориальные)."""
    out = df.dropna(subset=["Trip_Price"]).copy()
    for col in NUMERIC_COLUMNS:
        out[col] = out[col].fillna(out[col].median())
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna(out[col].mode().iloc[0])
    return out.reset_index(drop=True)


def expected_price(df: pd.DataFrame) -> pd.Series:
    """Цена = посадка + км * тариф_км + минуты * тариф_мин."""
    return (
        df["Base_Fare"]
        + df["Trip_Distance_km"] * df["Per_Km_Rate"]
        + df["Trip_Duration_Minutes"] * df["Per_Minute_Rate"]
    )


def price_formula_match_share(df: pd.DataFrame, tol: float = 0.01) -> float:
    """Доля полных строк, в которых Trip_Price совпадает с формулой тарифа."""
    cols = ["Base_Fare", "Trip_Distance_km", "Per_Km_Rate",
            "Trip_Duration_Minutes", "Per_Minute_Rate", "Trip_Price"]
    full = df.dropna(subset=cols)
    if full.empty:
        return 1.0
    diff = (full["Trip_Price"] - expected_price(full)).abs()
    return float((diff <= tol).mean())


# ---------------------------------------------------------------------------
# Часть 2. Great Expectations
# ---------------------------------------------------------------------------
def _build_expectation_specs(strict_nulls: bool) -> list[tuple[str, dict]]:
    mostly = 1.0 if strict_nulls else MIN_NON_NULL_SHARE
    specs: list[tuple[str, dict]] = [
        # 1. Проверка существования колонок
        ("expect_table_columns_to_match_set", {"column_set": COLUMN_ORDER, "exact_match": True}),
        ("expect_column_to_exist", {"column": "Trip_Distance_km"}),
        ("expect_column_to_exist", {"column": "Trip_Price"}),
    ]
    # 2. Проверка отсутствия пропусков
    for col in COLUMN_ORDER:
        specs.append(("expect_column_values_to_not_be_null", {"column": col, "mostly": mostly}))
    # 3. Значения в физически допустимом интервале
    for col, (lo, hi) in NUMERIC_RANGES.items():
        specs.append(("expect_column_values_to_be_between",
                      {"column": col, "min_value": lo, "max_value": hi}))
    # 4. Категории только из допустимого множества
    for col, values in CATEGORICAL_VALUES.items():
        specs.append(("expect_column_values_to_be_in_set", {"column": col, "value_set": values}))
    return specs


def _snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _key(name: str, kwargs: dict) -> str:
    return f"{name}[{kwargs.get('column', 'table')}]"


def run_great_expectations(df: pd.DataFrame, strict_nulls: bool = False) -> dict[str, bool]:
    """Прогоняет все ожидания и возвращает словарь {ожидание[колонка]: успех}.

    Работает и со старым API (great_expectations < 1.0, ge.from_pandas),
    и с новым API (great_expectations >= 1.0).
    """
    import great_expectations as gx

    specs = _build_expectation_specs(strict_nulls)
    results: dict[str, bool] = {}

    # --- Старый API (0.x): ge.from_pandas ---
    if hasattr(gx, "from_pandas"):
        ge_df = gx.from_pandas(df)
        for name, kwargs in specs:
            key = _key(name, kwargs)
            try:
                results[key] = bool(getattr(ge_df, name)(**kwargs).success)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"{key}: {exc}")
                results[key] = False
        return results

    # --- Новый API (1.x): эфемерный контекст в памяти ---
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name="taxi_source")
    asset = source.add_dataframe_asset(name="taxi_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("taxi_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    for name, kwargs in specs:
        key = _key(name, kwargs)
        try:
            expectation = getattr(gx.expectations, _snake_to_camel(name))(**kwargs)
            results[key] = bool(batch.validate(expectation).success)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"{key}: {exc}")
            results[key] = False
    return results


def validate_with_great_expectations(df: pd.DataFrame, strict_nulls: bool = False) -> bool:
    """True, если все ожидания Great Expectations выполнены."""
    return all(run_great_expectations(df, strict_nulls=strict_nulls).values())


# ---------------------------------------------------------------------------
# Отчёт при ручном запуске
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print(f"Файл: {DATA_PATH}\nСтрок: {len(data)}, колонок: {data.shape[1]}\n")
    print("Пропуски по колонкам:\n", data.isna().sum().to_string(), "\n", sep="")

    try:
        validate_data(data, lazy=True)
        print("Pandera (сырые данные): OK")
    except SchemaErrors as err:
        print("Pandera (сырые данные): ОШИБКИ\n", err.failure_cases)

    cleaned = clean_data(data)
    validate_data(cleaned)
    print(f"Pandera (очищенные данные, {len(cleaned)} строк): OK\n")

    print("Great Expectations (сырые данные):")
    for k, v in run_great_expectations(data).items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    print(f"\nGE строгая проверка пропусков (очищенные данные): "
          f"{validate_with_great_expectations(cleaned, strict_nulls=True)}")
    print(f"Доля строк, где цена совпадает с формулой тарифа: "
          f"{price_formula_match_share(data):.1%}")
