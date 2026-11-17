FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GX_ANALYTICS_ENABLED=False

WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код проекта и данные (data/taxi_trip_pricing.csv)
COPY . .

# Запуск авто-тестов внутри контейнера при его старте
CMD ["pytest", "-v"]
