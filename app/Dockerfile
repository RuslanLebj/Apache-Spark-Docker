FROM python:3.9-slim

WORKDIR /app

# Установка Java (Java 17)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless ca-certificates-java && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем переменную окружения JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Копируем requirements.txt
COPY requirements.txt /app/

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . /app/

# Команда по умолчанию для запуска Python приложения
CMD ["python", "analysis.py"]
