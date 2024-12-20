from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Инициализация Spark сессии
spark = SparkSession.builder \
    .appName("Customer Analysis") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# # Загрузка данных
# data = spark.read.csv("/app/customers.csv", header=True, inferSchema=True)

# # Преобразование категориальных признаков
# indexer = StringIndexer(inputCol="health_status", outputCol="health_status_index")
# indexed_data = indexer.fit(data).transform(data)

# # Обработка данных (создание признаков)
# indexed_data = indexed_data.withColumn("features", Vectors.dense(col("age"), col("income")))

# # Разделение на обучающую и тестовую выборки
# train_data, test_data = indexed_data.randomSplit([0.8, 0.2], seed=1234)

# # Обучение модели RandomForest
# rf = RandomForestClassifier(labelCol="health_status_index", featuresCol="features")
# model = rf.fit(train_data)

# # Оценка модели на тестовых данных
# predictions = model.transform(test_data)

# # Показать результаты
# predictions.select("id", "name", "prediction", "health_status", "health_status_index").show()

# # Показать несколько строк с результатами
# print("### Прогнозы модели на тестовых данных ###")
# predictions.select("id", "name", "health_status", "health_status_index", "prediction").show(10)

# # Подсчитать метрики качества модели
# correct_predictions = predictions.filter(predictions["health_status_index"] == predictions["prediction"]).count()
# total_predictions = predictions.count()

# accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

# print(f"### Метрики качества модели ###")
# print(f"Всего примеров: {total_predictions}")
# print(f"Правильных предсказаний: {correct_predictions}")
# print(f"Точность: {accuracy:.2f}")

# Завершаем сессию Spark
spark.stop()

# print("Hello World")