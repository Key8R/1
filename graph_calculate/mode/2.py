from pyspark.ml.linalg import DenseVector
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, DoubleType
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.utils import resample
from EdgeData import EdgeData
from FeatureEngineering import FeatureEngineering
from NodeData import NodeData
import numpy as np


def get_last_char(s):
    return s[-1] if s else None


def string_to_vector(string_data):
    array_data = string_data.split(":")
    return Vectors.dense([float(x) for x in array_data])


def extract(row):
    return tuple(row.toArray().tolist())


extract_udf = udf(extract, ArrayType(DoubleType()))


def train_and_evaluate_model(training_data, feature_vector_size):
    null_labels = training_data.filter(col("edge_type").isNull())
    nan_labels = training_data.filter(expr("isnan(edge_type)"))
    if null_labels.count() > 0:
        print("Null labels found in the dataset:")
        null_labels.show()
        return
    elif nan_labels.count() > 0:
        print("NaN labels found in the dataset:")
        nan_labels.show()
        return

    training_data = training_data.withColumn("features_arr", extract_udf(col("features")))
    for i in range(feature_vector_size):
        training_data = training_data.withColumn(f"features_{i}", col("features_arr")[i])

    string_indexer = StringIndexer(inputCol="edge_type", outputCol="indexed_edge_type_tmp")
    indexed_df = string_indexer.fit(training_data).transform(training_data)
    indexed_df = indexed_df.withColumnRenamed("indexed_edge_type_tmp", "indexed_edge_type")

    train_data, test_data = indexed_df.randomSplit([0.8, 0.2], seed=1234)

    features_col = [f"features_{i}" for i in range(feature_vector_size)]
    assembler = VectorAssembler(
        inputCols=features_col,
        outputCol="features_assembled"
    )

    train_data = assembler.transform(train_data)
    train_features_np = np.array(train_data.select("features_assembled").collect())
    train_labels_np = np.array(train_data.select("indexed_edge_type").collect())

    xgb = XGBClassifier()
    model = xgb.fit(train_features_np, train_labels_np)

    test_data = assembler.transform(test_data)
    predictions = model.predict(np.array(test_data.select("features_assembled").collect()))
    test_labels = np.array(test_data.select("indexed_edge_type").collect())
    accuracy = accuracy_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions, average='weighted')
    f1_score_val = f1_score(test_labels, predictions, average='weighted')
    return accuracy, recall, f1_score_val


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Graph Model Training") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    spark.udf.register("get_last_char_udf", get_last_char, returnType="string")
    spark.udf.register("string_to_vector_udf", string_to_vector, VectorUDT())
    edge_data_loader = EdgeData("hdfs://192.168.75.100:8020/a/session_edges.csv")
    node_data_loader = NodeData("hdfs://192.168.75.100:8020/a/session_nodes.csv")
    nodes_df = node_data_loader.load_data(spark)
    edges_df = edge_data_loader.load_data(spark)
    processed_nodes_df = node_data_loader.process_data(nodes_df)
    feature_engineering = FeatureEngineering(processed_nodes_df, edges_df)
    combined_df = feature_engineering.combine_data()
    feature_vector_size = 128
    accuracy, recall, f1_score = train_and_evaluate_model(combined_df, feature_vector_size)
    if accuracy is not None:
        print("Model Accuracy:", accuracy)
        print("Model Recall:", recall)
        print("Model F1 Score:", f1_score)
