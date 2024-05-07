from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.ml.linalg import Vectors, VectorUDT

from EdgeData import EdgeData
from FeatureEngineering import FeatureEngineering
from NodeData import NodeData


def get_last_char(s):
    """获取字符串的最后一个字符"""
    return s[-1] if s else None


def string_to_vector(string_data):
    array_data = string_data.split(":")
    return Vectors.dense([float(x) for x in array_data])


def train_and_evaluate_model(training_data):
    """训练模型并评估准确度、召回率和F1分数"""
    # 检查标签列是否包含 null 或 NaN 值
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
    rf = RandomForestClassifier(featuresCol="features", labelCol="indexed_edge_type")
    model = rf.fit(training_data)
    predictions = model.transform(training_data)

    # 计算准确率
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="indexed_edge_type", predictionCol="prediction",
                                                           metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)

    # 计算召回率
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="indexed_edge_type", predictionCol="prediction",
                                                         metricName="weightedRecall")
    recall = evaluator_recall.evaluate(predictions)

    # 计算F1分数
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="indexed_edge_type", predictionCol="prediction",
                                                     metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)

    return accuracy, recall, f1_score


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Graph Model Training") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()

    # 注册UDF
    spark.udf.register("get_last_char_udf", get_last_char, returnType="string")
    spark.udf.register("string_to_vector_udf", string_to_vector, VectorUDT())

    # 数据加载
    edge_data_loader = EdgeData("hdfs://192.168.75.100:8020/a/session_edges.csv")
    node_data_loader = NodeData("hdfs://192.168.75.100:8020/a/session_nodes.csv")
    nodes_df = node_data_loader.load_data(spark)
    edges_df = edge_data_loader.load_data(spark)

    processed_nodes_df = node_data_loader.process_data(nodes_df)

    # 特征工程
    feature_engineering = FeatureEngineering(processed_nodes_df, edges_df)
    combined_df = feature_engineering.combine_data()

    # 使用StringIndexer将字符串标签列转换为数值标签列
    string_indexer = StringIndexer(inputCol="edge_type", outputCol="indexed_edge_type")
    indexed_df = string_indexer.fit(combined_df).transform(combined_df)

    # 训练和评估
    accuracy, recall, f1_score = train_and_evaluate_model(indexed_df)
    if accuracy is not None:
        print("Model Accuracy:", accuracy)
        print("Model Recall:", recall)
        print("Model F1 Score:", f1_score)
