from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType
import numpy as np

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("Node Feature Engineering") \
    .getOrCreate()

# 读取数据，为列指定名称
nodes_df = spark.read.csv("hdfs://192.168.75.100:8020/a/session_nodes.csv", inferSchema=True, header=False) \
    .toDF("node_id", "node_type", "node_atts")

# 使用UDF解析节点特征
def parse_features(x):
    if x is None:
        return None
    return np.fromstring(x, sep=":").tolist()

parse_features_udf = udf(parse_features, ArrayType(FloatType()))
nodes_df = nodes_df.withColumn("features", parse_features_udf("node_atts"))

# 特征工程的UDF
def extract_features(features):
    if not features:
        return (None, None, None, None)
    features = np.array(features)
    return (float(np.mean(features)), float(np.max(features)), float(np.min(features)), float(np.sum(features)))

schema = StructType([
    StructField("mean", FloatType(), True),
    StructField("max", FloatType(), True),
    StructField("min", FloatType(), True),
    StructField("sum", FloatType(), True)
])

extract_features_udf = udf(extract_features, schema)
nodes_df = nodes_df.withColumn("extracted_features", extract_features_udf("features"))

# 拆分特征到多列
nodes_df = nodes_df.select(
    "node_id",
    "node_type",
    col("extracted_features").getField("mean").alias("mean"),
    col("extracted_features").getField("max").alias("max"),
    col("extracted_features").getField("min").alias("min"),
    col("extracted_features").getField("sum").alias("sum")
)

# 保存处理后的数据到 CSV 文件
nodes_df.write.csv("hdfs://192.168.75.100:8020/a/processed_nodes.csv", mode="overwrite", header=True)

# 显示处理后的数据
nodes_df.show()

print("数据已保存")

# 关闭 SparkSession
spark.stop()
