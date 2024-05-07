from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import matplotlib.pyplot as plt

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("Data Analysis") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 定义 schema，假设所有字段都是字符串类型
edge_schema = StructType([
    StructField("source", StringType(), True),
    StructField("target", StringType(), True),
    StructField("source_type", StringType(), True),
    StructField("target_type", StringType(), True),
    StructField("edge_type", StringType(), True)
])

node_schema = StructType([
    StructField("node_id", StringType(), True),
    StructField("node_type", StringType(), True),
    StructField("node_atts", StringType(), True)
])

label_schema = StructType([
    StructField("item_id", StringType(), True)
])

# 使用 PySpark 读取 HDFS 上的数据，指定 schema
edges_df = spark.read.csv("hdfs://192.168.75.100:8020/a/session_edges.csv", schema=edge_schema)
nodes_df = spark.read.csv("hdfs://192.168.75.100:8020/a/session_nodes.csv", schema=node_schema)
labels_df = spark.read.csv("hdfs://192.168.75.100:8020/a/session1_test_ids.txt", schema=label_schema)

# 采样数据
edges_sampled_df = edges_df.sample(fraction=0.1, withReplacement=False)
nodes_sampled_df = nodes_df.sample(fraction=0.1, withReplacement=False)
labels_sampled_df = labels_df.sample(fraction=0.1, withReplacement=False)

# 转换为 Pandas DataFrame 进行数据分析
edges_pdf = edges_sampled_df.toPandas()
nodes_pdf = nodes_sampled_df.toPandas()
labels_pdf = labels_sampled_df.toPandas()

print("读取数据完成")

# 统计节点类型分布
node_type_counts = nodes_pdf["node_type"].value_counts()

# 绘制节点类型分布图
plt.figure(figsize=(10, 8))
node_type_counts.plot(kind="bar", color="skyblue")
plt.title("Node Type Distribution")
plt.xlabel("Node Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

print("节点类型分布图完成")

# 统计边类型分布
edge_type_counts = edges_pdf["edge_type"].value_counts()

# 绘制边类型分布图
plt.figure(figsize=(10, 8))
edge_type_counts.plot(kind="bar", color="salmon")
plt.title("Edge Type Distribution")
plt.xlabel("Edge Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

print("边类型分布图完成")

# 关闭 SparkSession
spark.stop()
