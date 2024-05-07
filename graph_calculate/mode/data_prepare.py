from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, split

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("Data Processing on Spark") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# 读取数据
edges_df = spark.read.csv("hdfs://192.168.75.100:8020/a/session_edges.csv", header=False, inferSchema=True)
nodes_df = spark.read.csv("hdfs://192.168.75.100:8020/a/session_nodes.csv", header=False, inferSchema=True)
test_ids_df = spark.read.csv("hdfs://192.168.75.100:8020/a/session1_test_ids.txt", header=False, inferSchema=True)

# 抽样
edges_df = edges_df.sample(withReplacement=False, fraction=0.1, seed=42)
nodes_df = nodes_df.sample(withReplacement=False, fraction=0.1, seed=42)
test_ids_df = test_ids_df.sample(withReplacement=False, fraction=0.1, seed=42)

# 解析节点特征并将数组转换为字符串
nodes_df = nodes_df.withColumn("features", split(col("_c2"), ":"))
nodes_df = nodes_df.withColumn("features_str", concat_ws(",", col("features")))

# 输出原始数据信息
print("原始数据信息：")
edges_df.show()
nodes_df.show()
test_ids_df.show()

print("Edges 数据行数:", edges_df.count())
print("Nodes 数据行数:", nodes_df.count())
print("Test IDs 数据行数:", test_ids_df.count())

# 去除重复数据
edges_df = edges_df.dropDuplicates()
nodes_df = nodes_df.dropDuplicates(["_c0", "_c1", "features_str"])

# 处理缺失值
missing_values_before = nodes_df.na.drop().count()
nodes_df = nodes_df.na.drop()
edges_df = edges_df.na.drop()
missing_values_after = nodes_df.count()

print("处理缺失值：")
print("处理前缺失值数量:", missing_values_before)
print("处理后缺失值数量:", missing_values_after)

# 删除异常值
nodes_df = nodes_df.filter((col("_c1") == "a") | (col("_c1") == "f"))
edges_df = edges_df.filter((col("_c2") == "a") | (col("_c2") == "f")).filter((col("_c3") == "a") | (col("_c3") == "f"))

# 输出处理后数据信息
print("处理后数据信息：")
edges_df.show()
nodes_df.show()
test_ids_df.show()

print("Edges 数据行数:", edges_df.count())
print("Nodes 数据行数:", nodes_df.count())
print("Test IDs 数据行数:", test_ids_df.count())

# 保存处理后的数据到新的CSV文件
nodes_df.select("_c0", "_c1", "features_str").write.csv("hdfs://192.168.75.100:8020/a/processed_nodes.csv", mode="overwrite")
edges_df.select("_c0", "_c1", "_c4", "_c2", "_c3").write.csv("hdfs://192.168.75.100:8020/a/processed_edges.csv", mode="overwrite")
test_ids_df.write.csv("hdfs://192.168.75.100:8020/a/processed_test_ids.csv", mode="overwrite")

print("数据预处理完成并保存到文件。")

# 关闭 SparkSession
spark.stop()
