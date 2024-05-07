from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# 初始化 SparkSession，配置内存管理相关参数
spark = SparkSession.builder \
    .appName("Degree Centrality Calculation") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# 读取边数据，为列指定名称
edges_df = spark.read.csv("hdfs://192.168.75.100:8020/a/session_edges.csv", inferSchema=True, header=False) \
    .toDF("source", "target", "source_type", "target_type", "edge_type")

# 计算每个节点的入度
in_degree_df = edges_df.groupBy("target").count() \
    .withColumnRenamed("count", "in_degree") \
    .withColumnRenamed("target", "node_id")

# 计算每个节点的出度
out_degree_df = edges_df.groupBy("source").count() \
    .withColumnRenamed("count", "out_degree") \
    .withColumnRenamed("source", "node_id")

# 将入度和出度合并到一个DataFrame中
degree_df = in_degree_df.join(out_degree_df, "node_id", "outer") \
    .fillna(0)  # 使用外连接并填充缺失值为0

# 计算节点的度中心性
total_nodes = degree_df.count() - 1  # 减去1，因为中心性计算不包括节点自身
degree_df = degree_df.withColumn("degree_centrality",
                                 (col("in_degree") + col("out_degree")) / total_nodes)

# 保存处理后的数据到CSV文件
degree_df.write.csv("hdfs://192.168.75.100:8020/a/processed_edges.csv", mode="overwrite", header=True)

# 打印处理后的边数据
degree_df.show()

print("数据已保存")

# 关闭 SparkSession
spark.stop()
