from pyspark.sql import SparkSession
class EdgeData:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self, spark):
        # 读取边数据
        edges_df = spark.read.csv(self.file_path, header=False, inferSchema=True)
        edges_df = edges_df.withColumnRenamed("_c0", "source") \
            .withColumnRenamed("_c1", "target") \
            .withColumnRenamed("_c2", "source_type") \
            .withColumnRenamed("_c3", "target_type") \
            .withColumnRenamed("_c4", "edge_type")
        return edges_df.limit(int(edges_df.count() / 10))

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Graph Analysis with PySpark") \
        .getOrCreate()
    ed = EdgeData("hdfs://192.168.75.100:8020/a/session_edges.csv")
    ed.load_data(spark=spark).show()