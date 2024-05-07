from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.functions import udf, expr
from pyspark.sql import SparkSession

def string_to_vector(string_data):
    array_data = string_data.split(":")
    return Vectors.dense([float(x) for x in array_data])

class NodeData:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self, spark):
        # 加载数据并为列重命名
        nodes_df = spark.read.csv(self.file_path, header=False, inferSchema=True)
        nodes_df = nodes_df.withColumnRenamed("_c0", "node_id") \
                           .withColumnRenamed("_c1", "node_type") \
                           .withColumnRenamed("_c2", "node_atts")
        # 直接限制加载的数据行数以避免使用count()，提高效率
        return nodes_df.limit(1000)

    def process_data(self, nodes_df):
        # 使用注册的UDF转换特征
        processed_nodes_df = nodes_df.withColumn("features", expr("string_to_vector_udf(node_atts)"))
        return processed_nodes_df

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Graph Analysis with PySpark") \
        .getOrCreate()

    # 注册UDF，指定返回类型为 VectorUDT（DenseVector）
    spark.udf.register("string_to_vector_udf", string_to_vector, VectorUDT())

    # 创建NodeData实例，并加载处理数据
    nd = NodeData("hdfs://192.168.75.100:8020/a/session_nodes.csv")
    nodes_df = nd.load_data(spark)
    nodes_df.show()
    processed_df = nd.process_data(nodes_df)
    processed_df.show()
