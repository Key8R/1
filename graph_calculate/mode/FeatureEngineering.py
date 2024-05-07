from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


class FeatureEngineering:
    def __init__(self, nodes_df, edges_df):
        self.nodes_df = nodes_df
        self.edges_df = edges_df

    def check_and_prepare_data(self):
        # 检查必要的列是否存在
        required_node_columns = ['node_id', 'node_features']
        required_edge_columns = ['source', 'target', 'edge_features', 'edge_type']

        missing_node_cols = [col for col in required_node_columns if col not in self.nodes_df.columns]
        missing_edge_cols = [col for col in required_edge_columns if col not in self.edges_df.columns]

        if missing_node_cols:
            raise ValueError("缺少必要的节点列: {}".format(missing_node_cols))
        if missing_edge_cols:
            raise ValueError("缺少必要的边列: {}".format(missing_edge_cols))

        # 可以在这里添加更多的数据预处理步骤

    def combine_data(self):
        # 将节点和边数据合并成一个 DataFrame，确保所有关联节点都有边
        combined_df = self.nodes_df.join(self.edges_df,
                                         (self.nodes_df.node_id == self.edges_df.source) |
                                         (self.nodes_df.node_id == self.edges_df.target), "inner")
        return combined_df

    def build_features(self, combined_df):
        # 构建节点特征向量
        assembler = VectorAssembler(inputCols=["node_features"], outputCol="features")  # 修改这里的输入列和输出列
        combined_df = assembler.transform(combined_df)
        return combined_df

    def select_features(self, combined_df):
        # 选择用于训练的特征列和标签列
        selected_cols = ["features", "edge_type"]  # 修改这里选择的列
        processed_df = combined_df.select(selected_cols)
        return processed_df
