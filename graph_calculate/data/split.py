import pandas as pd

# 读取整个数据集
data = pd.read_csv(r"F:\bishe\graph_calculate\data\icdm2022_session1_nodes.csv")

# 随机抽取数据集的八分之一
sampled_data = data.sample(frac=1/8)

# 可以选择将抽样数据保存到新文件中
sampled_data.to_csv(r"F:\bishe\graph_calculate\data\icdm2022_session1_nodes_sampled.csv", index=False)
