import pandas as pd

# 从HDFS中读取CSV文件的前20000行
hdfs_csv_path = "hdfs://path/to/your/file.csv"
df = pd.read_csv(hdfs_csv_path, nrows=20000)

# 将数据保存为新文件
new_file_path = "/path/to/save/new_file.csv"
df.to_csv(new_file_path, index=False)

print("文件已保存至:", new_file_path)
