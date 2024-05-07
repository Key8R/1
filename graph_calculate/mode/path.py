import os
import platform


def get_spark_jar_path():
    # 获取当前操作系统
    current_os = platform.system()

    # 根据操作系统设置XGBoost的JAR文件路径
    if current_os == 'Windows':
        # Windows上的JAR文件路径
        return "/path/to/windows/xgboost4j.jar,/path/to/windows/xgboost4j-spark.jar,/path/to/windows/sparkxgb.zip"
    else:
        # Linux上的JAR文件路径
        return "/path/to/linux/xgboost4j.jar,/path/to/linux/xgboost4j-spark.jar,/path/to/linux/sparkxgb.zip"


if __name__ == "__main__":
    # 获取XGBoost的JAR文件路径
    spark_jar_path = get_spark_jar_path()
    print("Spark JAR Path:", spark_jar_path)

    # 设置PYSPARK_SUBMIT_ARGS环境变量
    os.environ["PYSPARK_SUBMIT_ARGS"] = f'--jars {spark_jar_path} pyspark-shell'
