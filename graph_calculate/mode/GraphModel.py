from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class GraphModel:
    def __init__(self, spark_session, data):
        self.spark_session = spark_session
        self.data = data

    def train(self):
        # 划分数据集
        train_data, test_data = self.data.randomSplit([0.8, 0.2], seed=42)

        # 定义模型
        gbt = GBTClassifier(labelCol="type", featuresCol="node_features", maxIter=10)

        # 训练模型
        model = gbt.fit(train_data)

        # 在测试集上进行预测
        predictions = model.transform(test_data)

        # 评估模型性能
        evaluator = MulticlassClassificationEvaluator(labelCol="type", predictionCol="prediction",
                                                      metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)

        return accuracy
