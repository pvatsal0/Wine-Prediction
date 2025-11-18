import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

conf = SparkConf().setAppName('wine').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

val = spark.read.format("csv").load("s3://wine123/ValidationDataset.csv", header=True, sep=";")
val = VectorAssembler(inputCols=val.columns[1:-1], outputCol='features').transform(val).select(['features', 'label'])
features = np.array(val.select('features').collect())
label = np.array(val.select('label').collect())

def to_labeled_point(sc, features, labels):
    labeled_points = []
    for x, y in zip(features, labels):
        lp = (Vectors.dense(x), y)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)

dataset = to_labeled_point(sc, features, label)

model = RandomForestModel.load(sc, "s3://wine123/trainingmodel.model")
print("model loaded successfully")

preds = model.predict(dataset.map(lambda x: x[0]))

label = dataset.map(lambda lp: lp[1]).zip(preds)

labelpred_df = label.toDF(["label", "pred"]).toPandas()

F1score = f1_score(labelpred_df['label'], labelpred_df['pred'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'], labelpred_df['pred']))
print(classification_report(labelpred_df['label'], labelpred_df['pred']))
print("Accuracy", accuracy_score(labelpred_df['label'], labelpred_df['pred']))

testErr = label.filter(lambda lp: lp[0] != lp[1]).count() / float(dataset.count())
print('Test Error = ' + str(testErr))