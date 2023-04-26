import findspark
findspark.init()
findspark.find()
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

conf = pyspark.SparkConf().setAppName('wine').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

df = spark.read.format("csv").load("s3://wine123/TrainingDataset.csv" , header=True, sep=";")
df.printSchema()
df.show()

features = np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select('label').collect())

VectorAssembler = VectorAssembler(inputCols=df.columns[1:-1], outputCol='features')
df_tr = VectorAssembler.transform(df)
df_tr = df_tr.select(['features', 'label'])

def to_labeled_point(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)

dataset = to_labeled_point(sc, features, label)

training, test = dataset.randomSplit([0.7, 0.3], seed=11)

model = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},numTrees=21, featureSubsetStrategy="auto", impurity='gini', maxDepth=30, maxBins=32)

preds = model.predict(test.map(lambda x: x.features))
label = test.map(lambda lp: lp.label).zip(preds)

label_df = label.toDF()
labelpred = label.toDF(["label", "pred"])
labelpred.show()
labelpred_df = labelpred.toPandas()

F1score = f1_score(labelpred_df['label'], labelpred_df['pred'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'], labelpred_df['pred']))
print(classification_report(labelpred_df['label'], labelpred_df['pred']))
print("Accuracy", accuracy_score(labelpred_df['label'], labelpred_df['pred']))

testErr = label.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())
print('Test Error = ' + str(testErr))

model.save(sc, 's3://wine123/train.model')
