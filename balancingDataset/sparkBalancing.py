"""Rebalancing the dataset with spark
Encountered a situation at work where the data size is around 2m records
Applying SMOTE on the data with pandas took ages
Let's see if using Spark would be any better
"""
import random
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.sql.functions import rand, col, isnull, when, concat, substring, lit, udf, lower, \
sum as ps_sum, count as ps_count, row_number, array, create_map, struct
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.window import Window
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer, VectorAssembler, BucketedRandomProjectionLSH, VectorSlicer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Create a spark session
spark = SparkSession.builder \
    .appName("SparkCreditRisk") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "2g")\
    .config("spark.sql.shuffle.partitions", "2")\
    .config("spark.sql.adaptive.enabled", "true")\
    .config("spark.jars.packages", "mjuez:approx-smote:1.1.2")\
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel('ERROR')



#===========================
# Spark SMOTE Oversampling
#===========================
# For categorical columns, must take its stringIndexed form 
# (smote should be after string indexing, default by frequency)

def vectorizerFunction(dataInput, TargetFieldName):
    if(dataInput.select(TargetFieldName).distinct().count() != 2):
        raise ValueError("Target field must have only 2 distinct classes")
    columnNames = list(dataInput.columns)
    columnNames.remove(TargetFieldName)
    dataInput = dataInput.select((','.join(columnNames)+','+TargetFieldName).split(','))
    assembler=VectorAssembler(inputCols = columnNames, outputCol = 'features')
    pos_vectorized = assembler.transform(dataInput)
    vectorized = pos_vectorized.select('features',TargetFieldName).withColumn('label',pos_vectorized[TargetFieldName]).drop(TargetFieldName)
    return vectorized

def SmoteSampling(vectorized, k = 5, minorityClass = 1, majorityClass = 0, percentageOver = 200, percentageUnder = 100):
    if(percentageUnder > 100|percentageUnder < 10):
        raise ValueError("Percentage Under must be in range 10 - 100");
    if(percentageOver < 100):
        raise ValueError("Percentage Over must be in at least 100");
    dataInput_min = vectorized[vectorized['label'] == minorityClass]
    dataInput_maj = vectorized[vectorized['label'] == majorityClass]
    feature = dataInput_min.select('features')
    feature = feature.rdd
    feature = feature.map(lambda x: x[0])
    feature = feature.collect()
    feature = np.asarray(feature)
    nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feature)
    neighbours =  nbrs.kneighbors(feature)
    gap = neighbours[0]
    neighbours = neighbours[1]
    min_rdd = dataInput_min.drop('label').rdd
    pos_rddArray = min_rdd.map(lambda x : list(x))
    pos_ListArray = pos_rddArray.collect()
    min_Array = list(pos_ListArray)
    newRows = []
    nt = len(min_Array)
    # make the division result an int
    nexs = percentageOver//100
    for i in range(nt):
        for j in range(nexs):
            neigh = random.randint(1,k)
            difs = min_Array[neigh][0] - min_Array[i][0]
            newRec = (min_Array[i][0]+random.random()*difs)
            newRows.insert(0,(newRec))
    newData_rdd = sc.parallelize(newRows)
    newData_rdd_new = newData_rdd.map(lambda x: Row(features = x, label = 1))
    new_data = newData_rdd_new.toDF()
    new_data_minor = dataInput_min.unionAll(new_data)
    new_data_major = dataInput_maj.sample(False, (float(percentageUnder)/float(100)))
    return new_data_major.unionAll(new_data_minor)

file_path = '/Users/kangwei/development/repo/notebookRepo/creditRiskRating/german_credit_data.csv'
cred_df = spark.read.format('csv').options(header='true',inferSchema='true').load(file_path).dropna()
cred_df = cred_df.drop('_c0')

cred_df = cred_df.withColumnRenamed('Age', 'age')\
                 .withColumnRenamed('Sex', 'sex')\
                 .withColumnRenamed('Job', 'job')\
                 .withColumnRenamed('Housing', 'housing')\
                 .withColumnRenamed('Saving accounts', 'savingAcc')\
                 .withColumnRenamed('Checking account', 'checkingAcc')\
                 .withColumnRenamed('Credit amount', 'creditAmt')\
                 .withColumnRenamed('Duration', 'duration')\
                 .withColumnRenamed('Purpose', 'purpose')\
                 .withColumnRenamed('Risk', 'risk')

# Some int columns are in string format, convert the columns to numeric
cred_df = cred_df.select(col('age').cast('int'), 
                        col('sex'), 
                        col('job').cast('int'),
                        col('housing'),
                        col('savingAcc'), col('checkingAcc'),
                        col('creditAmt').cast('int'),
                        col('duration').cast('int'),
                        col('purpose'), col('risk'))

# Create a list of the columns that are string typed
categoricalColumns = [item[0] for item in cred_df.dtypes if item[1].startswith('string')]

# Define a list of stages in your pipeline. The string indexer will be one stage
stages = []
# Iterate through all categorical values
for categoricalCol in categoricalColumns:
    # Create a string indexer for those categorical values and assign a new name including the word 'Index'
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    # Append the string Indexer to our list of stages
    stages += [stringIndexer]

#Create the pipeline. Assign the satges list to the pipeline key word stages
pipeline = Pipeline(stages = stages)
#fit the pipeline to our dataframe
pipelineModel = pipeline.fit(cred_df)
#transform the dataframe
cred_indexed_df = pipelineModel.transform(cred_df)
cred_cleansed_df = cred_indexed_df.drop(*categoricalColumns)
# print(cred_cleansed_df.printSchema())

# cred_cleansed_df = cred_cleansed_df.select(col('age').cast('int'), 
#                         col('job').cast('int'),
#                         col('creditAmt'),
#                         col('duration'),
#                         col('sexIndex').cast('int'),
#                         col('housingIndex').cast('int'),
#                         col('savingAccIndex').cast('int'),
#                         col('checkingAccIndex').cast('int'),
#                         col('purposeIndex').cast('int'),
#                         col('riskIndex').cast('int'))

rebalance_df = SmoteSampling(vectorizerFunction(cred_cleansed_df, 'riskIndex'), k = 2, 
minorityClass = 0, majorityClass = 1, percentageOver = 200, percentageUnder = 5)
print(rebalance_df.groupBy('label').count().show())

