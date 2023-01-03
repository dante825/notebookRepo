"""A PySpark test on the German Credit Risk data
As a practice to get a better understanding on PySpark
"""
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull, when, count, col
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

# Create a spark session
spark = SparkSession.builder \
    .appName("Spark Titanic") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "2g")\
    .config("spark.sql.shuffle.partitions", "2")\
    .config("spark.sql.adaptive.enabled", "true")\
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel('ERROR')

cred_df = spark.read.csv('./creditRiskRating/german_credit_data.csv', header=True)
cred_df = cred_df.drop('_c0')
# print(cred_df.printSchema())
# print(cred_df.show())
# print(cred_df.dtypes)

# Rename the column to my liking, no space
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
# print(categoricalColumns)

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

# Get the mapping of the indexed columns
# own = 0
# rent = 1
# free = 2
# print(cred_indexed_df.select('housing', 'housingIndex').distinct().orderBy('housingIndex', ascending=True).show())

# male = 0
# female = 1
# print(cred_indexed_df.select('sex', 'sexIndex').distinct().orderBy('sexIndex', ascending=True).show())

# little = 0
# NA = 1
# moderate = 2
# quite rich = 3
# rich = 4
# print(cred_indexed_df.select('savingAcc', 'savingAccIndex').distinct().orderBy('savingAccIndex', ascending=True).show())

# NA = 0
# little = 1
# moderate = 2
# rich = 3
# print(cred_indexed_df.select('checkingAcc', 'checkingAccIndex').distinct().orderBy('checkingAccIndex', ascending=True).show())

# car = 0
# radio/TV = 1
# furniture/equipment = 2
# business = 3
# education = 4
# repairs = 5
# domestic appliances = 6
# vacation/others = 7
# print(cred_indexed_df.select('purpose', 'purposeIndex').distinct().orderBy('purposeIndex', ascending=True).show())

# good = 0
# bad = 1
# print(cred_indexed_df.select('risk', 'riskIndex').distinct().orderBy('riskIndex', ascending=True).show())

cred_cleansed_df = cred_indexed_df.drop(*categoricalColumns)
# print(cred_cleansed_df.printSchema())
# print(cred_cleansed_df.show())

# Vector Assembler
required_features = cred_cleansed_df.columns[:-1]
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_df = assembler.transform(cred_cleansed_df)
# print(transformed_df.show())

#========================
# Feature correlation
# method = pearson, spearman
corr = Correlation.corr(transformed_df, column='features', method='spearman')
# print(corr.collect()[0]["pearson({})".format('features')].values)
matrix = corr.collect()[0][0]
corr_matrix = matrix.toArray().tolist()
cor_matrix_df = pd.DataFrame(data=corr_matrix, columns=required_features, index=required_features)
cor_matrix_df.style.background_gradient(cmap='coolwarm').set_precision(2)
print(cor_matrix_df)

plt.figure(figsize=(16,6))
sns.heatmap(cor_matrix_df, xticklabels=cor_matrix_df.columns.values, 
            yticklabels=cor_matrix_df.columns.values, cmap='Greens', annot=True)
plt.title('Features correlation')
plt.show()