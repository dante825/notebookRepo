from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull, when, count, col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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
# findspark.find()

df = spark.read.format("csv").option('header', True)\
    .load('/Users/kangwei/development/repo/notebookRepo/titanic/titanicData/train.csv')

dataset = df.select(col('Survived').cast('float'),
            col('Pclass').cast('float'),
            col('Sex'), 
            col('Age').cast('float'),
            col('Fare').cast('float'),
            col('Embarked'))

# To check number of null values in columms
dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()
# Replace null values in columns
dataset = dataset.replace('?', None).dropna(how='any')

# StringIndexer also accepts arrays
dataset = StringIndexer(inputCols=['Sex', 'Embarked'], outputCols=['Gender', 'Boarded'], handleInvalid='keep', 
stringOrderType='frequencyDesc').fit(dataset).transform(dataset)
# Drop unnecesary columns
dataset = dataset.drop('Sex')
dataset = dataset.drop('Embarked')

required_features = ['Pclass', 'Age', 'Fare', 'Gender', 'Boarded']
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_data = assembler.transform(dataset)

# Modeling
(training_data, test_data) = transformed_data.randomSplit([0.8, 0.2])

# Random Forest
rf = RandomForestClassifier(labelCol='Survived', featuresCol='features', maxDepth=5, seed=42)
model = rf.fit(training_data)
predictions = model.transform(test_data)

# Logistic Regression
# family options: auto, binomial, multinomial
# lr = LogisticRegression(labelCol='Survived', featuresCol='features', 
#                         maxIter=10, regParam=0.3, elasticNetParam=0.8, family='binomial')
# model = lr.fit(training_data)
# print(f"Coefficient: {model.coefficients}")
# print(f"Intercept: {model.intercept}")
# predictions = model.transform(test_data)

# Linear SVC (SVM)
# svm = LinearSVC(labelCol='Survived', featuresCol='features', maxIter=10, regParam=0.01)
# model = svm.fit(training_data)
# predictions = model.transform(test_data)


# Model Evaluation
evaluator = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
# print(f'Test accuracy: {accuracy}')

trainingSummary = model.summary
# ROC curve
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'], roc['TPR'])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')
plt.show()

# Precision-Recall Curve
pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'], pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.show()

# metricName options: accuracy, f1, precisionByLabel, recallByLabel
eval_acc = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='accuracy')
eval_pre = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='precisionByLabel')
eval_rec = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='recallByLabel')
eval_f1 = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='f1')
accuracy = eval_acc.evaluate(predictions)
precision = eval_pre.evaluate(predictions)
recall = eval_rec.evaluate(predictions)
f1 = eval_f1.evaluate(predictions)

print(f'Accracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'f1 score: {f1:.2f}')

# Alternative: using sklearn confusion matrix
# if the data size is huge, sklearn may not be able to handle
y_true = predictions.select('Survived').collect()
y_pred = predictions.select('prediction').collect()

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(cm, display_labels=['Not Survived', 'Survived'])
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Confusion Matrix from sklearn')
plt.show()

# Pyspark way to get confusion matrix but the display part is using the sklearn ConfusionMatrixDisplay
#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = predictions.select(['prediction','Survived']).withColumn('Survived', F.col('Survived').cast(FloatType())).orderBy('prediction')
# select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','Survived'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
cm2 = metrics.confusionMatrix().toArray()
disp2 = ConfusionMatrixDisplay(cm2)
disp2.plot()
plt.title('Confusion Matrix from PySpark')
plt.show()

