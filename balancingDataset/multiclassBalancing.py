import pandas as pd
import numpy as np
import warnings
from sklearn.utils import class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from scikitplot.metrics import plot_roc, plot_precision_recall
import matplotlib.pyplot as plt

df = pd.read_csv('./balancingDataset/glass.csv')
# print(df.head())

features = []
for feature in df.columns:
    if feature != 'target':
        features.append(feature)
        
# print(features)

X = df[features]
y = df['target']

#======================
# Train test split
#======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# print(type(y_train))
# print(y_train)

count = y_train.value_counts()
count.plot.bar()
plt.title('target distribution without rebalance')
plt.ylabel('Number of records')
plt.xlabel('Target class')
plt.show()

#================
# Modeling
#================
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Plot metrics
plot_roc(y_test, y_score)
plt.title('ROC curve without rebalancing')
plt.show()

plot_precision_recall(y_test, y_score)
plt.title('PR curve without rebalancing')
plt.show()

#=============
# Sampling
#=============

n_samples = int(count.median())
# print(n_samples)
warnings.filterwarnings('ignore')

# A utility function, which receives as input the dataset, the threshold (`n_samples`) 
# and the involved classes (`majority` or `minority`). 
# This function returns a `dict` which contains the number of desired samples for each class 
# belonging to the involved classes.
def sampling_strategy(X, y, n_samples, t='majority'):
    target_classes = ''
    if t == 'majority':
        target_classes = y.value_counts() > n_samples
    elif t == 'minority':
        target_classes = y.value_counts() < n_samples
        
    tc = target_classes[target_classes == True].index
    sampling_strategy = {}
    for target in tc:
        sampling_strategy[target] = n_samples
    return sampling_strategy

under_sampler = ClusterCentroids(sampling_strategy=sampling_strategy(X_train, y_train, n_samples,t='majority'))
X_under, y_under = under_sampler.fit_resample(X_train, y_train)

count = y_under.value_counts()
count.plot.bar()
plt.ylabel('Number of records')
plt.xlabel('Target classes')
plt.title('Target distribution after under sampling')
plt.show()

over_sampler = SMOTE(sampling_strategy=sampling_strategy(X_under, y_under, n_samples, t='minority'), k_neighbors=2)
X_bal, y_bal = over_sampler.fit_resample(X_under, y_under)

count = y_bal.value_counts()
count.plot.bar()
plt.ylabel('Number of records')
plt.xlabel('Target classes')
plt.title('Target distribution after over sampling')
plt.show()

#=================================
# Modelling with balanced dataset
#=================================
model = KNeighborsClassifier()
model.fit(X_bal, y_bal)
y_score = model.predict_proba(X_test)

y_pred = model.predict(X_test)

plot_roc(y_test, y_score)
plt.title('ROC curve with balanced dataset')
plt.show()

plot_precision_recall(y_test, y_score)
plt.title('PR curve with balanced dataset')
plt.show()

#=============================
# Manipulating class weights
#=============================
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
weights = dict(zip(classes, cw))

model = DecisionTreeClassifier(class_weight=weights)
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)

y_pred = model.predict(X_test)

plot_roc(y_test, y_score)
plt.title('ROC curve, class weight')
plt.show()

plot_precision_recall(y_test, y_score)
plt.title('PR curve, class weight')
plt.show()