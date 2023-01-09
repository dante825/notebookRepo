"""This is a continuance of the credit risk model.
This scipt would rebalance the training data by oversampling the minority class.
Since the oversampling method used here: RandomOverSampler is better with pandas,
thus this script would use pandas
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

cred_df = pd.read_csv('./creditRiskRating/german_credit_data.csv', header=0, index_col=0)
riskIndEncoder = LabelEncoder()
cred_df['riskIndex'] = riskIndEncoder.fit_transform(cred_df['Risk'])
# Simply fillna because randomForestClassifier can't handle na
cred_df = cred_df.fillna({'Housing': 'NA', 'Saving accounts': 'NA', 'Checking account': 'NA', 'Purpose': 'NA'})
ordEncoder = OrdinalEncoder()
cred_df[['sexIndex', 'housingIndex', 'savingAccInd', 'checkingAccInd', 'purposeInd']] = \
    ordEncoder.fit_transform(cred_df[['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']])

# print(ordEncoder.inverse_transform(cred_df[['sexIndex', 'housingIndex', 'savingAccInd', 'checkingAccInd', 'purposeInd']]))
# print(cred_df.head())
data = cred_df[['sexIndex', 'housingIndex', 'savingAccInd', 'checkingAccInd', 'purposeInd', 'riskIndex']]
# 0 -> 300
# 1 -> 700
# print(data.groupby(data['riskIndex'])['riskIndex'].count())
# print(data.head())


X = data.drop('riskIndex', axis=1)
y = data['riskIndex']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#=====================================
# Rebalancing with RandomOverSampler
#=====================================
# oversampler = RandomOverSampler(random_state=42)
# X_res, y_res = oversampler.fit_resample(X_train, y_train)
# print(f"Training target statistics: {Counter(y_res)}")
# print(f"Testing target staticstics: {Counter(y_test)}")

#==============================
# Rebalancing with SMOTE
#==============================
oversampler = SMOTE(k_neighbors=2)
X_res, y_res = oversampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y_test)}")


#==============
# Modeling
#==============
clf = RandomForestClassifier(max_depth=2, random_state=42)
clf.fit(X_res, y_res)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Confusion Matrix from sklearn')
plt.show()