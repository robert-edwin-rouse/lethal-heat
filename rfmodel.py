"""
Trains and tests the random forest model for classifying heatwaves as either
lethal or nonlethal, using a precompiled database.

@author: robertrouse
"""


import pandas as pd
import sklearn.ensemble as sle
import sklearn.metrics as slm
from imblearn.over_sampling import SMOTE
from apollo import mechanics as ma


### Set plotting style parameters
ma.textstyle()


### Open datafile
input_file = 'Lethal_Heat_90th.csv'
df = pd.read_csv(input_file)
df = df.fillna(value=0)
df['DocuMortHetWav'][df['DocuMortHetWav']>0] = 1


### Test/train split
subset = df.sample(frac=0.4, random_state=42)
val_df = subset.sample(frac=0.5, random_state=42)
tst_df = subset[~subset.isin(val_df)].dropna()
trn_df = df[~df.isin(subset)].dropna()


### Input/output variables
features = ['Max_T',
            'Mean_H',
            'Mean_W',
            'Rate_30',
            'Rate_90',
            'Rate_180',
            'Adaptive_T',
            'Avg_Age',
            'Gradient',
            'Mean BMI',]
labels = ['Max T','Mean RH','Mean WS','30 ∆T','90 ∆T',
          '180 ∆T','Ada T','Mean Age','Pop Grad','Mean BMI']
targets = ['DocuMortHetWav']
xspace = ma.featurelocator(df, features)
yspace = ma.featurelocator(df, targets)


### Dataframes to Arrays
trn_set = trn_df.to_numpy()
x_train = trn_set[:,xspace].reshape(len(trn_set), len(xspace)).astype(float)
y_train = trn_set[:,yspace].reshape(len(trn_set), ).astype(float)


### SMOTE
sm = SMOTE(random_state=42, k_neighbors=5)
x_res, y_res = sm.fit_resample(x_train, y_train)


### Random Forest Setup
model = sle.RandomForestClassifier(n_estimators=200, max_depth=50)
model.fit(x_res, y_res)


### Prediction
tst_set = df.to_numpy()
x_test = tst_set[:,xspace].reshape(len(tst_set), len(xspace)).astype(float)
y_pred = model.predict(x_test)
df['Predicted'] =  y_pred
df_test = df[~df.isin(trn_df)].dropna()
df_train = df[df.isin(trn_df)].dropna()
print('Training Accuracy: ' + str(slm.accuracy_score(df['DocuMortHetWav'], df['Predicted'])))
print('Training F1 score: ' + str(slm.f1_score(df['DocuMortHetWav'], df['Predicted'])))
CM = slm.confusion_matrix(df_test['DocuMortHetWav'], df_test['Predicted'])
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
print('Test Accuracy: ' + str(slm.accuracy_score(df_test['DocuMortHetWav'], df_test['Predicted'])))
print('Test F1 score: ' + str(slm.f1_score(df_test['DocuMortHetWav'], df_test['Predicted'])))
print('Test Precision:' + str(TP/(TP+FP)))
print('Test Recall:' + str(TP/(TP+FN)))


### Output results for visualisation
output_file = 'Lethal_Heat_90th_Results.csv'
df.to_csv(output_file)