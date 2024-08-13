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


### Open datafile
input_file = 'Lethal_Heat_90th_Processed.csv'
df = pd.read_csv(input_file)
df = df.fillna(value=0)
df['Documented'] = (df['Documented'] > 0).astype(int)


### Test/train split
trn_df = df.sample(frac=0.6, random_state=32)


### Input/output variables
features = ['Max_T','Mean_H','Mean_W','Rate_30','Rate_90',
            'Rate_180','Adaptive_T','Avg_Age','Gradient','Mean BMI',]
labels = ['Max T','Mean RH','Mean WS','30 ∆T','90 ∆T',
          '180 ∆T','Ada T','Mean Age','Pop Grad','Mean BMI']
targets = ['Documented']
xspace = ma.featurelocator(df, features)
yspace = ma.featurelocator(df, targets)


### Dataframes to Arrays
trn_set = trn_df.to_numpy()
x_train = trn_set[:,xspace].reshape(len(trn_set), len(xspace)).astype(float)
y_train = trn_set[:,yspace].reshape(len(trn_set), ).astype(float)


### SMOTE
sm = SMOTE(random_state=16, k_neighbors=5)
x_res, y_res = sm.fit_resample(x_train, y_train)


### Random Forest Setup
model = sle.RandomForestClassifier(n_estimators=200, max_depth=50)
model.fit(x_res, y_res)


### Prediction
subset = df[~df.isin(trn_df)].dropna()
val_df = subset.sample(frac=0.5, random_state=8)
tst_df = subset[~subset.isin(val_df)].dropna()
frames = [trn_df,val_df,tst_df]
modes = ['Training','Validation','Test']
for i in range(len(frames)):
    frame = frames[i]
    text = modes[i]
    x = frame.to_numpy()[:,xspace].reshape(len(frame),len(xspace)).astype(float)
    y = model.predict(x)
    frame['Predicted'] = y
    CM = slm.confusion_matrix(frame['Documented'], frame['Predicted'])
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1] 
    print('- - - - - - - - - - - - - - -')
    print(text + ' Accuracy: ' + str(slm.accuracy_score(frame['Documented'],
                                                        frame['Predicted'])))
    print(text + ' F1 score: ' + str(slm.f1_score(frame['Documented'],
                                                  frame['Predicted'])))
    print(text + ' Precision:' + str(TP/(TP+FP)))
    print(text + ' Recall:' + str(TP/(TP+FN)))


### Output results for visualisation
output_file = 'Lethal_Heat_90th_Results.csv'
results_df = pd.concat([val_df, tst_df])
results_df.to_csv(output_file)