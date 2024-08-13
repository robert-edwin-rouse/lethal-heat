"""
Trains and tests the random forest model for classifying heatwaves as either
lethal or nonlethal, using a precompiled database.

@author: robertrouse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cartopy.crs as ccrs
from apollo import mechanics as ma


### Set plotting style parameters
ma.textstyle()


### Load results for visualisation
input_file = 'Lethal_Heat_90th_Results.csv'
df = pd.read_csv(input_file)


### Split out features and presentation labels
features = ['Max_T','Mean_H','Mean_W','Rate_30','Rate_90',
            'Rate_180','Adaptive_T','Avg_Age','Gradient','Mean BMI',]
labels = ['Max T','Mean RH','Mean WS','30 ∆T','90 ∆T',
          '180 ∆T','Ada T','Mean Age','Pop Grad','Mean BMI']


### Feature correlation heat map
corr_df = df[features]
corr_df.columns = labels
corrMatrix = corr_df.corr()
corrMatrix = corrMatrix.round(decimals=2)
mask = np.triu(np.ones_like(corrMatrix.corr()), k=1)
fig, ax = plt.subplots(figsize=(16, 12))
sn.heatmap(corrMatrix, vmin= -0.6, vmax=1.01, annot=True, cmap='flare', mask=mask)
ax.set_xticklabels(labels, rotation=-45, ha='left')
plt.show()


### Result categorisation
df['Observed +'] = (df['Documented'] > 0).astype(int)
df['Observed -'] = (df['Documented'] == 0).astype(int)
df['Predicted +'] = (df['Predicted'] > 0).astype(int)
df['Predicted -'] = (df['Predicted'] == 0).astype(int)
df['True +'] = (df['Documented'] > 0) & (df['Predicted'] > 0).astype(int)
df['True -'] = (df['Documented'] == 0) & (df['Predicted'] == 0).astype(int)
df['False +'] = (df['Documented'] == 0) & (df['Predicted'] > 0).astype(int)
df['False -'] = (df['Documented'] > 0) & (df['Predicted'] == 0).astype(int)
retain = ['Latitude','Longitude','Observed +','Observed -',
          'Predicted +','Predicted -','True +','True -','False +','False -']
df_compressed = df[retain].groupby(retain[0:2],as_index=False).sum()


def dotogram(xf, data, lons, lats, colours, scaling, num):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = plt.subplot(projection = ccrs.PlateCarree())
    ax.coastlines(resolution='50m', alpha=0.5)
    ax.gridlines(draw_labels=False)
    ax.set_xlim(lons[0], lons[1])
    ax.set_ylim(lats[0], lats[1])
    f = lambda x: (x**scaling[0])*scaling[1]
    g = lambda y: (y/scaling[1])**(1/scaling[0])
    sc = ax.scatter(xf['Longitude'], xf['Latitude'], s=f(xf[data]),
                    c=colours[0], alpha=0.65, edgecolors=colours[1], lw=2.5)
    ax.legend(*sc.legend_elements("sizes", num=num, func=g), labelspacing=1.2,
              loc='upper left')
    plt.show()


dotogram(df_compressed, 'Observed +', [-20, 40], [30, 70],
                  ['indianred','crimson'], [2,20], 4)
dotogram(df_compressed, 'Predicted +', [-20, 40], [30, 70],
                  ['indianred','crimson'], [2,20], 5)
dotogram(df_compressed, 'False -', [-20, 40], [30, 70],
                  ['indianred','crimson'], [2,20], 5)

dotogram(df_compressed, 'Observed -', [-20, 40], [30, 70],
                  ['cadetblue','steelblue'], [2,0.025], 5)
dotogram(df_compressed, 'Predicted -', [-20, 40], [30, 70],
                  ['cadetblue','steelblue'], [2,0.025], 5)
dotogram(df_compressed, 'False +', [-20, 40], [30, 70],
                  ['cadetblue','steelblue'], [2,10], 5)

dotogram(df_compressed, 'Observed +', [60, 180], [0, 80],
                  ['indianred','crimson'], [2,10], 4)
dotogram(df_compressed, 'Predicted +', [60, 180], [0, 80],
                  ['indianred','crimson'], [2,10], 5)
dotogram(df_compressed, 'False -', [60, 180], [0, 80],
                  ['indianred','crimson'], [2,10], 5)

dotogram(df_compressed, 'Observed -', [60, 180], [0, 80],
                  ['cadetblue','steelblue'], [2,0.025], 5)
dotogram(df_compressed, 'Predicted -', [60, 180], [0, 80],
                  ['cadetblue','steelblue'], [2,0.025], 5)
dotogram(df_compressed, 'False +', [60, 180], [0, 80],
                  ['cadetblue','steelblue'], [2,10], 5)

dotogram(df_compressed, 'Observed +', [-170, -35], [0, 90],
                  ['indianred','crimson'], [2,10], 4)
dotogram(df_compressed, 'Predicted +', [-170, -35], [0, 90],
                  ['indianred','crimson'], [2,10], 4)
dotogram(df_compressed, 'False -', [-170, -35], [0, 90],
                  ['indianred','crimson'], [2,10], 5)

dotogram(df_compressed, 'Observed -', [-170, -35], [0, 90],
                  ['cadetblue','steelblue'], [2,0.025])
dotogram(df_compressed, 'Predicted -', [-170, -35], [0, 90],
                  ['cadetblue','steelblue'], [2,0.025])
dotogram(df_compressed, 'False +', [-170, -35], [0, 90],
                  ['cadetblue','steelblue'], [2,10])
