"""
Trains and tests the random forest model for classifying heatwaves as either
lethal or nonlethal, using a precompiled database.

@author: robertrouse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import seaborn as sn
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
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
CM = corr_df.corr()
CM = CM.round(decimals=2)
mask = np.triu(np.ones_like(CM.corr()), k=1)
fig, ax = plt.subplots(figsize=(16, 12))
sn.heatmap(CM, vmin= -0.6, vmax=1.01, annot=True, cmap='flare', mask=mask)
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
df_bubble = df[retain].groupby(retain[0:2],as_index=False).sum()


### Regional bubble plots
def heat_bubble(df, data, lons, lats, colours, scaling, num):
    proj = ccrs.PlateCarree()    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection':proj})
    f = lambda x: (x**scaling[0])*scaling[1]
    g = lambda y: (y/scaling[1])**(1/scaling[0])
    sc = ax.scatter(df['Longitude'], df['Latitude'], s=f(df[data]),
                    c=colours[0], alpha=0.65, edgecolors=colours[1], lw=2.5)
    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], proj)  
    ax.coastlines(resolution='50m', alpha=0.5)
    gl = ax.gridlines(crs=proj, draw_labels=True, )
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlocator = mtk.MaxNLocator(5)
    gl.ylocator = mtk.MaxNLocator(5)
    gl.top_labels = False
    gl.right_labels = False
    ax.legend(*sc.legend_elements("sizes", num=num, func=g), labelspacing=2)
    plt.show()

cset = [['indianred','crimson'],['cadetblue','steelblue']]

heat_bubble(df_bubble, 'Observed +', [-20, 40], [30, 70], cset[0], [2,10], 5)
heat_bubble(df_bubble, 'Predicted +', [-20, 40], [30, 70], cset[0], [2,10], 6)
heat_bubble(df_bubble, 'False -', [-20, 40], [30, 70], cset[0], [2,10], 6)
heat_bubble(df_bubble, 'Observed -', [-20, 40], [30, 70], cset[1], [2,0.005], 5)
heat_bubble(df_bubble, 'Predicted -', [-20, 40], [30, 70], cset[1], [2,0.005], 5)
heat_bubble(df_bubble, 'False +', [-20, 40], [30, 70], cset[1], [2,10], 5)

heat_bubble(df_bubble, 'Observed +', [60, 180], [0, 80], cset[0], [2,10], 5)
heat_bubble(df_bubble, 'Predicted +', [60, 180], [0, 80], cset[0], [2,10], 6)
heat_bubble(df_bubble, 'False -', [60, 180], [0, 80], cset[0], [2,10], 5)
heat_bubble(df_bubble, 'Observed -', [60, 180], [0, 80], cset[1], [2,0.01], 5)
heat_bubble(df_bubble, 'Predicted -', [60, 180], [0, 80], cset[1], [2,0.01], 5)
heat_bubble(df_bubble, 'False +', [60, 180], [0, 80], cset[1], [2,10], 5)
