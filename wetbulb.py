"""
Code for reproducing the wetbulb temperature plot in the paper 'Reclassifying
Lethal Heat'

@author: robert-edwin-rouse
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from apollo import mechanics as ma
from apollo import thermo as th


### Set plotting style parameters
ma.textstyle()


### Set global model parameters
np.random.seed(42)


### Open datafile
filename = 'Lethal_Heat_90th.csv'
df = pd.read_csv(filename)
df = df.fillna(value=0)
df['DocuMortHetWav'][df['DocuMortHetWav']>0] = 1


### Temperature and wet bulb temperature calculations
df['Wetbulb'] = th.wet_bulb(df['Max_T'], df['Mean_RH'])
df['WB35'] = np.where(df['Wetbulb']>=35, 1, 0)
df['WB30'] = np.where(df['Wetbulb']>=30, 1, 0)
df['WB25'] = np.where(df['Wetbulb']>=25, 1, 0)
df['WB20'] = np.where(df['Wetbulb']>=20, 1, 0)
humd_arr = np.linspace(0, 99.9, 1000)
temp_arr20 = [th.threshold_wb(h, 20) for h in humd_arr]
temp_arr25 = [th.threshold_wb(h, 25) for h in humd_arr]
temp_arr30 = [th.threshold_wb(h, 30) for h in humd_arr]
temp_arr35 = [th.threshold_wb(h, 35) for h in humd_arr]


### Subsample the number of nonlethal heatwave events
subset_A = df[df['DocuMortHetWav'] == 1]
subset_B = df[df['DocuMortHetWav'] == 0].sample(len(subset_A)*2)


### Format and generate plot
colours = []
cmap = sn.color_palette("flare", as_cmap=True)
c_points = np.linspace(0,1,4)
for point in c_points:
    colours.append(cmap(point))

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(subset_A['Max_T'], subset_A['Mean_RH'], marker='o', s=48,
           c='indianred', label='Lethal')
ax.scatter(subset_B['Max_T'], subset_B['Mean_RH'], marker='x', s=48,
           c='cadetblue', label='Nonlethal')
ax.plot(temp_arr20, humd_arr, lw=4, c=colours[3], label='20°C WBT')
ax.plot(temp_arr25, humd_arr, lw=4, c=colours[2], label='25°C WBT')
ax.plot(temp_arr30, humd_arr, lw=4, c=colours[1], label='30°C WBT')
ax.plot(temp_arr35, humd_arr, lw=4, c=colours[0], label='35°C WBT')
ax.set_xlim([0, 50])
ax.set_ylim([0, 100])
ax.set_xlabel('Maximum Temperature (°C)')
ax.set_ylabel('Relative Humidity (%)')
ax.legend(loc="lower left")
plt.show()