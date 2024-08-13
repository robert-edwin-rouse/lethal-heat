"""
Compiles the heatwave database required for modelling heat lethality based on
meteorological variables, using era5 data retrieval script and labelledheatwave
data handling functions from the apollo library.

@author: robertrouse
"""

import numpy as np
import xarray as xr
import pandas as pd
import geopy as ge
from apollo import era5 as er


### 
my_email = 'user1234@somewhere.ac.uk'
geolocator = ge.geocoders.Nominatim(user_agent=my_email)




### Specify meteorological variables and spatiotemporal ranges
area = ['60.00/-8.00/48.00/4.00']
yyyy = [str(y) for y in range(1979,2021,1)]
mm = [str(m) for m in range(1,13,1)]
dd = [str(d) for d in range(1,32,1)]
hh = [str(t).zfill(2) + ':00' for t in range(0, 24, 1)]
met = ['total_precipitation','temperature','u_component_of_wind',
       'v_component_of_wind','relative_humidity',
]


### Download meteorological variable sets from Copernicus Data Store
for yy in yyyy:
    filename = 'Rainfall_' + str(yy)
    rain_query = er.query(filename, 'reanalysis-era5-single-levels', met[0],
                          area, yy, mm, dd, hh)
    rain_data = er.era5(rain_query).download()
    er.aggregate_mean(str(rain_query['file_stem']) + '.nc',
                      str(rain_query['file_stem']) + '_aggregated.nc')
full_rain_data = xr.open_mfdataset('Rainfall_*_aggregated.nc', concat_dim='time')
full_rain_data.to_netcdf(path='Rainfall.nc')
pressure_query = er.query('Pressure','reanalysis-era5-pressure-levels',
                          met[1:5], area, yyyy, mm, dd, '12:00', ['1000'])
pressure_data = er.era5(pressure_query).download()
soil_query = er.query('Soil_Moisture','reanalysis-era5-land', met[5:],
                      area, yyyy, mm, dd, '12:00')
soil_data = er.era5(soil_query).download()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:31:35 2023

@author: robertrouse
"""

import numpy as np
import pandas as pd
import datetime as dt
import geopy as ge
import requests as rq
import matplotlib.pyplot as plt
import cdsapi
import xarray as xr
import difflib
import country_converter as coco
import statistics
import math


### Global style parameters
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
geolocator = ge.geocoders.Nominatim(user_agent='rer44@cam.ac.uk')


### Define data handling functions
def geolocate(latitude, longitude):
    """
    Parameters
    ----------
    latitude : location latitude
    longitude : location longitude
        
    Returns
    -------
    The city closest to the latitude and longitude coordinates
    """
    location = geolocator.reverse(str(latitude)+","+str(longitude),
                                   addressdetails=True, language='en')
    address = location.raw['address']
    city = address.get('city', '')
    country = address.get('country', '')
    return city, country

def missing_column_match(df1, df2, targets, columns):
    df1 = df1.merge(df2, on=targets, how="left")
    for col in columns:
        x_col = str(col)+'_x'
        y_col = str(col)+'_y'
        df1[y_col] = df1[y_col].fillna(df1[x_col])
        df1.drop([x_col],inplace=True,axis=1)
        df1.rename(columns={y_col:col},inplace=True)
    return df1

def grid_square(lat, lon):
    latmax = max(math.ceil(lat), math.floor(lat))
    latmin = min(math.ceil(lat), math.floor(lat))
    lonmax = max(math.ceil(lon), math.floor(lon))
    lonmin = min(math.ceil(lon), math.floor(lon))
    grid_square_actual = [latmax, lonmin, latmin, lonmax]
    return grid_square_actual

def fuzzy_match(x, scan_array, threshold=0.9):
    aliases = difflib.get_close_matches(x, scan_array, len(scan_array), threshold)
    if not aliases:
        return x
    else:
        closest = statistics.mode(aliases)
        return closest

def UN_API_Call(relative_path:str, topic_list:bool = False, simple=False) -> pd.DataFrame:
        base_url = "https://population.un.org/dataportalapi/api/v1" 
        target = base_url + relative_path # Query string parameters may be appended here or directly in the provided relative path
        response = rq.get(target)
        j = response.json()
        if simple==True:
            try:
                df = pd.json_normalize(j['data'])
            except:
                return None
        else:
            try:
                df = pd.json_normalize(j['data'])
                while j['nextPage'] is not None:
                    response = rq.get(j['nextPage'])
                    j = response.json()
                    cache = pd.json_normalize(j['data'])
                    df = df.concat(cache)
            except:
                if topic_list:
                    df = pd.json_normalize(j, 'indicators')
                else:
                    df = pd.DataFrame(j)
        return(df)

def percent_pop_by_sex(df, column, category, target):
    split = df[df[column]==category]
    cache = split[target].values
    total = sum(cache)
    percent = cache/total
    return percent, total 

def relative_from_specific_humidity(p, q, t, t0):
    rh = 0.263*p*q*(np.exp(17.67*(t-t0)/(t-29.65)))**(-1)
    return rh

def antecedent_adaptation(array, hw_start, years=10):
    cache = []
    for y in range(years):
        past_end = hw_start - dt.timedelta(days=(365*y))
        past_start = past_end - dt.timedelta(days=30)
        t = np.mean(array.sel(time=slice(past_start, past_end))).values.item()
        cache.append(t)
    average = np.mean(cache)
    return average

def ERA5_retrieval(country, city, grid_square, pressure_level=1000):
    era5_file = str(country)+ '_' + str(city) + '_era5_heatwave.nc'
    c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type":   "reanalysis",
            "format":         "netcdf",
            "variable":       ['relative_humidity', 'temperature', 'u_component_of_wind',
            'v_component_of_wind'],
            "pressure_level": [pressure_level],
            "area":           grid_square,
            "year":           ['1969','1970','1971','1972','1973',
                                '1974','1975','1976','1977','1978',
                                '1979','1980','1981','1982','1983',
                                '1984','1985','1986','1987','1988',
                                '1989','1990','1991','1992','1993',
                                '1994','1995','1996','1997','1998',
                                '1999','2000','2001','2002','2003',
                                '2004','2005','2006','2007','2008',
                                '2009','2010','2011','2012','2013',
                                '2014','2015','2016','2017','2018',
                                '2019'],
            "month":          ["01","02","03","04","05","06","07","08","09","10","11","12"],
            "day":            ["01","02","03","04","05","06","07","08","09","10","11",
                              "12","13","14","15","16","17","18","19","20","21","22",
                              "23","24","25","26","27","28","29","30","31"],
            "time":           "12"
        }, era5_file)

def ERA_heatwave_extraction(lon, lat, country, city, gridsquare, start, end, hour_base=12):
    era5_file = str(country)+ '_' + str(city) + '_era5_heatwave.nc'
    try:
        array = xr.open_dataset(era5_file)
    except:
        ERA5_retrieval(country, city, grid_square, pressure_level=1000)
        array = xr.open_dataset(era5_file)
    cache = array.interp(coords={'longitude':lon,'latitude':lat},
                         method='nearest')
    hour_fragment = dt.time(12)
    hw_start = dt.datetime.combine(start, hour_fragment)
    hw_end = dt.datetime.combine(end, hour_fragment)
    pre_30 = hw_start - dt.timedelta(days=30)
    pre_90 = hw_start - dt.timedelta(days=90)
    pre_180 = hw_start - dt.timedelta(days=180)
    max_t = np.max(cache.sel(time=slice(hw_start, hw_end)).t).values.item()
    mean_r = np.mean(cache.sel(time=slice(hw_start, hw_end)).r).values.item()
    mean_u = np.mean(cache.sel(time=slice(hw_start, hw_end)).u).values.item()
    mean_v = np.mean(cache.sel(time=slice(hw_start, hw_end)).v).values.item()
    mean_w = (mean_u**2 + mean_v**2)**(1/2)
    ante_30 = np.mean(cache.sel(time=slice(pre_30, hw_start)).t).values.item()
    ante_90 = np.mean(cache.sel(time=slice(pre_90, hw_start)).t).values.item()
    ante_180 = np.mean(cache.sel(time=slice(pre_180, hw_start)).t).values.item()
    adaptation = antecedent_adaptation(cache.t, hw_start, years=10)
    return max_t, mean_r, mean_w, ante_30, ante_90, ante_180, adaptation

def ERA_heatwave_extraction_2(weather, lon, lat, country, city, gridsquare,
                              start, end, hour_base=12):
    cache = weather.interp(coords={'longitude':lon,'latitude':lat},
                         method='linear')
    hour_fragment = dt.time(0)
    hw_start = dt.datetime.combine(start, hour_fragment)
    hw_end = dt.datetime.combine(end, hour_fragment)
    pre_30 = hw_start - dt.timedelta(days=30)
    pre_90 = hw_start - dt.timedelta(days=90)
    pre_180 = hw_start - dt.timedelta(days=180)
    t = cache['2m_temperature']
    u = cache['10m_u_component_of_wind']
    v = cache['10m_v_component_of_wind']
    p = cache['surface_pressure']
    q = cache['specific_humidity']
    rh = relative_from_specific_humidity()
    max_t = np.max(cache.sel(time=slice(hw_start, hw_end)).t).values.item()    
    mean_r = np.mean(cache.sel(time=slice(hw_start, hw_end)).r).values.item()
    mean_u = np.mean(cache.sel(time=slice(hw_start, hw_end)).u).values.item()
    mean_v = np.mean(cache.sel(time=slice(hw_start, hw_end)).v).values.item()
    mean_w = (mean_u**2 + mean_v**2)**(1/2)
    ante_30 = np.mean(cache.sel(time=slice(pre_30, hw_start)).t).values.item()
    ante_90 = np.mean(cache.sel(time=slice(pre_90, hw_start)).t).values.item()
    ante_180 = np.mean(cache.sel(time=slice(pre_180, hw_start)).t).values.item()
    adaptation = antecedent_adaptation(cache.t, hw_start, years=10)
    return max_t, mean_r, mean_w, ante_30, ante_90, ante_180, adaptation

def best_fit_slope(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    return m

def avg(xs,ys):
    a = np.sum(xs*ys)
    return a

def ERA_grads(lon, lat, country, city, gridsquare, start, end, hour_base=12):
    era5_file = 'Heat City Data/' + str(country)+ '_' + str(city) + '_era5_heatwave.nc'
    array = xr.open_dataset(era5_file)
    cache = array.interp(coords={'longitude':lon,'latitude':lat},
                         method='nearest')
    hour_fragment = dt.time(12)
    hw_start = dt.datetime.combine(start, hour_fragment)
    hw_end = dt.datetime.combine(end, hour_fragment)
    pre_07 = hw_start - dt.timedelta(days=6)
    pre_30 = hw_start - dt.timedelta(days=29)
    pre_90 = hw_start - dt.timedelta(days=89)
    pre_180 = hw_start - dt.timedelta(days=179)
    
    max_t = np.max(cache.sel(time=slice(hw_start, hw_end)).t).values.item()
    
    subarr07 = np.append(cache.sel(time=slice(pre_07, hw_start)).t, max_t)
    subarr30 = np.append(cache.sel(time=slice(pre_30, hw_start)).t, max_t)
    subarr90 = np.append(cache.sel(time=slice(pre_90, hw_start)).t, max_t)
    subarr180 = np.append(cache.sel(time=slice(pre_180, hw_start)).t, max_t)

    arr_07 = np.array([x+1 for x in range(8)])
    arr_30 = np.array([x+1 for x in range(31)])
    arr_90 = np.array([x+1 for x in range(91)])
    arr_180 = np.array([x+1 for x in range(181)])
    
    grad07 = np.apply_along_axis(best_fit_slope, axis=0, arr=arr_07, ys=subarr07).item()
    grad30 = np.apply_along_axis(best_fit_slope, axis=0, arr=arr_30, ys=subarr30).item()
    grad90 = np.apply_along_axis(best_fit_slope, axis=0, arr=arr_90, ys=subarr90).item()
    grad180 = np.apply_along_axis(best_fit_slope, axis=0, arr=arr_180, ys=subarr180).item()

    return grad07, grad30, grad90, grad180

def ERA_RH(lon, lat, country, city, gridsquare, start, end, hour_base=12):
    era5_file = 'Heat City Data/' + str(country)+ '_' + str(city) + '_era5_heatwave.nc'
    array = xr.open_dataset(era5_file)
    cache = array.interp(coords={'longitude':lon,'latitude':lat},
                         method='nearest')
    hour_fragment = dt.time(12)
    hw_start = dt.datetime.combine(start, hour_fragment)
    hw_end = dt.datetime.combine(end, hour_fragment)
    pre_07 = hw_start - dt.timedelta(days=6)
    pre_30 = hw_start - dt.timedelta(days=29)
    pre_90 = hw_start - dt.timedelta(days=89)
    pre_180 = hw_start - dt.timedelta(days=179)
    
    mean_r = np.mean(cache.sel(time=slice(hw_start, hw_end)).r).values.item()

    ante_30 = np.mean(cache.sel(time=slice(pre_30, hw_start)).r).values.item()
    ante_90 = np.mean(cache.sel(time=slice(pre_90, hw_start)).r).values.item()
    ante_180 = np.mean(cache.sel(time=slice(pre_180, hw_start)).r).values.item()
    adaptation = antecedent_adaptation(cache.r, hw_start, years=10)
    return ante_30, ante_90, ante_180, adaptation


### Import and scrub data
heat_file1 = 'Results_NCEPDOE_90th.csv'
df = pd.read_csv(heat_file1)
df['Coordinates'] = df['CodeSite'].str.split("_")
df[['Longitude','Latitude']] = pd.DataFrame(df.Coordinates.tolist(),
                                            index= df.index)
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)
coordinates = df[['Longitude','Latitude']].copy()
coordinates = coordinates.drop_duplicates()
coordinates = coordinates.reset_index(drop=True)
coordinates['City'], coordinates['Country'] = zip(*coordinates.apply(lambda x: geolocate(x['Latitude'], x['Longitude']),
                      axis=1))
missing = pd.read_csv('Missing_cities.csv')
missing['Latitude'] = missing['Latitude'].astype(float)
missing['Longitude'] = missing['Longitude'].astype(float)
coordinates = missing_column_match(coordinates, missing, ['Longitude','Latitude'], ['City'])
coordinates['Grid_Square'] = coordinates.apply(lambda row: grid_square(row['Latitude'], row['Longitude']), axis=1)
df = pd.merge(df, coordinates, on=['Longitude','Latitude'])
core_df = df[['StartDate','EndDate','DocuMortHetWav','PercentMortality','Latitude','Longitude','Grid_Square','City','Country','Ave_MaxT','Ave_MeanRH']]
core_df['StartDate'] = pd.to_datetime(core_df['StartDate'], errors='coerce').dt.date
core_df['EndDate'] = pd.to_datetime(core_df['EndDate'], errors='coerce').dt.date
core_df['Year'] = pd.to_datetime(core_df['StartDate']).dt.strftime('%Y').apply(pd.to_numeric)


### Obtain Obesity Data
bmi_file = 'Lancet_2017_BMI.csv'
bmi_f = pd.read_csv(bmi_file, encoding = "ISO-8859-1")
w_var = 'Mean BMI'
bmi_f[w_var] = bmi_f[w_var].apply(pd.to_numeric)
bmi_f = bmi_f.replace('United States of America', 'United States')
male = bmi_f[bmi_f['Sex']=='Men'][['Country/Region/World','Year',w_var]]
female = bmi_f[bmi_f['Sex']=='Women'][['Country/Region/World','Year',w_var]]
bf = pd.merge(male, female, on=['Country/Region/World','Year'])
bf[w_var] = bf[['Mean BMI_x', 'Mean BMI_y']].mean(axis=1)
bf['Country/Region/World'] = bf['Country/Region/World'].apply(
    lambda x: fuzzy_match(x, coordinates['Country']))
bf = bf.rename(columns={bf.columns[0]:'Country'})
core_df = pd.merge(core_df, bf, on=['Country','Year'], how='left')
core_df['GEO'] = coco.convert(core_df['Country'], src="regex", to="ISO3")


### Obtain Population Data
# wf = pd.DataFrame(columns=['id','name'])
# subset_urls = ['/locations?sort=id&pageNumber=' + str(x+1) for x in range(3)]
# for url in subset_urls:
#     locations = UN_API_Call(url, simple=True)
#     wf = pd.concat([wf, locations],join='inner', ignore_index=True)
# wf = wf.drop(wf.index[237:])
# countrycodes = wf['id'].values
# start = 1980
# indicator = 46
# pf = pd.DataFrame(columns=['Country','Year','Total Population','Pyramid'])
# for code in countrycodes: 
#     for i in range(40):
#         year = start+i
#         year_sub_url = '/locations/' + str(code) +'/start/' + str(year) + '/end/' + str(year)
#         target_url = '/data/indicators/' + str(indicator) + year_sub_url
#         cache = UN_API_Call(target_url, simple=True)
#         try:
#             country = statistics.mode(cache['location'])
#             pyramid, total = percent_pop_by_sex(cache, 'sex', 'Both sexes', 'value')
#             row_to_append = pd.DataFrame({'Country': country,'Year': year,
#                                           'Total Population': total, 'Pyramid': 0}, index=[0])
#             row_to_append['Pyramid']=row_to_append['Pyramid'].astype('object')
#             row_to_append.at[0, 'Pyramid'] = pyramid
#             pf = pd.concat([pf, row_to_append], ignore_index=True)
#         except:
#             pass
# age_band = ['P:0-4','P:5-9','P:10-14','P:15-19','P:20-24','P:25-29','P:30-34',
#             'P:35-39','P:40-44','P:45-49','P:50-54','P:55-59','P:60-64','P:65-69',
#             'P:70-74','P:75-79','P:80-84','P:85-89','P:90-94','P:95-99','P:100+']
# pf[age_band] = pd.DataFrame(pf.Pyramid.tolist())
# pf['Country'][pf['Country'] == 'China, Taiwan Province of China'] = 'Taiwan'
# pf['Country'] = coco.convert(pf['Country'], src="regex", to="ISO3")
# pf = pf.rename(columns={pf.columns[0]:'GEO'})
pf = pd.read_csv('PopulationCache.csv')
core_df = pd.merge(core_df, pf, on=['GEO','Year'], how='left')


era5_url = 'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2'
era5 = xr.open_zarr(era5_url, chunks=128, consolidated=True)
variable_set = ['2m_temperature',
                'surface_pressure',
                '10m_v_component_of_wind',
                '10m_u_component_of_wind',
                'specific_humidity']
surface_set = era5[variable_set[:4]]
pressure_set = era5[variable_set[4:]].sel(level=1000)
full_set = xr.merge([surface_set, pressure_set])
full_set['rh'] = relative_from_specific_humidity(full_set['surface_pressure'],full_set['specific_humidity'],full_set['2m_temperature'],273.16)





# ### Obtain ERA5 data for each location
c = cdsapi.Client()
# # coordinates.apply(lambda row: ERA5_retrieval(row['Country'], row['City'], row['Grid_Square']), axis=1)

# headers = ['Max_T', 'Mean_H', 'Mean_W', '30_T', '90_T', '180_T', 'Adaptive_T']
# core_df['Extracted'] = core_df.apply(lambda x: ERA_heatwave_extraction(x['Longitude'],
#                          x['Latitude'], x['Country'], x['City'], x['Grid_Square'],
#                          x['StartDate'], x['EndDate']), axis=1)
# core_df[headers] = pd.DataFrame(core_df['Extracted'].tolist(), index=core_df.index)


filename = 'Lethal_Heat_90.csv'
df = pd.read_csv(filename)
df = df.fillna(value=0)

df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce').dt.date
df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce').dt.date

headers = ['30_H','90_H','180_H','Adaptive_H']
df['Humids'] = df.apply(lambda x: ERA_RH(x['Longitude'],
                          x['Latitude'], x['Country'], x['City'], x['Grid_Square'],
                          x['StartDate'], x['EndDate']), axis=1)
df[headers] = pd.DataFrame(df['Humids'].tolist(), index=df.index)




# headers = ['T Grad 07', 'T Grad 30', 'T Grad 90', 'T Grad 180']
df['Grads'] = df.apply(lambda x: ERA_grads(x['Longitude'],
                          x['Latitude'], x['Country'], x['City'], x['Grid_Square'],
                          x['StartDate'], x['EndDate']), axis=1)
df[headers] = pd.DataFrame(df['Grads'].tolist(), index=df.index)


# cols = ['P:0-4', 'P:5-9', 'P:10-14',
# 'P:15-19', 'P:20-24', 'P:25-29', 'P:30-34', 'P:35-39', 'P:40-44',
# 'P:45-49', 'P:50-54', 'P:55-59', 'P:60-64', 'P:65-69', 'P:70-74',
# 'P:75-79', 'P:80-84', 'P:85-89', 'P:90-94', 'P:95-99', 'P:100+']
# ys = [2 + 5*x for x in range(21)]
# pop_array = df[cols].to_numpy()
# result1 = np.apply_along_axis(best_fit_slope, axis=1, arr=pop_array, ys=ys)
# result2 = np.apply_along_axis(avg, axis=1, arr=pop_array, ys=ys)
# result1 = pd.Series(result1)
# result2 = pd.Series(result2)
# df = df.assign(Gradient=result1)
# df = df.assign(Avg_Age=result2)
# df = df.drop(columns=cols)
# df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
# df['Rate_30'] = (df['Max_T'] - df['30_T'])/30
# df['Rate_90'] = (df['Max_T'] - df['90_T'])/90
# df['Rate_180'] = (df['Max_T'] - df['180_T'])/180
# df.to_csv('Lethal_Heat_90C.csv')