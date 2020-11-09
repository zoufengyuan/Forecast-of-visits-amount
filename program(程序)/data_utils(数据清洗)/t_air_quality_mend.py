# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:06:14 2020

@author: aid
"""

import pandas as pd
import numpy as np
from py_sql_connect import Sql_df
from scipy.interpolate import lagrange
import gc

data_input = Sql_df('aid-livelihood')
t_air_quality = data_input.sql_input_all('select site_name,monitor_date,so2,no2,co,o31h,o38h,pm10,pm2 from t_air_quality_monitor')

t_air_quality = t_air_quality.drop_duplicates(keep = 'first')

site_df = pd.read_excel('t_air_quality_data//环保数据站点地址(含地区)2.xlsx')
site_dic = {site_df['site'].iloc[i]:site_df['地区'].iloc[i] for i in range(len(site_df))}
t_air_quality['area'] = t_air_quality['site_name']
t_air_quality['area'].replace(site_dic,inplace = True)


np.set_printoptions(suppress=True)
for var in ['so2','no2','co','o31h','o38h','pm10','pm2']:
    t_air_quality[var] = t_air_quality[var].astype(float)

describe_df = t_air_quality[['so2','no2','co','o31h','o38h','pm10','pm2']].describe().T

def anomaly_recognition(data,var_name):
    tmp_df = data[data[var_name].isnull() == False]
    percent_1 = np.percentile(tmp_df[var_name],25)
    percent_3 = np.percentile(tmp_df[var_name],75)
    percent_dis = percent_3 - percent_1
    abnormal_record = {}
    for i in range(len(data[var_name])):
        if data[var_name][i:i+1].isnull().values == False:
            value = data[var_name].iloc[i]
            if (value-percent_3) > 2*percent_dis or (percent_1 - value)>2*percent_dis:
                try:
                    abnormal_record[value] += 1
                except:
                    abnormal_record[value] = 1
                data[var_name].iloc[i] = None
            else:
                pass
        else:
            pass
    abnormal_record = dict(sorted(abnormal_record.items(),key = lambda x:x[0]))
    return data,abnormal_record
abnormal = []
for var in ['so2','no2','co','o31h','o38h','pm10','pm2']:
    t_air_quality,abnormal_record = anomaly_recognition(t_air_quality,var)
    abnormal.append(abnormal_record)
result_df = pd.DataFrame({'vars':['so2','no2','co','o31h','o38h','pm10','pm2'],'abnormal':abnormal})
#result_df.to_excel('abnoraml.xlsx',index = None)


#应用拉格朗日插补法进行缺失填补
#对各个站点分别进行填补，按照时间排序进行填补
#先构造拉格朗日插补法函数

def ployinterp_column(s, n, var, k=1):
    used_list = list(range(n-k, n)) +[n]+ list(range(n+1, n+1+k))
    tmp_y = s[used_list]
    y = tmp_y.reset_index()
    del y['index']
    used_list = list(y.index)
    used_list.remove(k)
    y = y[var]
    y = y[used_list] 
    y = y[y.notnull()] 
    result = lagrange(y.index, list(y))(k)
    if result == 0:
        for x,val in enumerate(s.isnull()[n:].values):
            if val == False:
                result = s[n:].iloc[x]
                break
    return round(result,3)

def fill_na(data,columns_list):
    for i in columns_list:
        for j in range(len(data)):
            if data[i].isnull()[j]:
                
                data[i].iloc[j] = ployinterp_column(data[i], j,i)
    return data

columns_list = ['so2','no2','co','o38h','pm10','pm2']
t_air_quality = t_air_quality.sort_values(by = 'monitor_date')
new_air_quality = pd.DataFrame()
for site in t_air_quality['site_name'].unique():
    tmp_df = t_air_quality[t_air_quality['site_name'] == site]
    tmp_df = tmp_df.sort_values(by = 'monitor_date')
    tmp_df_index = list(tmp_df.index)
    tmp_df = tmp_df.reset_index()
    del tmp_df['index']
    tmp_df = fill_na(tmp_df,columns_list)
    tmp_df.index = tmp_df_index
    new_air_quality = new_air_quality.append(tmp_df)
    #del tmp_df
    gc.collect()

new_air_quality = new_air_quality.sort_index()

t_air_quality = t_air_quality.sort_index()
new_air_quality['area'] = t_air_quality['area']
del new_air_quality['o31h']

new_air_quality.to_pickle('t_air_quality_data//new_air_quality_add2015.pkl')
#new_air_quality = pd.read_pickle('t_air_quality_data//new_air_quality.pkl')
        
            
    
    
    
    
    
    
    
    
    
    
    
    