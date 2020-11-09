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
t_meteorological_data = data_input.sql_input_all('select monitor_date,avg_pressure,max_pressure,\
min_pressure,avg_temp,max_temp,min_temp,avg_humidity,max_humidity,min_humidity,precipitation,avg_wind_speed,sunshine from t_meteorological_data')
t_meteorological_data = t_meteorological_data.drop_duplicates(keep = 'first')
t_meteorological_data = t_meteorological_data.reset_index()
del t_meteorological_data['index']
np.set_printoptions(suppress=True)

selected_vars = ['avg_pressure','max_pressure','min_pressure','avg_temp','max_temp','min_temp','avg_humidity','max_humidity','min_humidity','precipitation','avg_wind_speed','sunshine']
for var in selected_vars:
    t_meteorological_data[var] = t_meteorological_data[var].astype(float)

describe_df = t_meteorological_data[selected_vars].describe().T

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
t_meteorological_data,abnormal_record = anomaly_recognition(t_meteorological_data,'precipitation')
abnormal.append(abnormal_record)
result_df = pd.DataFrame({'vars':['precipitation'],'abnormal':abnormal})
result_df.to_excel('abnoraml.xlsx',index = None)


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

#构造函数对含有avg的指标进行处理
def avg_var_mend(data,var,all_vars):
    selected_vars = []
    for v in all_vars:
        if var in v:
            selected_vars.append(v)
    selected_df = data[selected_vars]
    null_df = selected_df[selected_df['avg_'+var].isnull() == True]
    max_null = null_df['max_'+var].isnull()
    min_null = null_df['min_'+var].isnull()
    for i in range(len(null_df)):
        max_flag,min_flag = max_null.iloc[i],min_null.iloc[i]
        if max_flag == False and min_flag == False:
            null_df['avg_'+var].iloc[i] = (null_df['max_'+var].iloc[i]+null_df['min_'+var].iloc[i])/2
        elif max_flag == False and min_flag == True:
            null_df['avg_'+var].iloc[i] = null_df['max_'+var].iloc[i]
        elif max_flag == True and min_flag == False:
            null_df['avg_'+var].iloc[i] = null_df['min_'+var].iloc[i]
        else:
            null_df['avg_'+var].iloc[i] = None
    data.loc[list(null_df.index),'avg_'+var] = null_df.loc[:,'avg_'+var].values
    mean = data['avg_'+var].mean()
    data['avg_'+var] = data['avg_'+var].fillna(mean)
    del data['max_'+var]
    del data['min_'+var]
    
all_vars = ['avg_pressure','max_pressure','min_pressure','avg_temp','max_temp','min_temp','avg_humidity','max_humidity','min_humidity']
for var in ['pressure','temp','humidity']:
    avg_var_mend(t_meteorological_data,var,all_vars)

t_meteorological_data = t_meteorological_data.sort_values(by = 'monitor_date')
t_meteorological_data = t_meteorological_data.reset_index()
del t_meteorological_data['index']
t_meteorological_data = fill_na(t_meteorological_data,['precipitation','avg_wind_speed'])
del t_meteorological_data['sunshine']

t_meteorological_data.to_pickle('t_meteorological_data\\t_meteorological_data.pkl')


    
    
            
    
    
    

        
            
    
    
    
    
    
    
    
    
    
    
    
    