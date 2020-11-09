# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:02:56 2020

@author: aid
"""

import pandas as pd
import numpy as np
import os
from py_sql_connect import Sql_df
import gc
import re
import time
from datetime import datetime


data_input = Sql_df('aid-livelihood')
#visit_info = pd.read_pickle('visit_data_with_info//pa_visit_j00_j06.pkl.pkl')

pa_empi = data_input.sql_input_all('select patient_id,addr_home,addr_hukou from pa_empi where addr_home is not null or addr_hukou is not null')

'''
对含有基本信息的门诊数据进行处理，处理流程如下：
1、删除一些列：sex_y,disease_diag_name,disease_diag_code
2、整理地区：通过addr_home和addr_hukou整理出一列addr，并根据addr划分出区，得到addr_area,以居住地址为主
3、整理marital_status
4、根据visit_date和date_birth整理出门诊时的年龄
'''

def area_tran(x):
    findword=u"(越秀|海珠|荔湾|天河|白云|黄埔|花都|番禺|南沙|从化|增城|其他)"
    pattern = re.compile(findword)
    results =  pattern.findall(x)
    if len(results)>=1:
        return results[0]
    else:
        return '其他'
def date_tran(x):
    y = x.replace(' ','')
    y = y.replace('-','')
    if len(y)>=5:
        if y[4:] == '0':
            z = y[:4]+'-1'
        else:
            z = y[:4]+'-'+y[4:]
        return z
    else:
        return '9999-1'
#将剩下没有addr的运用pa_empi中的数据看是否可以补全一些，先整理pa_empi数据
pa_empi_with_home_addr = pa_empi[pa_empi['addr_home'].isnull() == False]
pa_empi_without_home_addr = pa_empi[pa_empi['addr_home'].isnull() == True]
pa_empi_with_home_addr['addr'] = pa_empi_with_home_addr['addr_home']
pa_empi_without_home_addr['addr'] = pa_empi_without_home_addr['addr_hukou']
pa_empi = pa_empi_with_home_addr.append(pa_empi_without_home_addr)
pa_empi = pa_empi.sort_index()
pa_empi = pa_empi.drop(['addr_home','addr_hukou'],axis = 1)
pa_empi = pa_empi.drop_duplicates(subset = ['patient_id'],keep = 'last')
del pa_empi_with_home_addr
del pa_empi_without_home_addr
gc.collect()

visit_info_dir_list = os.listdir('visit_data_with_info')
for info in visit_info_dir_list:
    start = time.time()
    visit_info = pd.read_pickle('visit_data_with_info//'+info)
    visit_info = visit_info.drop(['sex_y','disease_diag_name','disease_diag_code','visit_dept_name'],axis = 1)#进行第一步
# In[] 进行第二步
#先以家庭住址为基准，要是家庭住址缺失，则用户口住址填补，若都缺失，则用pa_empi中的家庭住址填补，否则为缺失值
    visit_without_addr = visit_info[visit_info['gy_xzz'].isnull() == True]
    visit_with_addr = visit_info[visit_info['gy_xzz'].isnull() == False]
    visit_with_hukou_addr = visit_without_addr[visit_without_addr['gy_hjdz'].isnull() == False]
    visit_without_hukou_addr = visit_without_addr[visit_without_addr['gy_hjdz'].isnull() == True]
    del visit_without_addr
    gc.collect()
    visit_with_addr['addr'] = visit_with_addr['gy_xzz']
    visit_with_hukou_addr['addr'] = visit_with_hukou_addr['gy_hjdz']
    
    visit_without_hukou_addr = pd.merge(visit_without_hukou_addr,pa_empi,on = 'patient_id',how = 'left')
    for df in [visit_with_hukou_addr,visit_without_hukou_addr]:
        visit_with_addr = visit_with_addr.append(df)
    visit_info = visit_with_addr.copy()
    del visit_with_hukou_addr
    del visit_without_hukou_addr
    del visit_with_addr
    gc.collect()
    
    visit_info = visit_info.drop(['addr_home','addr_hukou'],axis = 1)
    visit_info['addr'] = visit_info['addr'].fillna('其他')
    visit_info['addr_area'] = visit_info['addr'].apply(lambda x:area_tran(x))

# In[]进行第三步，整理marital_status
    del visit_info['marital_status']
    gc.collect()

# In[]进行第四步，整理出患者出生日期
    visit_info['date_birth'] = visit_info['date_birth'].fillna('2030-1')
    visit_info['date_birth'] = visit_info['date_birth'].apply(lambda x:date_tran(x))
    visit_info['date_birth'] = visit_info['date_birth'].apply(lambda x:datetime.strptime(x, "%Y-%m"))
    visit_info['date_birth'] = visit_info['date_birth'].apply(lambda x:datetime.strptime('2030-1', "%Y-%m") if len(str(x)) != 19 else x)
    visit_info = visit_info.reset_index()
    del visit_info['index']
    gc.collect()
    visit_date_array = pd.to_datetime(visit_info['visit_date'], unit='s',errors  = 'ignore')
    visit_date_array = visit_date_array.dt.to_pydatetime()
    birth_date_array = pd.to_datetime(visit_info['date_birth'],errors  = 'ignore')
    visit_info['visit_age'] =visit_date_array-birth_date_array
    visit_info['visit_age'] = visit_info['visit_age'].apply(lambda x:round(x.days/365,0))
    visit_info['visit_age'][visit_info['visit_age']<0] = None 
    end = time.time()
    during_time = round(end-start,0)/60
    print('mend data has spend %f minutes'%during_time)
    visit_info.to_pickle('visit_data_with_info_mended//'+info)
    



        
        
    
    
    
    
    


























