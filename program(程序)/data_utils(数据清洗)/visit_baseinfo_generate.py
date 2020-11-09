# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:46:00 2020

@author: aid
"""

import pandas as pd
import numpy as np
from py_sql_connect import Sql_df
import gc
import re
import os
from datetime import datetime
import time


data_input = Sql_df('aid-livelihood')
#base_info = data_input.sql_input_all('select patient_id,sex,date_birth,addr_home,addr_hukou,marital_status,input_flag from pa_patient_info1')
def disease_detect(data):
    result_dict = {}
    diag_code_dis = data['disease_diag_code'].groupby(data['disease_diag_name']).value_counts(dropna = False)
    name_list = list(data['disease_diag_name'].unique())
    for i,disease_name in enumerate(name_list):
        tmp = diag_code_dis[disease_name]
        all_n = tmp.sum()
        try:
            code_num = all_n - tmp['-999']
        except:
            code_num = all_n
        if code_num/all_n < 0.005:
            result_dict[disease_name] = round(code_num/all_n,3)
    return result_dict
def remove_detect_disease(file_name):
    visit_data = pd.read_pickle('visit_data//'+file_name)
    visit_data['disease_diag_name'] = visit_data['disease_diag_name'].fillna('-999')
    visit_data['disease_diag_name'] = visit_data['disease_diag_name'].apply(lambda x:x.strip())
    visit_data['disease_diag_code'] = visit_data['disease_diag_code'].fillna('-999')
    abnormal_disease_dict = disease_detect(visit_data)
    remove_name = []
    for name in abnormal_disease_dict.keys():
        if abnormal_disease_dict[name]>=0.002:
            remove_name.append(name)
    visit_data = visit_data[~visit_data['disease_diag_name'].isin(remove_name)]
    return visit_data


def info_duplicates(base_info):
    base_info['miss'] = base_info.isnull().sum(axis = 1)
    tmp_miss_df = base_info['miss'].groupby(base_info['patient_id']).min()
    zip_list = list(zip(tmp_miss_df.index,list(tmp_miss_df)))
    base_info.index = pd.MultiIndex.from_arrays([list(base_info['patient_id']),list(base_info['miss'])])
    base_info['flag'] = base_info.index.isin(zip_list)
    base_info = base_info.reset_index()
    del base_info['level_0']
    del base_info['level_1']
    base_info = base_info[base_info['flag'] == True]
    base_info = base_info.drop_duplicates(subset = ['patient_id'],keep = 'first')
    del base_info['miss']
    del base_info['flag']
    gc.collect()
    return base_info

data_dir = os.listdir('visit_data')
info_list = ['pa_patient_info1','pa_patient_info2','pa_patient_info3','pa_patient_info4','pa_patient_info5',
             'pa_patient_info6','pa_patient_info7','pa_patient_info8','pa_patient_info9','pa_patient_info10']
for visit_name in data_dir:
    start = time.time()
    visit_data = remove_detect_disease(visit_name)
    result_visit_data = pd.DataFrame()
    for info in info_list:
        base_info = data_input.sql_input_all('select patient_id,sex,date_birth,addr_home,addr_hukou,marital_status,input_flag,gy_xzz,gy_hjdz from '+info)
        base_info = info_duplicates(base_info)
        visit_data_info = pd.merge(visit_data,base_info,on = 'patient_id',how = 'inner')
        visit_data = visit_data[~visit_data['patient_id'].isin(visit_data_info['patient_id'])]
        del base_info
        gc.collect()
        result_visit_data = result_visit_data.append(visit_data_info)
    result_visit_data = result_visit_data.drop_duplicates(subset = ['patient_id','visit_date'],keep = 'last')
    result_visit_data.to_pickle('visit_data_with_info//'+visit_name)
    del visit_data
    del visit_data_info
    del result_visit_data
    gc.collect()
    end = time.time()
    during_time = round(end-start,0)/60
    print('mend data has spend %f minutes'%during_time)
    
    
        
        
        
        
        
        
































