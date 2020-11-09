# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:02:18 2020

@author: aid
"""

import pandas as pd
import numpy as np
from py_sql_connect import Sql_df
import re
from datetime import datetime
import time


data_input = Sql_df('aid-livelihood')
#t_death_cause = data_input.sql_input_all('select id,sex,birthday,age,before_address_detailed,permanent_addres,death_time from t_death_cause')
t_death_cause = data_input.sql_input_all('select * from t_death_cause')
t_death_cause['new_date'] = t_death_cause['death_time'].apply(lambda x:str(x)[:4])
need_death = t_death_cause[t_death_cause['new_date'].isin(['2017','2018','2019'])]
del need_death['new_date']
'''
def area_tran(x):
    findword=u"(越秀|海珠|荔湾|天河|白云|黄埔|花都|番禺|南沙|从化|增城|其他)"
    pattern = re.compile(findword)
    results =  pattern.findall(x)
    if len(results)>=1:
        return results[0]
    else:
        return '其他'                                       
t_death_cause['permanent_addres'].fillna('其他',inplace = True)
t_death_cause['before_addr'] = t_death_cause['before_address_detailed'].apply(lambda x:area_tran(x))
t_death_cause['permanent_addr'] = t_death_cause['permanent_addres'].apply(lambda x:area_tran(x))
# In[]对年龄进行处理
def age_mend(x):
    x = x.strip()
    if '岁' in x:
        return int(x.split('岁')[0])
    elif '月' in x:
        y = int(x.split('月')[0])
        return round(y/12,3)
    elif '天' in x:
        y = int(x.split('天')[0])
        return round(y/365,3)
    else:
        return None
t_death_cause['new_age'] = t_death_cause['age'].apply(lambda x:age_mend(x))

t_death_with_icd = pd.read_csv('death_cause_data//icd_death_mended.csv')



death_new = t_death_cause.merge(t_death_with_icd,on = 'id',how = 'left')
death_new['sex'][~death_new['sex'].isin(['男','女'])] = '未知'
death_new['death_date'] = death_new['death_time'].apply(lambda x:x.strftime('%Y%m%d'))
t_death_cause.to_pickle('death_cause_data//death_cause_data.pkl')

'''

