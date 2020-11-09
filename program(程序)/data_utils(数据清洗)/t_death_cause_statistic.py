# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:02:58 2020

@author: aid
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from py_sql_connect import Sql_df
import time


data_input = Sql_df('aid-livelihood')
t_death_cause = data_input.sql_input_all('select * from t_death_cause_new_2')

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
date_range = pd.date_range('20140101','20181231',freq = '1D')
date_range_list = [x.strftime('%Y%m%d') for x in date_range]
t_death_cause = t_death_cause.sort_values(by = 'death_time')
t_death_cause['death_time'] = t_death_cause['death_time'].apply(lambda x:x.strftime('%Y%m%d'))

def statistic_amount(set_df,date_range_list):
    try:
        distribution_df = set_df.groupby(['death_time','sex']).size().unstack().reset_index()[['death_time','男','女']]
        distribution_df['总计'] = distribution_df[['男','女']].sum(axis = 1)
    except:
        distribution_df = set_df.groupby(['death_time','sex']).size().unstack().reset_index()[['death_time','女']]
        distribution_df['总计'] = distribution_df[['女']].sum(axis = 1)
    miss_date_df = []
    miss_date = list(set(date_range_list)-set(distribution_df['death_time']))
    if len(miss_date) != 0:
        for date in miss_date:
            if distribution_df.shape[1] == 4:
                miss_date_df.append([date,0,0,0])
            else:
                miss_date_df.append([date,0,0])
    else:
        pass
    distribution_df =distribution_df.append(pd.DataFrame(miss_date_df,columns = distribution_df.columns))
    distribution_df = distribution_df.sort_values(by = 'death_time')
    distribution_df = distribution_df.fillna(0)
    return distribution_df


death_statistic = pd.ExcelWriter('death_cause_data//death_statistic.xlsx')

pinshu = t_death_cause.groupby(['year','sex']).size().unstack().reset_index()[['year','男','女']]
pinshu.rename(columns = {'男':'男_数量','女':'女_数量'},inplace = True)
age_dis = t_death_cause.groupby(['year','sex'])['new_age'].mean().unstack().reset_index()[['year','男','女']]
age_dis.rename(columns = {'男':'男_年龄','女':'女_年龄'},inplace = True)
dis_df = pinshu.merge(age_dis,on = 'year',how = 'inner')
dis_df.to_excel(death_statistic,sheet_name = '年度统计',index = None)

icd_list = t_death_cause['new_icd'].value_counts().index
distribution_df_all = statistic_amount(t_death_cause,date_range_list)
distribution_df_all.to_excel(death_statistic,sheet_name = '总计',index = None)
for icd in icd_list:
    set_df = t_death_cause[t_death_cause['new_icd']==icd]
    distribution_df = statistic_amount(set_df,date_range_list)
    distribution_df.to_excel(death_statistic,sheet_name = icd,index = None)
death_statistic.save()



    
    
    









