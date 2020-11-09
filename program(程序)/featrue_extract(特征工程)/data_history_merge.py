# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:05:30 2020

@author: 86156
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
'''
用最近一天的门诊特征和前一天的环境特征去预测当天的就诊量
'''

def data_input():
    meteorological_df = pd.read_pickle('..//wide_data(宽表)//t_meteorological_data.pkl')
    air_quality_df = pd.read_pickle('..//wide_data(宽表)//air_quality_wide_data.pkl')
    death_df = pd.read_pickle('..//wide_data(宽表)//death_wide_data.pkl')
    visit_icd_embedding_data = pd.read_pickle('..//wide_data(宽表)//visit_icd_embedding_data.pkl')
    visit_df = pd.read_pickle('..//wide_data(宽表)//visit_wide_data.pkl')
    return meteorological_df,air_quality_df,death_df,visit_icd_embedding_data,visit_df

def recent_day_find_visit(base_info_df):
    group = base_info_df.groupby(['area','disease'])['visit_date']
    history_samples = []
    for df in tqdm(group):
        for i in range(len(df[1])):
            if i == 0:
                sample = [df[0][0],df[0][1],df[1].iloc[i],np.nan]
            else:
                sample = [df[0][0],df[0][1],df[1].iloc[i],df[1].iloc[i-1]]
            history_samples.append(sample)
    history_df = pd.DataFrame(history_samples,columns = ['area','disease','visit_date_target','visit_date_recent_visit'])
    return history_df
def recent_day_find_other(base_info_df,type_name):
    group = base_info_df.groupby(['area'])['visit_date']
    history_samples = []
    for df in tqdm(group):
        for i in range(len(df[1])):
            if i == 0:
                sample = [df[0],df[1].iloc[i],np.nan]
            else:
                sample = [df[0],df[1].iloc[i],df[1].iloc[i-1]]
            history_samples.append(sample)
    history_df = pd.DataFrame(history_samples,columns = ['area','visit_date_target','visit_date_recent_'+type_name])
    return history_df

def histroy_base_data_merge():
    meteorological_df,air_quality_df,death_df,visit_icd_embedding_data,visit_df = data_input()
    visit_base_info_df = visit_df[['visit_date','area','disease']]
    visit_base_history_df = recent_day_find_visit(visit_base_info_df)
    
    air_quality_df.rename(columns = {'monitor_date':'visit_date'},inplace = True)
    air_quality_df['visit_date'] = air_quality_df['visit_date'].apply(lambda x:pd.to_datetime(x))
    air_quality_base_info_df = air_quality_df[['visit_date','area']]
    air_quality_base_history_df = recent_day_find_other(air_quality_base_info_df,'air_quality')
    
    death_df.rename(columns = {'death_time':'visit_date','permanent_addr':'area'},inplace = True)
    death_df['visit_date'] = death_df['visit_date'].apply(lambda x:pd.to_datetime(x))
    death_base_info_df = death_df[['visit_date','area']]
    death_base_history_df = recent_day_find_other(death_base_info_df,'death')

    meteorological_df.rename(columns = {'monitor_date':'visit_date_target'},inplace = True)
    meteorological_df['visit_date_target'] = meteorological_df['visit_date_target'].apply(lambda x:pd.to_datetime(x))
    meteorological_base_history_df = meteorological_df[['visit_date_target']]
    meteorological_base_history_df = meteorological_base_history_df.sort_values(by = 'visit_date_target')
    meteorological_recent = []
    for i in range(len(meteorological_base_history_df)):
        if i == 0:
            val = np.nan
        else:
            val = meteorological_base_history_df['visit_date_target'].iloc[i-1]
        meteorological_recent.append(val)
    meteorological_base_history_df['visit_date_recent_meteorological'] = meteorological_recent
    
    history_df = visit_base_history_df.merge(air_quality_base_history_df,on = ['area','visit_date_target'],how = 'left')
    history_df = history_df.merge(death_base_history_df,on = ['area','visit_date_target'],how = 'left')
    history_df = history_df.merge(meteorological_base_history_df,on = 'visit_date_target',how = 'left')
    return history_df,meteorological_df,air_quality_df,death_df,visit_icd_embedding_data,visit_df


def history_data_merge():
    '''
    根据时间对数据进行拼接
    '''
    history_base_df,meteorological_df,air_quality_df,death_df,visit_icd_embedding_data,visit_df = histroy_base_data_merge()
    target_df = visit_df[['visit_date','area','disease','visit_amount']]
    visit_df.rename(columns = {'visit_date':'visit_date_recent_visit','death_gender':'visit_gender','visit_amount':'visit_amount_recent'},inplace = True)
    
    merged_df = history_base_df.merge(visit_df,on = ['area','disease','visit_date_recent_visit'],how = 'left')
    
    air_quality_df.rename(columns = {'visit_date':'visit_date_recent_air_quality'},inplace = True)
    merged_df = merged_df.merge(air_quality_df,on = ['area','visit_date_recent_air_quality'],how = 'left')
    
    death_df.rename(columns = {'visit_date':'visit_date_recent_death'},inplace = True)
    merged_df = merged_df.merge(death_df,on = ['area','visit_date_recent_death'],how = 'left')
    
    meteorological_df.rename(columns = {'visit_date_target':'visit_date_recent_meteorological'},inplace = True)
    merged_df = merged_df.merge(meteorological_df,on = 'visit_date_recent_meteorological',how = 'left')
    
    merged_df = merged_df.merge(visit_icd_embedding_data,on = 'disease',how = 'left')
    
    #添加一个日期是周几的特征
    merged_df['weekday_target'] = merged_df['visit_date_target'].apply(lambda x:x.weekday())
    
    target_df.rename(columns = {'visit_date':'visit_date_target'},inplace = True)
    merged_df = merged_df.merge(target_df,on = ['area','disease','visit_date_target'],how = 'left')
    
    merged_df.to_pickle('..//wide_data(宽表)//merged_history_data.pkl')
    return merged_df
meteorological_df,air_quality_df,death_df,visit_icd_embedding_data,visit_df = data_input()
#merged_df = history_data_merge()

    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    








































    
    
    
    
    
    
