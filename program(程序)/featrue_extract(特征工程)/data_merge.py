# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:10:16 2020

@author: 86156
"""

import pandas as pd
'''
对整理好的宽表进行拼接
'''
def data_input():
    meteorological_df = pd.read_pickle('..//wide_data(宽表)//t_meteorological_data.pkl')
    air_quality_df = pd.read_pickle('..//wide_data(宽表)//air_quality_wide_data.pkl')
    death_df = pd.read_pickle('..//wide_data(宽表)//death_wide_data.pkl')
    visit_icd_embedding_data = pd.read_pickle('..//wide_data(宽表)//visit_icd_embedding_data.pkl')
    visit_df = pd.read_pickle('..//wide_data(宽表)//visit_wide_data.pkl')
    return meteorological_df,air_quality_df,death_df,visit_icd_embedding_data,visit_df

def visit_continue_day_mend(visit_df):
    '''
    此方法不可取，太多地区的疾病就诊量不是连续都有的
    '''
    continue_date_list = pd.date_range('20140101','20181231',freq = '1D')
    area_list = visit_df['area'].unique()
    disease_list = visit_df['disease'].unique()
    date_n = len(continue_date_list)
    area_n = len(area_list)
    disease_n = len(disease_list)
    continue_date_list = list(continue_date_list)*area_n*disease_n
    area_list = list(area_list)*date_n*disease_n
    area_list.sort()
    disease_list = list(disease_list)*date_n*area_n
    disease_list.sort()
    base_visit_df = pd.DataFrame({'visit_date':continue_date_list,'area':area_list,'disease':disease_list})
    continue_visit_df = base_visit_df.merge(visit_df,on = ['visit_date','area','disease'],how = 'left')
    return continue_visit_df

def main():
    meteorological_df,air_quality_df,death_df,visit_icd_embedding_data,visit_df = data_input()
    meteorological_df.rename(columns = {'monitor_date':'visit_date'},inplace = True)
    air_quality_df.rename(columns = {'monitor_date':'visit_date'},inplace = True)
    death_df.rename(columns = {'death_time':'visit_date','permanent_addr':'area'},inplace = True)
    
    feature_df = visit_df.merge(visit_icd_embedding_data,on = 'disease',how = 'left')
    print(feature_df.shape)
    death_df['visit_date'] = death_df['visit_date'].apply(lambda x:pd.to_datetime(x))
    feature_df = feature_df.merge(death_df,on = ['visit_date','area'],how = 'left')
    print(feature_df.shape)
    air_quality_df['visit_date'] = air_quality_df['visit_date'].apply(lambda x:pd.to_datetime(x))
    feature_df = feature_df.merge(air_quality_df,on = ['visit_date','area'],how = 'left')
    print(feature_df.shape)
    meteorological_df['visit_date'] = meteorological_df['visit_date'].apply(lambda x:pd.to_datetime(x))
    feature_df = feature_df.merge(meteorological_df,on = 'visit_date',how = 'left')
    print(feature_df.shape)
    feature_df.to_pickle('..//wide_data(宽表)//feature_wide_data.pkl')
if __name__ == '__main__':
    main()






























