# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:24:05 2020

@author: 86156
"""

import pandas as pd
import numpy as np
import gc
import time
from tqdm import tqdm
import os
import random
from gensim.models import Word2Vec

death_cause_df = pd.read_pickle('..//death_cause(死亡数据)//death_cause_data.pkl')[:1000]
death_cause_df = death_cause_df[death_cause_df['permanent_addr']!= '其他']#不纳入常驻地不为广州的样本
death_cause_df = death_cause_df[death_cause_df['sex'].isin(['男','女'])]#剔除少量未知性别的样本
first_aid_df = pd.read_pickle('..//first_aid(急诊数据)//t_first_aid.pkl')[:1000]
air_quality_df = pd.read_pickle('..//t_air_quality(环保数据)//new_air_quality_add2015.pkl')[:1000]
meteorological_df = pd.read_pickle('..//t_meteorological(气象数据)//t_meteorological_data.pkl')[:1000]

air_quality_df = air_quality_df.sort_values(by = ['area','monitor_date'])
air_quality_df['area'][air_quality_df['area'] == '越秀麓湖'] = '越秀'
air_quality_df['area'][air_quality_df['area'] == '黄埔镇龙'] = '黄埔'
air_quality_df['area'][air_quality_df['area'] == '天河体育西'] = '天河'
air_quality_samples = air_quality_df[-2000:]

visit_name_list = os.listdir('..//pa_visit_data(门诊数据)//visit_data')
visit_df = pd.DataFrame()
for visit_file in visit_name_list:
    little_df = pd.read_pickle('..//pa_visit_data(门诊数据)//visit_data//'+visit_file)
    summary_icd = visit_file.split('.')[0][6:]
    summary_icd_list = [summary_icd]*len(little_df)
    little_df['disease'] = summary_icd_list
    visit_df = visit_df.append(little_df)

#death_cause_df.to_pickle('death_cause_data.pkl')
#first_aid_df.to_pickle('t_first_aid_data.pkl')
#air_quality_df.to_pickle('air_quality_data.pkl')
#meteorological_df.to_pickle('meteorological_data.pkl')


# In[]
#构造特征的规则
'''
按时间、icd、地区、患者数量、年龄分布、性别数量比例、环境指标
疑问点：急诊和死亡信息怎么刻画
'''

# In[]
#对空气质量数据进行整理
'''
得到的字段有：
日期、地区、指标平均值(x_mean)、指标平均值(去除最大最小值)(x_mean_remove)、指标标准差(x_std)、指标标准差(去除最大值最小值)(x_std_remove)、指标多值特征(x_multi)
'''
def std_remove_calculate(row,index_var):
    row_list = list(row[index_var])
    min_ = np.min(row_list)
    max_ = np.max(row_list)
    row_list.remove(min_)
    row_list.remove(max_)
    return np.std(row_list)
def index_calculate(df,monitor_date_var,area_var,site_var,index_var):
    '''
    monitor_date_var:观测日期
    area_var:区域
    site_var:站点
    index_var:指标
    '''
    mean = df.groupby([monitor_date_var,area_var])[index_var].mean().reset_index()
    mean.rename(columns = {index_var:index_var+'_mean'},inplace = True)
    
    std = df.groupby([monitor_date_var,area_var])[index_var].std().reset_index()
    std[index_var].fillna(0,inplace = True)
    std.rename(columns = {index_var:index_var+'_std'},inplace = True)
    
    n_samples = df.groupby([monitor_date_var,area_var])[site_var].size().reset_index()
    n_samples.rename(columns = {site_var:index_var+'_amounts'},inplace = True)
    
    mean_remove = mean.copy()
    std_remove = std.copy()
    #开始计算去除最值的mean和std
    need_remove_var_df = n_samples[n_samples[index_var+'_amounts']>2][[monitor_date_var,area_var]]
    
    remove_cal_df = need_remove_var_df.merge(df[[monitor_date_var,area_var,site_var,index_var]],on = [monitor_date_var,area_var],how = 'left')
    mean_remove_cal = remove_cal_df.groupby([monitor_date_var,area_var]).apply(lambda x:(np.sum(x[index_var])-np.max(x[index_var])-np.min(x[index_var]))/(len(x[index_var])-2)).reset_index()
    mean_remove_cal.rename(columns = {0:index_var+'_mean_remove'},inplace = True)
    
    std_remove_cal = remove_cal_df.groupby([monitor_date_var,area_var]).apply(lambda x:std_remove_calculate(x,index_var)).reset_index()
    std_remove_cal.rename(columns = {0:index_var+'_std_remove'},inplace = True)
    
    mean_remove = mean_remove.merge(mean_remove_cal,on = [monitor_date_var,area_var],how = 'left')
    mean_remove.loc[mean_remove[index_var+'_mean_remove'].isna()==True,index_var+'_mean_remove'] = mean_remove.loc[mean_remove[index_var+'_mean_remove'].isna()==True,index_var+'_mean'].values
    del mean_remove[index_var+'_mean']
    std_remove = std_remove.merge(std_remove_cal,on = [monitor_date_var,area_var],how = 'left')
    std_remove.loc[std_remove[index_var+'_std_remove'].isna()==True,index_var+'_std_remove'] = std_remove.loc[std_remove[index_var+'_std_remove'].isna()==True,index_var+'_std'].values
    del std_remove[index_var+'_std']
    result_df = mean.copy()
    for little_df in [std,mean_remove,std_remove]:
        result_df = result_df.merge(little_df,on = [monitor_date_var,area_var],how = 'left')
    gc.collect()
    return result_df

def air_quality_mend_statistic(air_quality_df):
    for i,index_var in tqdm(enumerate(['so2', 'no2', 'co', 'o38h', 'pm10', 'pm2'])):
        if i == 0:
            air_quality_wide_df = index_calculate(air_quality_df,'monitor_date','area','site_name',index_var)
        else:
            tmp_df = index_calculate(air_quality_df,'monitor_date','area','site_name',index_var)
            air_quality_wide_df = air_quality_wide_df.merge(tmp_df,on = ['monitor_date','area'],how = 'left')
    return air_quality_wide_df
def air_quality_mend_multi(air_quality_df):
    air_quality_need_df = air_quality_df[['monitor_date','area','so2', 'no2', 'co', 'o38h', 'pm10', 'pm2']]
    index_need_group = air_quality_need_df.groupby(['monitor_date','area'])
    air_quality_multi_df = pd.DataFrame()
    for df in tqdm(index_need_group):
        tmp_dict = {'monitor_date':[df[1]['monitor_date'].iloc[0]],'area':[df[1]['area'].iloc[0]]}
        for index in ['so2', 'no2', 'co', 'o38h', 'pm10', 'pm2']:
            item_list = [' '.join([str(x) for x in df[1][index]])]
            tmp_dict[index+'_multi'] = item_list
        tmp_df = pd.DataFrame(tmp_dict)
        air_quality_multi_df = air_quality_multi_df.append(tmp_df)
    return air_quality_multi_df
def air_quality_mend(air_quality_df):
    print('开始整理空气质量表.....')
    start_time = time.time()
    air_quality_statistic_df = air_quality_mend_statistic(air_quality_df)
    air_quality_multi_df = air_quality_mend_multi(air_quality_df)
    air_quality_wide_df = air_quality_statistic_df.merge(air_quality_multi_df,on = ['monitor_date','area'],how = 'left')
    air_quality_wide_df.to_pickle('..//wide_data(宽表)//air_quality_wide_data.pkl')
    end_time = time.time()
    spend_time = end_time-start_time
    print('空气质量表整理完成！用时{}s'.format(round(spend_time,2)))
    return air_quality_wide_df
#air_quality_wide_df = air_quality_mend(air_quality_df)
    
# In[]
## In[]
#对死亡数据进行整理 
'''
得到的字段有：
日期、地区、死亡人数(death_amount)、性别分布(death_gender)、年龄均值(death_age_mean)、年龄标准差(death_age_std)、年龄多值特征(death_age_multi)
'''
def death_cause_mend(death_cause_df):
    '''
    按照死亡人数、性别和年龄的顺序进行整理
    '''
    print('开始整理死亡数据....')
    start_time = time.time()
    #死亡人数
    death_amount_df = death_cause_df.groupby(['death_time','permanent_addr']).size().reset_index()
    death_amount_df.rename(columns = {0:'death_amount'},inplace = True)
    #性别分布
    death_gender_df = death_cause_df.groupby(['death_time','permanent_addr'])['sex'].value_counts().unstack().reset_index()
    gender_distribution = []
    for i in tqdm(range(len(death_gender_df))):
        tmp_dict = {'男':death_gender_df['男'].iloc[i],'女':death_gender_df['女'].iloc[i]}
        gender_distribution.append(str(tmp_dict))
    death_gender_df['death_gender'] = gender_distribution
    death_gender_df = death_gender_df.drop(['男','女'],axis = 1)
    #年龄
    death_age_mean_df = death_cause_df.groupby(['death_time','permanent_addr'])['new_age'].mean().reset_index()
    death_age_mean_df.rename(columns = {'new_age':'death_age_mean'},inplace = True)
    
    death_age_std_df = death_cause_df.groupby(['death_time','permanent_addr'])['new_age'].std().reset_index()
    death_age_std_df.rename(columns = {'new_age':'death_age_std'},inplace = True)
    
    death_age_group = death_cause_df.groupby(['death_time','permanent_addr'])
    death_age_multi = pd.DataFrame()
    for df in tqdm(death_age_group):
        tmp_dict = {'death_time':[df[1]['death_time'].iloc[0]],'permanent_addr':[df[1]['permanent_addr'].iloc[0]]}
        item_list = [' '.join([str(x) for x in df[1]['new_age'].unique()])]
        tmp_dict['death_age_multi'] = item_list
        tmp_df = pd.DataFrame(tmp_dict)
        death_age_multi = death_age_multi.append(tmp_df)
    
    death_wide_df = death_amount_df.copy()
    for df in [death_gender_df,death_age_mean_df,death_age_std_df,death_age_multi]:
        death_wide_df = death_wide_df.merge(df,on = ['death_time','permanent_addr'],how = 'left')
    death_wide_df.to_pickle('..//wide_data(宽表)//death_wide_data.pkl')
    end_time = time.time()
    spend_time = end_time-start_time
    print('死亡数据整理完成！用时{}s'.format(round(spend_time,2)))
    return death_wide_df
#death_wide_df = death_cause_mend(death_cause_df)


# In[]
#对门诊数据进行整理
'''
step 1:筛选出现住址或户籍地址在广州的患者,剔除patient_id为01的患者,根据patient_id、visit_date和gy_bm对门诊数据进行去重
step 2:构造与patient_id无关的特征(字段)：
日期、地区、icd编码(disease)、就诊量(visit_amount)、性别分布(visit_gender)、年龄均值(visit_age_mean)、年龄标准差(visit_age_std)、年龄多值特征(visit_age_multi)
step 3:构造patient_id相关的特征:
日期、地区、icd编码、patient_id--通过每个patient_id的看病总类型构造sentence进行word embedding。
'''
def visit_df_filter(visit_df):
    #step 1
    std_area = ['广州市越秀区','广州市白云区', '广州市海珠区','广州市荔湾区', '广州市番禺区',
       '广州市南沙区', '广州市黄埔区', '广州市天河区', '广州市花都区', '广州市从化区', '广州市增城区']
    area_tran_dict = {x:x[3:5] for x in std_area}
    visit_df['area'] = visit_df['gyxzd']
    visit_df['area'][~visit_df['area'].isin(std_area)] = np.nan
    visit_df['gyhjdz'][~visit_df['gyhjdz'].isin(std_area)] = np.nan
    condition = visit_df['area'].isna()
    visit_df.loc[condition==True,'area'] = visit_df.loc[condition == True,'gyhjdz'].values
    visit_df = visit_df[visit_df['area'].isna() == False]
    visit_df = visit_df[visit_df['patient_id'] != '01']
    visit_df = visit_df.drop_duplicates(['patient_id','visit_date','gy_bm'],keep = 'first')
    visit_df['area'].replace(area_tran_dict,inplace = True)
    return visit_df

def visit_df_mend(visit_df):
    print('开始整理门诊数据....')
    start_time = time.time()
    visit_df = visit_df_filter(visit_df)
    #step 2
    visit_amount_df = visit_df.groupby(['visit_date','area','disease']).size().reset_index()
    visit_amount_df.rename(columns = {0:'visit_amount'},inplace = True)
    
    visit_gender_df = visit_df.groupby(['visit_date','area','disease'])['sex'].value_counts().unstack().reset_index()
    visit_gender_df.fillna(0,inplace = True)
    gender_distribution = []
    for i in tqdm(range(len(visit_gender_df))):
        tmp_dict = {'男':visit_gender_df['男'].iloc[i],'女':visit_gender_df['女'].iloc[i],'未知':visit_gender_df['未知'].iloc[i]}
        gender_distribution.append(str(tmp_dict))
    visit_gender_df['visit_gender'] = gender_distribution
    visit_gender_df = visit_gender_df.drop(['男','女','未知'],axis = 1)
    #求年龄均值
    visit_age_mean_df = visit_df.groupby(['visit_date','area','disease'])['age'].mean().reset_index()
    visit_age_mean_df.rename(columns = {'age':'visit_age_mean'},inplace = True)
    #对缺失的age_mean进行均值填充
    describe_age_df = visit_age_mean_df.groupby(['area','disease'])['visit_age_mean'].mean().reset_index()
    describe_age_df.rename(columns = {'visit_age_mean':'describe_mean'},inplace = True)
    visit_age_mean_df = visit_age_mean_df.merge(describe_age_df,on = ['area','disease'],how = 'left')
    fill_flag = visit_age_mean_df['visit_age_mean'].isna()
    visit_age_mean_df.loc[fill_flag == True,'visit_age_mean'] = visit_age_mean_df.loc[fill_flag == True,'describe_mean'].values
    del visit_age_mean_df['describe_mean']
    
    #得到age多值特征
    visit_age_group = visit_df.groupby(['visit_date','area','disease'])
    visit_age_multi = pd.DataFrame()
    for df in tqdm(visit_age_group):
        tmp_dict = {'visit_date':[df[1]['visit_date'].iloc[0]],'area':[df[1]['area'].iloc[0]],'disease':[df[1]['disease'].iloc[0]]}
        item_list = [' '.join([str(x) for x in df[1]['age'].unique()])]
        tmp_dict['visit_age_multi'] = item_list
        tmp_df = pd.DataFrame(tmp_dict)
        visit_age_multi = visit_age_multi.append(tmp_df)
    
    #先根据visit_age_mean_df对visit_df中的age进行填补，后再求std
    visit_df = visit_df.merge(visit_age_mean_df,on = ['visit_date','area','disease'],how = 'left')
    visit_age_fill_flag = visit_df['age'].isna()
    visit_df.loc[visit_age_fill_flag == True,'age'] = visit_df.loc[visit_age_fill_flag == True,'visit_age_mean'].values
    visit_age_std_df = visit_df.groupby(['visit_date','area','disease'])['age'].std().reset_index()
    visit_age_std_df.rename(columns = {'age':'visit_age_std'},inplace = True)
    visit_age_std_df['visit_age_std'].fillna(0,inplace = True)
    
    visit_wide_df = visit_amount_df.copy()
    for df in [visit_gender_df,visit_age_mean_df,visit_age_std_df,visit_age_multi]:
        visit_wide_df = visit_wide_df.merge(df,on = ['visit_date','area','disease'],how = 'left')
    visit_wide_df.to_pickle('..//wide_data(宽表)//visit_wide_data.pkl')
    end_time = time.time()
    spend_time = end_time-start_time
    print('门诊数据非embedding部分整理完成！用时{}s'.format(round(spend_time,2)))
    return visit_wide_df
#visit_wide_df = visit_df_mend(visit_df)
def visit_df_icd_embedding(visit_df,embedding_var,embedding_size = 16):
    '''
    实现step3
    '''
    print('开始整理门诊数据icd编码embedding部分....')
    start_time = time.time()
    visit_df = visit_df_filter(visit_df)
    visit_df = visit_df.sort_values(by = ['patient_id','visit_date'])
    patientid_with_icd = visit_df.groupby('patient_id')[embedding_var].unique().reset_index()
    patientid_with_icd[embedding_var] = patientid_with_icd[embedding_var].apply(lambda x:list(x))
    sentence = list(patientid_with_icd[embedding_var])
    print('word2vec training......')
    random.shuffle(sentence)
    model = Word2Vec(sentence,size = embedding_size,window = 10,min_count = 1,workers = 4,iter = 10)
    print('output')
    
    icd_values = visit_df[embedding_var].unique()
    disease_w2v = []
    for icd in icd_values:
        try:
            tmp_list = [icd]
            tmp_list.extend(model[icd])
            disease_w2v.append(tmp_list)
        except:
            pass
    columns_names = [embedding_var]
    for i in range(embedding_size):
        columns_names.append(embedding_var+'_patientid_embedding_'+str(i))
    embedding_df = pd.DataFrame(disease_w2v,columns = columns_names)
    embedding_df.to_pickle('..//wide_data(宽表)//visit_icd_embedding_data.pkl')
    end_time = time.time()
    spend_time = end_time-start_time
    print('门诊数据embedding部分整理完成！用时{}s'.format(round(spend_time,2)))
    return embedding_df,patientid_with_icd
#embedding_df,patientid_with_icd = visit_df_icd_embedding(visit_df,'disease',16)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    




































