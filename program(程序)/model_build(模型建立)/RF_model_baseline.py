# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:13:07 2020

@author: FengY Z
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import os

source_model_df = pd.read_pickle('..//wide_data(宽表)//merged_history_data_all.pkl')
source_model_df.fillna(-1,inplace = True)
chosed_vars = ['visit_date_target','area','disease','visit_amount_recent','visit_age_mean','visit_age_std',
               'so2_mean','so2_std','so2_mean_remove','so2_std_remove',
               'no2_mean','no2_std','no2_mean_remove','no2_std_remove',
               'co_mean','co_std','co_mean_remove','co_std_remove',
               'o38h_mean','o38h_std','o38h_mean_remove','o38h_std_remove',
               'pm10_mean','pm10_std','pm10_mean_remove','pm10_std_remove',
               'pm2_mean','pm2_std','pm2_mean_remove','pm2_std_remove',
               'death_amount','death_age_mean','death_age_std',
               'avg_pressure','avg_temp','avg_humidity','precipitation','avg_wind_speed',
               'weekday_target','visit_amount']

def split_df(model_df):
    train_date = list(pd.date_range('20140101','20171231',freq = '1D'))
    test_date = list(pd.date_range('20180101','20181231',freq = '1D'))
    train_df = model_df[model_df['visit_date_target'].isin(train_date)]
    test_df = model_df[model_df['visit_date_target'].isin(test_date)]
    del train_df['visit_date_target']
    test_date = test_df.pop('visit_date_target')
    return train_df,test_df,test_date


multi_category_list = ['area','disease']

model_df = source_model_df[chosed_vars]

info_dict = {}
for var in multi_category_list:
    unique_type = model_df[var].unique()
    tran_dict = {}
    for i,value in enumerate(unique_type):
        tran_dict[value] = i
        tran_dict[i] = value
    info_dict[var] = tran_dict
    model_df[var].replace(tran_dict,inplace = True)

train_x,test_x,test_date = split_df(model_df)

train_y = train_x.pop('visit_amount')
test_y = test_x.pop('visit_amount')



rf = RandomForestRegressor(n_estimators=20,random_state=0,verbose = 1,n_jobs = 4)
rf.fit(train_x,train_y)
rf_coef = pd.Series(np.abs(rf.feature_importances_), index = train_x.columns)
rf_imp_coef = rf_coef.sort_values().reset_index()
rf_imp_coef.rename(columns = {'index':'vars',0:'coef'},inplace = True)

y_predict = rf.predict(test_x)
r2 = r2_score(test_y,y_predict)
y_df = pd.DataFrame({'y_true':test_y,'y_pred':y_predict,'visit_date':test_date,'area':test_x['area'],'disease':test_x['disease']})
y_df['area'].replace(info_dict['area'],inplace = True)
y_df['disease'].replace(info_dict['disease'],inplace = True)


groups = y_df.groupby(['disease','area'])
for df in groups:
    new_df = df[1].sort_values(by = 'visit_date')
    if not os.path.exists('..//predict_plot//'+new_df['disease'].iloc[0]):
        os.mkdir('..//predict_plot//'+new_df['disease'].iloc[0])
    figure = plt.figure()
    plt.plot(new_df['visit_date'][:100],new_df['y_true'][:100],marker = 'o',label = 'y_true',color = 'blue',linewidth = 0.2,markersize = 2)
    plt.plot(new_df['visit_date'][:100],new_df['y_pred'][:100],marker = 'o',label = 'y_pred',color = 'red',linewidth = 0.2,markersize = 2)
    plt.legend()
    plt.xlabel('visit_date')
    plt.ylabel('visit_amount')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(new_df['disease'].iloc[0]+'_'+new_df['area'].iloc[0]+' visit_amounts')
    plt.savefig('..//predict_plot//'+new_df['disease'].iloc[0]+'//'+new_df['disease'].iloc[0]+'_'+new_df['area'].iloc[0]+'.jpg',dpi  = 600)
    
    
    
    
    
    

    































        





