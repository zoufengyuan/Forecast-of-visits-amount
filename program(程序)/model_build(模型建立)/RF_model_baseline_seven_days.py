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
from tqdm import tqdm


#source_model_df = pd.read_pickle('..//wide_data(宽表)//merged_history_data_all_seven_days.pkl')  
#source_model_df = source_model_df[source_model_df['disease'] == icd_code]
#del source_model_df['disease']
def main(source_model_df,icd_code,area,flag = True):
    if flag:
        source_model_df = source_model_df[(source_model_df['disease'] == icd_code)&(source_model_df['area'] == area)]
    else:
        source_model_df = source_model_df[source_model_df['disease'] == icd_code]
    vars_list = list(source_model_df.columns)
    remove_vars_list = []
    for var in vars_list:
        if any(x in var for x in ['multi','date','embedding','gender']):
            remove_vars_list.append(var)
    remove_vars_list.remove('visit_date_target')
    
    chosed_vars = list(set(vars_list)-set(remove_vars_list))
    
    
    
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
    
    model_df.fillna(-1,inplace = True)
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
    
    #new_vars = ['visit_amount_recent_visit_7','visit_amount_recent_visit_1','weekday_target','visit_amount_recent_visit_3','visit_amount']
    #train_x = train_x[new_vars]
    #test_x = test_x[new_vars]
    
    
    train_y = train_x.pop('visit_amount')
    test_y = test_x.pop('visit_amount')
    
    
    
    rf = RandomForestRegressor(n_estimators=20,random_state=0,verbose = 1,n_jobs = 4)
    rf.fit(train_x,train_y)
    rf_coef = pd.Series(np.abs(rf.feature_importances_), index = train_x.columns)
    rf_imp_coef = rf_coef.sort_values(ascending = False).reset_index()
    rf_imp_coef.rename(columns = {'index':'vars',0:'coef'},inplace = True)
    rf_imp_coef = rf_imp_coef[:10]
    
    y_predict = rf.predict(test_x)
    r2 = r2_score(test_y,y_predict)
    y_df = pd.DataFrame({'y_true':test_y,'y_pred':y_predict,'visit_date':test_date,'area':test_x['area'],'disease':test_x['disease']})
    y_df['area'].replace(info_dict['area'],inplace = True)
    y_df['disease'].replace(info_dict['disease'],inplace = True)
    return rf_imp_coef,r2
    

source_model_df = pd.read_pickle('..//wide_data(宽表)//merged_history_data_all_seven_days.pkl') 
icd_list = source_model_df['disease'].unique()
area_list = source_model_df['area'].unique()
#variable_importance_df = pd.ExcelWriter('..//variables_importance//variables_imp.xlsx')


imp_result_df = pd.DataFrame()

for icd in tqdm(icd_list):
    for area in area_list:
        try:
            rf_imp_coef,r2 = main(source_model_df,icd,area)
            rf_imp_coef['disease'] = [icd]*10
            rf_imp_coef['area'] = [area]*10
            rf_imp_coef['r2'] = [r2]*10
            imp_result_df = imp_result_df.append(rf_imp_coef)
        except:
            print(icd,area)
for icd in tqdm(icd_list):
    try:
        rf_imp_coef,r2 = main(source_model_df,icd,area,False)
        rf_imp_coef['disease'] = [icd]*10
        rf_imp_coef['area'] = [area]*10
        rf_imp_coef['r2'] = [r2]*10
        imp_result_df = imp_result_df.append(rf_imp_coef)
    except:
        print(icd)
    
imp_result_df.to_excel('..//variables_importance//variables_imp_all.xlsx',index = None)


'''
groups = y_df.groupby(['disease','area'])
for df in groups:
    new_df = df[1].sort_values(by = 'visit_date')
    if not os.path.exists('..//predict_plot_seven_days//'+new_df['disease'].iloc[0]):
        os.mkdir('..//predict_plot_seven_days//'+new_df['disease'].iloc[0])
    figure = plt.figure()
    plt.plot(new_df['visit_date'][:100],new_df['y_true'][:100],marker = 'o',label = 'y_true',color = 'blue',linewidth = 0.2,markersize = 2)
    plt.plot(new_df['visit_date'][:100],new_df['y_pred'][:100],marker = 'o',label = 'y_pred',color = 'red',linewidth = 0.2,markersize = 2)
    plt.legend()
    plt.xlabel('visit_date')
    plt.ylabel('visit_amount')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(new_df['disease'].iloc[0]+'_'+new_df['area'].iloc[0]+' visit_amounts')
    plt.savefig('..//predict_plot_seven_days//'+new_df['disease'].iloc[0]+'//'+new_df['disease'].iloc[0]+'_'+new_df['area'].iloc[0]+'.jpg',dpi  = 600)
'''

        













        





