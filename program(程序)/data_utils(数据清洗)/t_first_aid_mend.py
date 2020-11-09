# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:11:30 2020

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
from tqdm import tqdm


data_input = Sql_df('aid-livelihood')
t_first_aid = data_input.sql_input_all('select id,sex,birth_date,age,admission_date,diagnosis from t_first_aid')

#对年龄进行处理
birth_flag = t_first_aid['birth_date'].isna()
age_flag = t_first_aid['age'].isna()
new_age_list = []
for i in tqdm(range(len(t_first_aid))):
    if age_flag[i] == False:
        value_list = re.findall("\d+",t_first_aid['age'].iloc[i])
        if len(value_list) == 0:
            try:
                value = (t_first_aid['admission_date'].iloc[i]-t_first_aid['birth_date'].iloc[i]).days//365
            except:
                value = np.nan
        else:
            value = value_list[0]      
    else:
        if birth_flag[i] == True:
            value = np.nan
        else:
            value = (t_first_aid['admission_date'].iloc[i]-t_first_aid['birth_date'].iloc[i]).days//365
    new_age_list.append(value)
t_first_aid['new_age'] = new_age_list
t_first_aid.to_pickle('first_aid_data//t_first_aid.pkl')

#t_first_aid_addr = data_input.sql_input_all('SELECT id,address,call_help_address from t_first_aid')
#t_first_aid_addr.to_csv('first_aid_data//t_first_aid_addr.csv')