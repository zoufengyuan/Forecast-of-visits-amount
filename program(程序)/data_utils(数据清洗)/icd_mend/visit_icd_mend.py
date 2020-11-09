# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:22:28 2020

@author: 86156
"""

import pandas as pd
import numpy as np
import re
from tfidf_like import Text_similarity,Text_similarity_2
from tqdm import tqdm
import os
from multiprocessing import Pool
import gc
import time
import warnings
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
text_test = Text_similarity()
text_test_2 = Text_similarity_2()
def tran_main(x):
    if str(x) == 'nan':
        return np.nan
    elif '-' in x:
        tmp = re.findall(r'[A-Za-z\. ]|\d+[-][A-Za-z\. ]|\d+', x)
        result = ''.join(tmp)
    else:
        tmp = re.findall(r'[A-Za-z\. ]|\d+', x)
        result = ''.join(tmp)
    return result
def name_tran(x):

    regStr = ".*?([\u4E00-\u9FA5]+).*?"
    mend_res = re.findall(regStr, x)
    if '型糖尿病' in mend_res:
        return '非胰岛素依赖型糖尿病'
    else:
        if len(mend_res) == 0:
            return np.nan
        else:
            tmp_result = {}
            for des in mend_res:
                result,val = text_test.jieba_cut(des)
                tmp_result[result] = [des,val]
            result = sorted(tmp_result.items(),key = lambda x:x[1][1],reverse = True)
            if result[0][1][1]<0.3:
                last_result = result[0][1][0]
            else:
                last_result = result[0][0]
                
            return last_result
def main(visit_df_file):
    global icd_df
    global text_test_2
    visit_df = pd.read_csv('visit_data//'+visit_df_file)[:10000]
    #visit_df['disease_diag_name'] = visit_df['disease_diag_name'].apply(lambda x:(x.strip() if str(x)!= 'nan'))
    #visit_df['disease_diag_code'] = visit_df['disease_diag_code'].apply(lambda x:(x.strip() if str(x)!= 'nan'))
    visit_df['ICD编码'] = visit_df['disease_diag_code'].apply(lambda x:tran_main(x))
    visit_df['disease_diag_name'][visit_df['disease_diag_name'].isin(['-','无'])] = np.nan
    
    #先以visit_df中地icd编码为基础进行匹配
    visit_df_1 = visit_df.merge(icd_df[['ICD编码','ICD中文名称']],on = 'ICD编码',how = 'left')
    
    visit_df_2 = visit_df_1[visit_df_1['ICD中文名称'].isna() == False]#已处理好的名称和编码
    visit_df_3 = visit_df_1[visit_df_1['ICD中文名称'].isna() == True]#根据归一诊断进行处理
    
    del visit_df_3['ICD中文名称']
    visit_df_3.rename(columns = {'disease_diag_name':'ICD中文名称'},inplace = True)
    icd_df_2 = icd_df.copy()
    icd_df_2.rename(columns = {'ICD编码':'ICD编码_new'},inplace = True)
    visit_df_4 = visit_df_3.merge(icd_df_2[['ICD中文名称','ICD编码_new']],on = 'ICD中文名称',how = 'left')
    
    visit_df_5 = visit_df_4[visit_df_4['ICD编码_new'].isna() == False]#已处理好的名称和编码
    visit_df_5['ICD编码'] = visit_df_5['ICD编码_new']
    del visit_df_5['ICD编码_new']
    visit_df_5['disease_diag_name'] = visit_df_5['ICD中文名称']
    visit_df_5 = visit_df_5.loc[:,['id','disease_diag_name','disease_diag_code','gy_diag','ICD编码','ICD中文名称']]
    
    visit_df_6 = visit_df_4[visit_df_4['ICD编码_new'].isna() == True]#
    del visit_df_6['ICD编码_new']
    
    
    visit_df_7 = visit_df_6[(visit_df_6['ICD中文名称'].isna() == True)&(visit_df_6['ICD编码'].isna() == True)]#无法进行处理地
    visit_df_7['disease_diag_name'] = visit_df_7['ICD中文名称']
    visit_df_7 = visit_df_7.loc[:,['id','disease_diag_name','disease_diag_code','gy_diag','ICD编码','ICD中文名称']]
    
    visit_df_8 = visit_df_6[(visit_df_6['ICD中文名称'].isna() == False)&(visit_df_6['ICD编码'].isna() == True)]#通过疾病名称去整理的数据集
    
    visit_df_9 = visit_df_6[(visit_df_6['ICD中文名称'].isna() == True)&(visit_df_6['ICD编码'].isna() == False)]#通过ICD编码去整理的数据集
    del visit_df_9['ICD中文名称']
    visit_df_9['ICD编码'] = visit_df_9['ICD编码'].apply(lambda x:text_test_2.jieba_cut(x))
    visit_df_9 = visit_df_9.merge(icd_df[['ICD编码','ICD中文名称']],on = 'ICD编码',how = 'left')
    visit_df_9 = visit_df_9.loc[:,['id','disease_diag_name','disease_diag_code','gy_diag','ICD编码','ICD中文名称']]
    
    visit_df_10 = visit_df_6[(visit_df_6['ICD中文名称'].isna() == False)&(visit_df_6['ICD编码'].isna() == False)]#结合两者去整理的数据集
    
    #对visit_df_8进行处理，将疾病名称进行处理
    #fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE)
    #fil.sub(' ', a)
    
    
    visit_df_8['ICD中文名称_new'] = visit_df_8['ICD中文名称'].apply(lambda x:name_tran(x))
    del visit_df_8['ICD编码']
    icd_df_3 = icd_df.copy()
    icd_df_3.rename(columns = {'ICD中文名称':'ICD中文名称_new'},inplace = True)
    visit_df_8 = visit_df_8.merge(icd_df_3[['ICD中文名称_new','ICD编码']],on = 'ICD中文名称_new',how = 'left')
    visit_df_8.rename(columns = {'ICD中文名称':'disease_diag_name'},inplace = True)
    visit_df_8.rename(columns = {'ICD中文名称_new':'ICD中文名称'},inplace = True)
    visit_df_8 = visit_df_8.loc[:,['id','disease_diag_name','disease_diag_code','gy_diag','ICD编码','ICD中文名称']]
    
    
    
    visit_df_10['ICD编码_new'] = visit_df_10['ICD编码'].apply(lambda x:text_test_2.jieba_cut(x))
    visit_df_10.rename(columns = {'ICD中文名称':'disease_diag_name'},inplace = True)
    icd_df_4 = icd_df.copy()
    icd_df_4.rename(columns = {'ICD编码':'ICD编码_new'},inplace = True)
    visit_df_10 = visit_df_10.merge(icd_df_4[['ICD编码_new','ICD中文名称']],on = 'ICD编码_new',how = 'left')
    cond = visit_df_10['ICD中文名称'].isna() == True
    
    visit_df_10.loc[cond,'ICD中文名称'] = visit_df_10.loc[cond,'disease_diag_name'].values
    del visit_df_10['ICD编码']
    visit_df_10.rename(columns = {'ICD编码_new':'ICD编码'},inplace = True)
    visit_df_10 = visit_df_10.loc[:,['id','disease_diag_name','disease_diag_code','gy_diag','ICD编码','ICD中文名称']]
    cond1 = visit_df_10['ICD编码'].isna() == True
    visit_df_10.loc[cond1,'ICD编码'] = visit_df_10.loc[cond1,'disease_diag_code'].values
    
    
    
    last_visit_df = pd.DataFrame()
    all_df = [visit_df_2,visit_df_5,visit_df_7,visit_df_8,visit_df_9,visit_df_10]
    for df in all_df:
        last_visit_df = last_visit_df.append(df)
    
    last_visit_df = last_visit_df.drop_duplicates('id',keep = 'last')
    return last_visit_df

def new_main(df_dict,file,i):
    batch_size = 10000
    icd_df = pd.read_excel('ICD-10编码.xls')
    #data = pd.read_csv('mended_visit_data//'+file)
    #data = df_dict[file]
    #mended_data = data[(data['disease_diag_code'].isna() == False)&(data['ICD编码'].isna() == True)]
    mended_data = df_dict[file]
    del mended_data['ICD编码']
    mended_data = mended_data[i*batch_size:(i+1)*batch_size]
    mended_data['ICD中文名称_new'] = mended_data['ICD中文名称'].apply(lambda x:name_tran(x))
    icd_df_3 = icd_df.copy()
    icd_df_3.rename(columns = {'ICD中文名称':'ICD中文名称_new'},inplace = True)
    mended_data = mended_data.merge(icd_df_3[['ICD中文名称_new','ICD编码']],on = 'ICD中文名称_new',how = 'left')
    del mended_data['ICD中文名称']
    mended_data.rename(columns = {'ICD中文名称_new':'ICD中文名称'},inplace = True)
    mended_data = mended_data.loc[:,['id','disease_diag_name','disease_diag_code','gy_diag','ICD编码','ICD中文名称']]
    mended_data = mended_data.drop_duplicates('id',keep = 'first')
    mended_data.to_csv('second_visit_data_new//'+file+'_'+str(i)+'.csv',index = None)
    
    

if __name__ == '__main__':
    icd_df = pd.read_excel('ICD-10编码.xls')
    visit_list = os.listdir('mended_visit_data')
    print("{beg}并行程序{beg}".format(beg='-'*16))
    startTime = time.time()
    '''
    for i in range(300):
        for file in tqdm(visit_list):
            #try:
            new_main(file,i)
            #except:
            print(i,file)
    '''
    df_dict = {}
    for file in tqdm(visit_list):
        data = pd.read_csv('mended_visit_data//'+file)
        df_dict[file] = data[(data['disease_diag_code'].isna() == False)&(data['ICD编码'].isna() == True)&(data['ICD中文名称'].isna() == False)]
        print(df_dict[file].shape)
        del data
        gc.collect()
    '''
    for i in tqdm(range(30)):
        for file in tqdm(visit_list[:4]):
            new_main(file,i)
    '''
    for i in tqdm(range(30)):
        Parallel(n_jobs=-1, backend='multiprocessing',verbose = 1)(delayed(new_main)(df_dict,file,i) for file in tqdm(visit_list))
    print("用时:%.3fs"%( time.time()-startTime ))
    '''
    print("{beg}串行程序{beg}".format(beg='-'*16))
    startTime = time.time()
    for file in tqdm(visit_list[2:3]):
        result_df = main(file)
        result_df.to_csv('visit_data_new//'+file,index = None)
    print("用时:%.3fs"%( time.time()-startTime ) )
    '''
    
    '''
    
    print("{beg}并行程序{beg}".format(beg='-'*16))
    startTime = time.time()
    results = Parallel(n_jobs=-1, backend='loky',verbose = 1)(delayed(main)(file) for file in tqdm(visit_list[2:3]))
    for i,df in enumerate(results):
        df.to_csv('visit_data_new//'+visit_list[i],index = None)
    print(len(results))
    print("用时:%.3fs"%( time.time()-startTime ))
    '''
    
    















