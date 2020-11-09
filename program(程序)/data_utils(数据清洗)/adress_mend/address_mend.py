# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:38:12 2020

@author: FengY Z
"""

import pandas as pd
import json
from urllib.request import urlopen, quote
import requests
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim,GoogleV3
from geopy.exc import GeocoderTimedOut
from time import time
import datetime,time
from tqdm import tqdm
from joblib import Parallel, delayed
empi = pd.read_csv(r'D:/zoufengyuan/200813广州疾控中心/t_first_aid_data/t_first_aid_addr.csv')

addr_flag = empi['address'].isna()
call_addr_flag = empi['call_help_address'].isna()
new_addr = []
for i in tqdm(range(len(empi))):
    if addr_flag[i] == False:
        addr = empi['address'].iloc[i]
    else:
        if call_addr_flag[i] == False:
            addr = empi['call_help_address'].iloc[i]
        else:
            addr = np.nan
    new_addr.append(addr)
empi['new_addr'] = new_addr




def word_find(x):
    findword = u'(越秀|荔湾|海珠|天河|白云|黄埔|番禺|花都|南沙|增城|从化)'
    pattern = re.compile(findword)  
    results =  pattern.findall(x)
    if len(results)!=0:
        result = '广州市'+results[0]+'区'
    else:
        result = '外地'
    return result


def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoding/v3/'
    output = 'json'
    ak = 'K9CGyI3cCOTIUTtnUMh7rsVlBLg0X24c' # 百度地图ak，具体申请自行百度，提醒需要在“控制台”-“设置”-“启动服务”-“正逆地理编码”，启动
    address = quote('广东省广州市'+address) # 由于本文地址变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + address  + '&output=' + output + '&ak=' + ak  +'&callback=showLocation%20'+'//GET%E8%AF%B7%E6%B1%82'
#     req = urlopen(uri)
#     res = req.read().decode() 这种方式也可以，和下面的效果一样，都是返回json格式
    res=requests.get(uri).text
    temp = json.loads(res) # 将字符串转化为json
    lat = temp['result']['location']['lat']
    lng = temp['result']['location']['lng']
    return lat,lng   # 纬度 latitude,经度 longitude 

def getlocation(lat,lng):
    ak = 'K9CGyI3cCOTIUTtnUMh7rsVlBLg0X24c'
    town = 'true'
    uri = 'http://api.map.baidu.com/reverse_geocoding/v3/?output=json&ak=%s&location=%s,%s&extensions_town=%s'%(ak,lat,lng,town)
    html = requests.get(uri)
    bs_get_detail = BeautifulSoup(html.text,'lxml')
    location = eval(bs_get_detail.p.text)['result']['addressComponent']['district']
    return '广州市'+location
    
    
# In[]
#对gyxzd进行处理

std_area = ['广州市越秀区','广州市荔湾区','广州市海珠区','广州市天河区','广州市白云区','广州市黄埔区',
            '广州市番禺区','广州市花都区','广州市南沙区','广州市增城区','广州市从化区',
            '广州市越秀区\n','广州市荔湾区\n','广州市海珠区\n','广州市天河区\n','广州市白云区\n','广州市黄埔区\n',
            '广州市番禺区\n','广州市花都区\n','广州市南沙区\n','广州市增城区\n','广州市从化区\n']
xzd_mend_df = empi[['id','addr_home','gyxzd']][~empi['gyxzd'].isin(std_area)]#处理好的不进行处理了
xzd_mend_df = xzd_mend_df[xzd_mend_df['addr_home'].isna() == False]#不处理addr_home是空的

xzd_mend_df['gyxzd_tmp'] = xzd_mend_df['addr_home'].apply(lambda x:word_find(x))

xzd_mend_df_1 = xzd_mend_df[xzd_mend_df['gyxzd_tmp']!='外地']#不需要处理了
xzd_mend_df_2 = xzd_mend_df[xzd_mend_df['gyxzd_tmp']=='外地']#需要处理

xzd_mend_df_3 = xzd_mend_df_2[~xzd_mend_df_2['gyxzd'].isin(['异常值','广州市','异常值\n','广州市\n','广州广州'])]#再将无法处理的排除


def waitToTomorrow():
     tomorrow = datetime.datetime.replace(datetime.datetime.now() + datetime.timedelta(days=1), 
           hour=0, minute=0, second=0)
     delta = tomorrow - datetime.datetime.now()
     time.sleep(delta.seconds)

need_1 = empi[['id','addr_home','gyxzd']][empi['gyxzd'].isin(std_area)]
xzd_mend_df_0 = empi[['id','addr_home','gyxzd']][~empi['gyxzd'].isin(std_area)]
need_2 = xzd_mend_df_0[xzd_mend_df_0['addr_home'].isna() == True]
need_3 = xzd_mend_df_2[xzd_mend_df_2['gyxzd'].isin(['异常值','广州市','异常值\n','广州市\n','广州广州'])]
need_4 = xzd_mend_df_1
need_1['gyxzd_tmp'] = need_1['gyxzd']
need_2['gyxzd_tmp'] = need_2['gyxzd']
need = pd.concat([need_1,need_2,need_3,need_4],axis = 0)
#need.to_csv('t_first_aid_data//t_first_aid_addr_0.csv',index = None)

'''
batch_size = 250000


#batch_num = 1
batch = 0
mend_df = xzd_mend_df_3[batch*batch_size:(batch+1)*batch_size]

def last_main(i):
    global mend_df
    global batch
    size = 10000
    gyxzd = []
    tmp_df = mend_df[i*size:(i+1)*size]
    for sdd in tqdm(tmp_df['addr_home']):
        try:
            lat,lng = getlnglat(sdd)
            new_add = getlocation(lat,lng)
            #print(new_add)
        except:
            new_add = '外地'
            #print(new_add)
        gyxzd.append(new_add)
    tmp_df['gyxzd_tmp'] = gyxzd
    tmp_df.to_csv('new_empi_data_2//empi_gyxzd_new_'+str(batch+1)+'_'+str(i)+'.csv',index = None)

Parallel(n_jobs=-1, backend='loky',verbose = 1)(delayed(last_main)(i) for i in range(25))
#last_main(1)

mend_df = pd.read_csv('empi_gyxzd_4.csv')

gyxzd = []
waitToTomorrow()
for sdd in tqdm(mend_df['addr_home']):
    try:
        lat,lng = getlnglat(sdd)
        new_add = getlocation(lat,lng)
    except:
        new_add = '外地'
    gyxzd.append(new_add)
mend_df['gyxzd_tmp'] = gyxzd
mend_df.to_csv('empi_gyxzd_'+str(4)+'.csv',index = None)
'''
























