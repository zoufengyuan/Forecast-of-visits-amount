# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pymysql
import pandas as pd
import gc
import time
import threading

class Sql_df(object):
    def __init__(self,input_db):
        self.host = '10.194.199.222'
        self.port = 3306
        self.username = 'root'
        self.password = 'aid_ms@2020'
        self.input_db = input_db
        self.conn = pymysql.connect(host = self.host,port = self.port,user = self.username,passwd = self.password,db = self.input_db,charset = 'utf8')
    def sql_input_all(self,sql_state):
        cur_1 = self.conn.cursor(cursor = pymysql.cursors.DictCursor)
        cur_1.execute(sql_state+' limit 1')
        column_df = cur_1.fetchall()
        column_list = column_df[0].keys()
        cur_2 = self.conn.cursor()
        start_time = time.time()
        cur_2.execute(sql_state)
        tmp_list = cur_2.fetchall()
        result_df = pd.DataFrame(list(tmp_list),columns = column_list)
        end_time = time.time()
        during_time = round(end_time-start_time,0)/60
        print('input data has spend %f minutes'%during_time)
        return result_df
    def sql_input_batch(self,sql_state,nums_sql_state,batch_size):
        cur_1 = self.conn.cursor(cursor = pymysql.cursors.DictCursor)
        cur_1.execute(sql_state+' limit 1')
        column_df = cur_1.fetchall()
        column_list = column_df[0].keys()
        cur_2 = self.conn.cursor()
        start_time = time.time()
        cur_2.execute(nums_sql_state)
        nums_sample = cur_2.fetchall()[0][0]
        batches = nums_sample//batch_size
        cur_3 = self.conn.cursor()
        result_df = pd.DataFrame()
        for i in range(batches):
            cur_3.execute(sql_state+' limit '+str(i*batch_size)+','+str(batch_size))
            tmp_list = list(cur_3.fetchall())
            tmp_df = pd.DataFrame(tmp_list,columns = column_list)
            del tmp_list
            gc.collect()
            result_df = result_df.append(tmp_df)
            del tmp_df
            gc.collect()
        last_index = batches*batch_size
        cur_3.execute(sql_state+' limit '+str(last_index)+','+str(nums_sample-last_index))
        tmp_list = list(cur_3.fetchall())
        tmp_df = pd.DataFrame(tmp_list,columns = column_list)
        result_df = result_df.append(tmp_df)
        end_time = time.time()
        during_time = round(end_time-start_time,0)/60
        print('input data has spend %f minutes'%during_time)
        del tmp_df
        gc.collect()
        return result_df

if __name__ == '__main__':
    #input_db = 'aid-livelihood'
    data_input = Sql_df('aid-livelihood')
    pa_visit_hypertension_2014_2016 = data_input.sql_input_all('select * from pa_visit_cerebral_infarction_2014_2015')
    #pa_visit_hypertension_2017_2018 = data_input.sql_input_all('select * from pa_visit_hypertension_2017_2018')
    #pa_inhosp_info = data_input.sql_input_all('select * from pa_inhosp_info')
    #rr = data_input.sql_input_batch('select * from pa_empi','select count(1) from pa_empi',5000)            
         
            
            
            
            
            
            
            
            
            
            
            
            


