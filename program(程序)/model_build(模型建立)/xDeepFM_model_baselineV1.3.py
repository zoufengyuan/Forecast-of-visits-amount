# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:27:46 2020

@author: 86156
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers,Sequential,optimizers,losses,metrics,regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

source_model_df = pd.read_pickle('..//wide_data(宽表)//merged_history_data_all.pkl')
chosed_icd = ['I10_I15','I20_I25','I30_I52','I60_I69','J00_J06','J20_J22','J30_J39','J40_J47','R05']
source_model_df = source_model_df[source_model_df['disease'].isin(chosed_icd)]
#source_model_df['visit_amount'][source_model_df['visit_amount']<=5] = 0
#source_model_df['visit_amount'] = 40 - source_model_df['visit_amount']

drop_date_list = ['visit_date_target','visit_date_recent_visit','visit_date_recent_meteorological',
                  'visit_date_recent_air_quality','visit_date_recent_death']

multi_category_list = ['area','weekday_target','disease']
deleted_multi_list = ['visit_gender','visit_age_multi','so2_multi', 'no2_multi',
                       'co_multi', 'o38h_multi', 'pm10_multi', 'pm2_multi','death_gender','death_age_multi']
multi_category_list = multi_category_list+deleted_multi_list

std_scalar_list = ['visit_amount_recent','visit_age_mean','visit_age_std','so2_mean', 
                   'so2_std', 'so2_mean_remove', 'so2_std_remove', 'no2_mean',
                   'no2_std', 'no2_mean_remove', 'no2_std_remove', 'co_mean', 'co_std',
                   'co_mean_remove', 'co_std_remove', 'o38h_mean', 'o38h_std',
                   'o38h_mean_remove', 'o38h_std_remove', 'pm10_mean', 'pm10_std',
                   'pm10_mean_remove', 'pm10_std_remove', 'pm2_mean', 'pm2_std',
                   'pm2_mean_remove', 'pm2_std_remove','death_amount','death_age_mean',
                   'death_age_std','avg_pressure', 'avg_temp', 'avg_humidity', 
                   'precipitation','avg_wind_speed']

embedding_list = ['disease_patientid_embedding_'+str(i) for i in range(16)]
#source_model_df = source_model_df.drop(embedding_list+deleted_multi_list,axis = 1)

# In[]
#对数据进行缺失-1填补
#对数据拆分成训练集和测试集
#对相关字段进行删除
def fill_df(model_df,multi_category_list,std_scalar_list):
    for var in multi_category_list:
        model_df[var].fillna('-1',inplace = True)
    for var in std_scalar_list:
        model_df[var].fillna(-1,inplace = True)
    return model_df
def split_df(model_df):
    train_date = list(pd.date_range('20140101','20171231',freq = '1D'))
    test_date = list(pd.date_range('20180101','20181231',freq = '1D'))
    train_df = model_df[model_df['visit_date_target'].isin(train_date)]
    test_df = model_df[model_df['visit_date_target'].isin(test_date)]
    return train_df,test_df
def drop_df(model_df,drop_date_list):
    model_df = model_df.drop(drop_date_list,axis = 1)
    return model_df

source_model_df = fill_df(source_model_df,multi_category_list,std_scalar_list)
train_df,test_df = split_df(source_model_df)
train_df = drop_df(train_df,drop_date_list)
test_df = drop_df(test_df,drop_date_list)
y_scalar = StandardScaler()
train_df['visit_amount'] = y_scalar.fit_transform(train_df[['visit_amount']])
test_df['visit_amount'] = y_scalar.transform(test_df[['visit_amount']])


# In[]
#构造ANN模型实现实体嵌入
def hash_category_batch(category_df,category_hash_num):
    category_df = category_df.values
    for line in tqdm(category_df):
        for i in range(len(line)):
            line[i] = abs(hash('key_'+str(i)+'value_'+str(line[i])))%category_hash_num
    return category_df
def entity_embedding(hash_category_df,category_hash_num,emb_size):
    global category_emb_v2
    hash_category_tensor = tf.convert_to_tensor(hash_category_df,dtype = tf.int64)
    category_emb_v2 = tf.Variable(tf.random.normal([category_hash_num,emb_size],mean = 0.0,stddev= 1.0))
    category_emb_used = tf.gather(category_emb_v2,hash_category_tensor)
    category_emb_used_reshaped =  tf.reshape(category_emb_used,[-1,hash_category_df.shape[1]*emb_size])
    return category_emb_used_reshaped
def std_scalar(train_df,test_df,std_scalar_list):
    for var in tqdm(std_scalar_list):
        scalar = StandardScaler()
        train_df[var] = scalar.fit_transform(train_df[[var]]).flatten()
        test_df[var] = scalar.transform(test_df[[var]]).flatten()
    return train_df,test_df
def df_to_tensor(df):
    df_y = df.pop('visit_amount')
    x = tf.convert_to_tensor(np.array(df),dtype = tf.float32)
    y = tf.convert_to_tensor(df_y,dtype = tf.float32)
    return x,y

train_category_df = train_df[multi_category_list]
train_hash_category_df = hash_category_batch(train_category_df,int(5e6))
train_category_embed = entity_embedding(train_hash_category_df,int(5e6),16)

test_category_df = test_df[multi_category_list]
test_hash_category_df = hash_category_batch(test_category_df,int(5e6))
#test_category_embed = entity_embedding(test_hash_category_df,int(5e6),16)

train_df,test_df = std_scalar(train_df,test_df,std_scalar_list)
train_df = train_df.drop(multi_category_list,axis = 1)
test_df = test_df.drop(multi_category_list,axis = 1)

train_x,train_y = df_to_tensor(train_df)
test_x,test_y = df_to_tensor(test_df)
train_hash_category_df = tf.convert_to_tensor(train_hash_category_df,dtype = tf.int64)
test_hash_category_df = tf.convert_to_tensor(test_hash_category_df,dtype = tf.int64)
#tmp_train = tf.concat([train_x,train_category_embed],axis = 1)
#test_x = tf.concat([test_x,test_category_embed],axis = 1)

train_db = tf.data.Dataset.from_tensor_slices((train_x,train_hash_category_df,train_y)).shuffle(train_x.shape[0]).batch(64)
test_db = tf.data.Dataset.from_tensor_slices((test_x,test_hash_category_df,test_y)).shuffle(test_x.shape[0]).batch(64)
val_db = tf.data.Dataset.from_tensor_slices((test_x[:1000],test_hash_category_df[:1000],test_y[:1000])).shuffle(1000).batch(64)


class XDeepFM(keras.Model):
    def __init__(self):
        super(XDeepFM,self).__init__()   

    def base_model(self,cross_x,res = False,direct = False,bias = False,reduce = False,f_dim = 2):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = cross_x.shape[1]
        
        field_nums.append(field_num)
        hidden_nn_layers.append(cross_x)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0],cross_x.shape[2]*[1],2)
        for idx,layer_size in enumerate([128,128]):
            print(hidden_nn_layers[-1].shape)
            split_tensor = tf.split(hidden_nn_layers[-1],hidden_nn_layers[-1].shape[2]*[1],2)
            print(idx)
            print(split_tensor0[0].shape)
            print(split_tensor[0].shape)
            dot_result_m = tf.matmul(split_tensor0,split_tensor,transpose_b = True)
            dot_result_o = tf.reshape(dot_result_m,shape = [16,-1,field_nums[0]*field_nums[-1]])
            
            dot_result = tf.transpose(dot_result_o,perm = [1,0,2])
            
            filters = tf.compat.v1.get_variable('f_'+str(idx),shape = [1,field_nums[-1]*field_nums[0],layer_size],dtype = tf.float32)
            
            curr_out = tf.nn.conv1d(dot_result,filters = filters,stride = 1,padding = 'VALID')
            
            if bias:
                b = tf.compat.v1.get_variable(name = 'f_b'+str(idx),
                                              shape = [layer_size],
                                              dtype = tf.float32,
                                              initializer = tf.zeros_initializer())
                curr_out = tf.nn.bias_add(curr_out,b)
            
            curr_out = tf.nn.relu(curr_out)
            
            curr_out = tf.transpose(curr_out,perm = [0,2,1])
            
            if idx != 1:
                next_hidden,direct_connect = tf.split(curr_out,2*[int(layer_size/2)],1)
                final_len += int(layer_size/2)
            else:
                direct_connect = curr_out
                next_hidden = 0
                final_len += layer_size
            field_nums.append(int(layer_size/2))
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = tf.concat(final_result,axis = 1)
        result = tf.reduce_sum(result,-1)
        return result
    def call(self,hash_x):
        result = self.base_model(hash_x)
        return result

#hash_x = train_hash_category_df[:48]
#xdeepfm_model = XDeepFM()
#result = xdeepfm_model(hash_x)


# In[]
#ANN建模
class Network(keras.Model):
    def __init__(self):
        super(Network,self).__init__()
        self.embedding_1 = layers.Embedding(int(5e6), 16)
        self.g_avg_pool = layers.GlobalAveragePooling1D()
        self.layer_bn_1=layers.BatchNormalization()
        
        self.fc3 = layers.Dense(64)
        self.layer_bn_4=layers.BatchNormalization()
        
        self.fc4 = layers.Dense(32)
        self.layer_bn_5=layers.BatchNormalization()
        
        self.fc5 = layers.Dense(1)#若是分类问题需要加激活函数sigmoid或softmax
    def xdeepfm_model(self,cross_x,res = False,direct = False,bias = False,reduce = False,f_dim = 2):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = cross_x.shape[1]
        
        field_nums.append(field_num)
        hidden_nn_layers.append(cross_x)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0],cross_x.shape[2]*[1],2)
        for idx,layer_size in enumerate([128,128]):
            split_tensor = tf.split(hidden_nn_layers[-1],hidden_nn_layers[-1].shape[2]*[1],2)
            dot_result_m = tf.matmul(split_tensor0,split_tensor,transpose_b = True)
            dot_result_o = tf.reshape(dot_result_m,shape = [16,-1,field_nums[0]*field_nums[-1]])
            
            dot_result = tf.transpose(dot_result_o,perm = [1,0,2])
            
            filters = tf.compat.v1.get_variable('f_'+str(idx),shape = [1,field_nums[-1]*field_nums[0],layer_size],dtype = tf.float32)
            
            curr_out = tf.nn.conv1d(dot_result,filters = filters,stride = 1,padding = 'VALID')
            
            if bias:
                b = tf.compat.v1.get_variable(name = 'f_b'+str(idx),
                                              shape = [layer_size],
                                              dtype = tf.float32,
                                              initializer = tf.zeros_initializer())
                curr_out = tf.nn.bias_add(curr_out,b)
            
            curr_out = tf.nn.relu(curr_out)
            
            curr_out = tf.transpose(curr_out,perm = [0,2,1])
            
            if idx != 1:
                next_hidden,direct_connect = tf.split(curr_out,2*[int(layer_size/2)],1)
                final_len += int(layer_size/2)
            else:
                direct_connect = curr_out
                next_hidden = 0
                final_len += layer_size
            field_nums.append(int(layer_size/2))
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = tf.concat(final_result,axis = 1)
        result = tf.reduce_sum(result,-1)
        return result
        
    def call(self,input_x,input_hash,train_able = True,mask = None):
        hash_x = self.embedding_1(input_hash)
        #print(hash_x.shape)
        
        multi_cross = self.xdeepfm_model(hash_x)
        #hash_x = self.g_avg_pool(hash_x)
        hash_x = tf.reshape(hash_x,[-1,hash_x.shape[1]*hash_x.shape[2]])
        inputs = tf.concat([input_x,hash_x,multi_cross],axis = 1)
        
        inputs = tf.cond(train_able,lambda: self.layer_bn_1(inputs,training = True),lambda: self.layer_bn_1(inputs,training = False))
        
        self.fc_out_3 = self.fc3(inputs)
        self.fc_out_3 = tf.cond(train_able,lambda: self.layer_bn_4(self.fc_out_3,training = True),lambda: self.layer_bn_4(self.fc_out_3,training = False))
        self.fc_out_3 = tf.nn.relu(self.fc_out_3)
        
        
        self.fc_out_4 = self.fc4(self.fc_out_3)
        self.fc_out_4 = tf.cond(train_able,lambda: self.layer_bn_5(self.fc_out_4,training = True),lambda: self.layer_bn_5(self.fc_out_4,training = False))
        self.fc_out_4 = tf.nn.relu(self.fc_out_4)
        
        
        self.fc_out_5 = self.fc5(self.fc_out_4)
        return self.fc_out_5,inputs

model = Network()

learning_rate = 0.001
#learning_rate = tf.train.exponential_decay(0.1, 10, 100, 0.96, staircase=True) 
optimizer = tf.keras.optimizers.Adam(learning_rate)


train_tot_loss = []
train_tot_mae = []
test_tot_loss = []
test_tot_mae = []


for epoch in range(2):
    for step,(x,hash_x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out,train_inputs = model(x,hash_x)
            #loss = tf.losses.BinaryCrossentropy()(y,out)
            loss = tf.reduce_mean(tf.square(y-out))#MSE
            mae_loss = tf.reduce_mean(tf.losses.MAE(y,out))
        train_tot_loss.append(float(loss))
        train_tot_mae.append(float(mae_loss))
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        if step %10 == 0:
            test_loss,test_mae_loss = 0,0
            y_predict = []
            y_true = []
            for x,hash_x,y in test_db:
                test_out,test_inputs = model(x,hash_x,train_able = False)
                pre = list(y_scalar.inverse_transform(np.array(test_out).flatten()))
                y_predict.extend(pre)
                #test_loss = tf.losses.BinaryCrossentropy()(y,out)
                test_loss = tf.reduce_mean(tf.square(y-test_out))
            
                test_mae_loss = tf.reduce_mean(tf.losses.MAE(y,test_out))
                y = list(y_scalar.inverse_transform(np.array(y).flatten()))
                y_true.extend(y)
                
            #print(epoch,step,float(test_loss))
            test_tot_loss.append(float(test_loss))
            test_tot_mae.append(float(test_mae_loss))
        if step%100 == 0:
            print('%d epoch,%d step,train_loss: %f,test_loss: %f'%(epoch,step,float(loss),float(test_loss)))

plt.figure()
plt.plot(train_tot_loss, 'b', label = 'train')
plt.plot(test_tot_loss, 'r', label = 'test')
plt.xlabel('Step')
plt.ylabel('MAE')
plt.legend()
plt.savefig('train_test_auto-MPG.png')
plt.show()

r2 = r2_score(y_true,y_predict)
y_df = pd.DataFrame({'y_true':y_true,'y_pred':y_predict})
print(r2)










    
      
        
    
    








































