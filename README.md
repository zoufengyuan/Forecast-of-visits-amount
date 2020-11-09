# 背景
主要是对医院患者数据和城市环境数据进行清洗，并建立模型预测后一天的门诊就诊量，结合医院急诊信息、环保数据和气象数据和门诊数据，用前若干天的特征去预测后一天的就诊量。
# 工程流程
## step 1 数据清洗
用python连接数据库读取原始数据，对各信息的数据表进行清洗得到清洗后的数据(见data_samples)
(先用sql对数据进行了一些描述统计)
主要的清洗有：
①应用文本相似度，对患者的疾病编码进行清洗，使各患者的疾病编码能与标准的ICD-10编码标准库进行匹配，清洗逻辑如下：
![icd编码清洗逻辑](https://github.com/zoufengyuan/Forecast-of-visits-amount/blob/main/icd%E7%BC%96%E7%A0%81%E6%B8%85%E6%B4%97%E9%80%BB%E8%BE%91.png)
②应用经纬度转化，对患者的地址编码进行清洗，得到每个患者的区域信息(如天河区)
![地址清洗逻辑](https://github.com/zoufengyuan/Forecast-of-visits-amount/blob/main/%E5%9C%B0%E5%9D%80%E6%B8%85%E6%B4%97%E9%80%BB%E8%BE%91.png)
③应用拉格朗日填补对环境数据按照时间顺序进行填补、整理年龄、性别、日期等字段
## step 2 特征工程
对原始数据进行清洗后得到各张信息完整的表，再结合建模目的，构造特征和目标变量，得到最终宽表
### data_utils文件：
对各表进行特征构建
空气质量数据整理，得到的字段有：
日期、地区、指标平均值(x_mean)、指标平均值(去除最大最小值)(x_mean_remove)、指标标准差(x_std)、指标标准差(去除最大值最小值)(x_std_remove)、指标多值特征(x_multi)
死亡数据进行整理 ，得到的字段有：
日期、地区、死亡人数(death_amount)、性别分布(death_gender)、年龄均值(death_age_mean)、年龄标准差(death_age_std)、年龄多值特征(death_age_multi)
对门诊数据进行整理：
①筛选出现住址或户籍地址在广州的患者,剔除patient_id为01的患者,根据patient_id、visit_date和gy_bm对门诊数据进行去重
②构造与patient_id无关的特征(字段)：
日期、地区、icd编码(disease)、就诊量(visit_amount)、性别分布(visit_gender)、年龄均值(visit_age_mean)、年龄标准差(visit_age_std)、年龄多值特征(visit_age_multi)
③应用word2vec构造patient_id相关的embedding特征:
日期、地区、icd编码、patient_id--通过每个patient_id的看病总类型构造sentence进行word embedding。
### data_merge文件
对处理好的宽表进行拼接
### data_history_merge文件
根据拼接好的宽表，进行时间滑窗处理，得到目标日期前七天每天的相关特征
## step 3 模型建立
起初运用ANN直接对分类变量进行embedding构造模型特征进行预测，后加入xdeepFM对多值特征进行向量转化，同时提取交叉特征，但效果均不佳
后去除多值特征，运用rf进行建模， 模型r2达到0.93，输出变量重要程度排序发现主要的影响因素为前几天的就诊量和当天是周几(很强的周期性)，与其他的环境变量关系不大，故运用ANN等方法对多值特征进行转化后，大大降低了这几个就诊量相关变量对目标变量的贡献能力，xdeepfm方法在此不可行。
最终的结果曲线如下：
![]


