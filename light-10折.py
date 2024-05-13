#增加分块读取训练集

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import joblib
import lightgbm as lightgbm
from sklearn.model_selection import StratifiedKFold
import gc

# 假设有一个名为'large_data.csv'的超大CSV文件
file_path = '..//TRN-CSV//rf600-TRN-cleaned.csv'
#file_path = '..//OOT1//OOT1_v.csv'
# 指定每次读取100000行
chunksize = 10000
# 初始化一个空的DataFrame来存放最终的结果
data = pd.DataFrame()
k=0
#title=['CUST_ID','lst_prd','lst_prd_seg','perf_date','tgt30_3m','weight']
#for i in range(1,1590):
#    title.append('MX'+str(i))
# 使用read_csv函数,指定chunksize参数
for chunk in pd.read_csv(file_path, chunksize=chunksize):
    # 在这里可以对每一个chunk数据进行处理
    processed_chunk = chunk   # 假设不需要处理,直接赋值
    
    # 将处理后的chunk添加到final_df
    data = pd.concat([data, processed_chunk])
    k=k+1
    print(k*chunksize)
    #if k>48:
    #    break
    del chunk
    del processed_chunk
    gc.collect()



# 加载数据
#data = pd.read_csv('..//TRN-CSV//rf600-TRN-cleaned.csv')
#data =data.replace(-9999, np.nan)
data = data.drop(['lst_prd','lst_prd_seg','perf_date','weight','CUST_ID','MX1277', 'MX130', 'MX439', 'MX954'],axis=1)

# 检查缺失值
print(data.isnull().sum())

# 将特征矩阵X和目标值y分离
X = data.drop(['tgt30_3m'], axis=1)
y = data['tgt30_3m']

# 对数值特征的缺失值进行简单имп纳值
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

# 对类别特征的缺失值进行最频繁值имп纳值
#categorical_features = X.select_dtypes(include=['object']).columns
#categorical_imputer = SimpleImputer(strategy='most_frequent')
#X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# 检查是否还有缺失值
print(X.isnull().sum())

# 对minority class进行过采样
#ros = RandomOverSampler(random_state=42)
#X_resampled, y_resampled = ros.fit_resample(X, y)


# 对类别型特征进行整数编码
categorical_features = X.select_dtypes(include=['object']).columns
encoders = {col: LabelEncoder() for col in categorical_features}
for col in categorical_features:
    X[col] = encoders[col].fit_transform(X[col])

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('划分成功')

# 创建LightGBM模型
# 设置scale_pos_weight参数
scale_pos_weight = 2/100

# 设置10折交叉验证
n_splits = 10
seed = 42
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

aucs = []
best_model = None  # 用于保存最好的模型
best_auc = 0  # 用于保存最高的AUC
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f'Fold {fold}')
    
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
 

    # 训练模型v1
    #lgb = LGBMClassifier(
    #    objective='binary', # 二分类问题
    #    metric='auc', # 评估指标为AUC
    #    n_estimators=4000, # 树的数量
    #    max_depth=6, # 树的最大深度
    #    num_leaves=63, # 叶子节点数
    #    learning_rate=0.05, # 学习率
    #    colsample_bytree=0.8, # 每棵树使用的特征比例
    #    subsample=0.9, # 每次迭代使用的样本比例
    #    reg_alpha=0.1, # L1正则化系数
    #    reg_lambda=0.1, # L2正则化系数
    #    random_state=42, # 随机种子
    #    n_jobs=-1, # 使用全部CPU核心
    #    scale_pos_weight=scale_pos_weight
    #)
    
    # 训练模型v2-auc-0.752-0.792
    #lgb = LGBMClassifier(                
    #    objective='binary', # 二分类问题
    #    metric='auc', # 评估指标为AUC
    #    n_estimators=2000, # 减小树的数量
    #    max_depth=4, # 减小树的最大深度
    #    num_leaves=31, # 减小叶子节点数
    #    learning_rate=0.03, # 减小学习率
    #    colsample_bytree=0.7, # 减小每棵树使用的特征比例
    #    subsample=0.8, # 减小每次迭代使用的样本比例
    #    reg_alpha=0.2, # 增大L1正则化系数
    #    reg_lambda=0.3, # 增大L2正则化系数
    #    min_child_samples=20, # 增大最小子样本数
    #    min_data_in_bin=5, # 增大每个节点最小数据量
    #    drop_rate=0.1, # 添加Dropout
    #    random_state=42, # 随机种子
    #    n_jobs=-1, # 使用全部CPU核心
    #    scale_pos_weight=scale_pos_weight
    #)
    
    # 训练模型v3-auc0.81(rf600-auc0.764-827)
    #lgb = LGBMClassifier(                
    #    objective='binary', # 二分类问题
    #    metric='auc', # 评估指标为AUC
    #    n_estimators=3000, # 减小树的数量
    #    max_depth=5, # 减小树的最大深度
    #    num_leaves=31, # 减小叶子节点数
    #    learning_rate=0.03, # 减小学习率
    #    colsample_bytree=0.7, # 减小每棵树使用的特征比例
    #    subsample=0.8, # 减小每次迭代使用的样本比例
    #    reg_alpha=0.2, # 增大L1正则化系数
    #    reg_lambda=0.3, # 增大L2正则化系数
    #    min_child_samples=20, # 增大最小子样本数
    #    min_data_in_bin=6, # 增大每个节点最小数据量
    #    drop_rate=0.1, # 添加Dropout
    #    random_state=42, # 随机种子
    #    n_jobs=-1, # 使用全部CPU核心
    #    scale_pos_weight=scale_pos_weight
    #)

    # 训练模型v4-optune
    lgb = LGBMClassifier(                
        objective='binary', # 二分类问题
        metric='auc', # 评估指标为AUC
        n_estimators=2900, # 减小树的数量
        max_depth=4, # 减小树的最大深度
        num_leaves=45, # 减小叶子节点数
        learning_rate=0.02, # 减小学习率
        colsample_bytree=0.7, # 减小每棵树使用的特征比例
        subsample=0.8, # 减小每次迭代使用的样本比例
        reg_alpha=1.7, # 增大L1正则化系数
        reg_lambda=0.11, # 增大L2正则化系数
        min_child_samples=49, # 增大最小子样本数
        min_data_in_bin=6, # 增大每个节点最小数据量
        drop_rate=0.1, # 添加Dropout
        random_state=42, # 随机种子
        n_jobs=-1, # 使用全部CPU核心
        scale_pos_weight=scale_pos_weight
    )
   
    
    # 训练模型
    callbacks = [lightgbm.early_stopping(stopping_rounds=40, verbose=False)]
    lgb.fit(X_train, y_train, 
            eval_set=(X_test, y_test), # 验证集
            eval_metric='auc', # 评估指标
            #early_stopping_rounds=50, # 早停回合数
            #verbose=True # 不显示训练过程
            callbacks=callbacks
            
        )
    
    # 计算训练集和测试集AUC
    train_auc = roc_auc_score(y_train, lgb.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, lgb.predict_proba(X_test)[:, 1])
    
    print(f'Train AUC: {train_auc}, Test AUC: {test_auc}')

    # 保存当前最好的模型
    if test_auc > best_auc:
        best_auc = test_auc
        best_model = lgb


# 保存模型到文件
joblib.dump(best_model, '..//pkl//rf600-422.pkl')

# 加载模型
loaded_model = joblib.load('..//pkl//rf600-422.pkl')

# 使用加载的模型进行预测
y_pred = loaded_model.predict_proba(X_test)[:, 1]
print('预测结果：', y_pred)

