import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np

# 加载数据
data = pd.read_csv('MX_data_cleaned.csv')

#print(data['MX37'])

#X = data.drop('tgt30_3m', axis=1)  # 特征矩阵
X = data.drop(['tgt30_3m', 'lst_prd', 'lst_prd_seg','CUST_ID','perf_date','weight'], axis=1)
y = data['tgt30_3m']  # 目标值


# 定义模型参数
#params = {
#    'boosting_type': 'gbdt',
#    'objective': 'binary',  # 根据实际情况调整
#    'metric': 'auc',
#    'verbosity': -1,
#    'seed': 42,
#    'nthread': 4
#}

## 进行5折交叉验证
#n_folds = 5
#stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
#cv_results = lgb.cv(params, lgb.Dataset(X, y), num_boost_round=1000, stratified=True, folds=stratified_kfold, metrics='auc', seed=42)
#
## 获取最佳迭代次数
#n_estimators = len(cv_results['auc-mean'])
#
## 在全量数据上训练模型
#lgb_train = lgb.Dataset(X, y)
#model = lgb.train(params, lgb_train, num_boost_round=n_estimators)
#
## 获取特征重要性并排序
#feature_importance = model.feature_importance(importance_type='gain')
#top_features_idx = np.argsort(feature_importance)[::-1]
#
## 输出最重要的500个特征
#num_top_features = 500
#top_500_features = X.columns[top_features_idx[:num_top_features]]
#print(f"Top {num_top_features} important features:")
#print(top_500_features)




# 定义模型参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',  # 根据实际情况调整
    'metric': 'auc',
    'verbosity': -1,
    'seed': 42,
    'nthread': 4,
    'max_bin': 63,  # 减少直方图的bins数量
    'min_data_in_bin': 3,  # 每个bin中的最小数据量
    'two_round': True  # 使用分位数端点切分算法
}

# 手动分块
num_blocks = 4  # 将数据划分为4个块
data_len = X.shape[0]
block_size = data_len // num_blocks
data_splits = []
for i in range(num_blocks):
    start = i * block_size
    end = (i + 1) * block_size if i < num_blocks - 1 else data_len
    X_split = X.iloc[start:end]
    y_split = y.iloc[start:end]
    data_splits.append(lgb.Dataset(X_split, y_split))

# 分别训练每个数据块的模型
models = []
for data in data_splits:
    model = lgb.train(params, data, num_boost_round=1000)
    models.append(model)

# 计算所有模型的平均特征重要性
total_importance = np.zeros(X.shape[1])
for model in models:
    total_importance += model.feature_importance(importance_type='gain')
avg_importance = total_importance / len(models)

# 根据平均特征重要性排序
top_features_idx = np.argsort(avg_importance)[::-1]

# 输出最重要的500个特征
num_top_features = 500
top_500_features = X.columns[top_features_idx[:num_top_features]]
print(f"Top {num_top_features} important features:")
for item1 in top_500_features:
    #print(top_500_features)
    print(item1)