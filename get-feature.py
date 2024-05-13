import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import time
from sklearn.model_selection import train_test_split
import gc
# 加载数据
#data = pd.read_csv('feature_500_cleaned.csv')
#data = data.drop(['lst_prd','lst_prd_seg','perf_date','weight','CUST_ID'], axis=1)

# 设置每批加载的行数
chunksize = 10000
# 加载数据集(假设数据集存储在data.csv文件中)
data_iter = pd.read_csv('feature_500_cleaned.csv',chunksize=chunksize, iterator=True)
#columns = next(data_iter).columns  # 获取列名
data = pd.DataFrame()
k=0
for chunk in data_iter:
    # 在这里可以对每一个chunk数据进行处理
    processed_chunk = chunk   # 假设不需要处理,直接赋值
    
    # 将处理后的chunk添加到final_df
    data = pd.concat([data, processed_chunk])
    k=k+1
    print(k*chunksize)
    del chunk  # 删除块以释放内存
    gc.collect()  # 强制进行垃圾回收
data = data.drop(['lst_prd','lst_prd_seg','perf_date','weight','CUST_ID'], axis=1)    




# 检查缺失值
print(data.isnull().sum())

# 将特征矩阵X和目标值y分离
X = data.drop(['tgt30_3m'], axis=1)
y = data['tgt30_3m']

# 对数值特征的缺失值进行简单имп纳值
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])


# 检查是否还有缺失值
print(X.isnull().sum())


# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('划分成功')

strart=time.time()

'''
# F检验特征选择
num_feats = 300
f_scores = f_classif(X, y)[0]
f_feats = np.argsort(f_scores)[-num_feats:][::-1]
f_feats1 = X.columns[f_feats].tolist()

# 保存F检验选择的特征到CSV文件
f_feats_df = pd.DataFrame({'F-test Features': f_feats1})
f_feats_df.to_csv('f_test_features.csv', index=False)

# 使用选择的特征训练LightGBM模型并评估
X_f = X.iloc[:, f_feats]
lgb_f = LGBMClassifier(objective='binary', # 二分类问题
                        metric='auc', # 评估指标为AUC
                        n_estimators=4000, # 树的数量
                        max_depth=6, # 树的最大深度
                        num_leaves=63, # 叶子节点数
                        learning_rate=0.05, # 学习率
                        colsample_bytree=0.8, # 每棵树使用的特征比例
                        subsample=0.9, # 每次迭代使用的样本比例
                        reg_alpha=0.1, # L1正则化系数
                        reg_lambda=0.1, # L2正则化系数
                        random_state=42, # 随机种子
                        n_jobs=-1, # 使用全部CPU核心
                        scale_pos_weight=50,
                        verbose=-1)
lgb_f.fit(X_f, y)
y_pred_f = lgb_f.predict_proba(X_test.iloc[:, f_feats])[:, 1]
auc_f = roc_auc_score(y_test, y_pred_f)
print(f'AUC score with F-test features: {auc_f}')
print('F检验特征选择耗时：', time.time()-strart)

# 互信息特征选择
mi_scores = mutual_info_classif(X, y, random_state=0)
mi_feats = np.argsort(mi_scores)[-num_feats:][::-1]

mi_feats1 = X.columns[mi_feats].tolist()

# 保存互信息选择的特征到CSV文件
mi_feats_df = pd.DataFrame({'Mutual Info Features': mi_feats1})
mi_feats_df.to_csv('mi_features.csv', index=False)

# 使用选择的特征训练LightGBM模型并评估
X_mi = X.iloc[:, mi_feats]
lgb_mi = LGBMClassifier(objective='binary', # 二分类问题
                        metric='auc', # 评估指标为AUC
                        n_estimators=4000, # 树的数量
                        max_depth=6, # 树的最大深度
                        num_leaves=63, # 叶子节点数
                        learning_rate=0.05, # 学习率
                        colsample_bytree=0.8, # 每棵树使用的特征比例
                        subsample=0.9, # 每次迭代使用的样本比例
                        reg_alpha=0.1, # L1正则化系数
                        reg_lambda=0.1, # L2正则化系数
                        random_state=42, # 随机种子
                        n_jobs=-1, # 使用全部CPU核心
                        scale_pos_weight=50,
                        verbose=-1)
lgb_mi.fit(X_mi, y)
y_pred_mi = lgb_mi.predict_proba(X_test.iloc[:, mi_feats])[:, 1]
auc_mi = roc_auc_score(y_test, y_pred_mi)
print(f'AUC score with mutual info features: {auc_mi}')
print('互信息特征选择耗时：', time.time()-strart)

'''
# Embedded方法: 使用递归特征消除(RFE)
estimator = LogisticRegression()
rfe = RFE(estimator, n_features_to_select=300)
rfe_feats = rfe.fit(X, y).get_support(indices=True)

rfe_feats1 = X.columns[rfe_feats].tolist()

# 保存Embedded方法选择的特征到CSV文件
rfe_feats_df = pd.DataFrame({'F-test Features': rfe_feats1})
rfe_feats_df.to_csv('rfe_test_features.csv', index=False)

print('Embedded方法特征选择耗时：', time.time()-strart)

