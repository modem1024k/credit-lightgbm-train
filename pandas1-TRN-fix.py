import pandas as pd
import numpy as np
import gc
from sklearn.impute import SimpleImputer

# 假设有一个名为'large_data.csv'的超大CSV文件
#file_path = '..//TRN-csv//TRN_v.csv'
file_path = '..//OOT2//OOT2_v.csv'
# 指定每次读取100000行
chunksize = 10000

#sel=pd.read_csv('..//TRN-csv//rf-600-feature.csv')
sel=pd.read_csv('..//OOT1//rf-600-feature-OOT1-fix.csv')
column_name = 'D2'
title = sel[column_name].tolist()
#sel_new=df[column_list]

# 初始化一个空的DataFrame来存放最终的结果
final_df = pd.DataFrame()
k=0
#title=['CUST_ID','lst_prd','lst_prd_seg','perf_date','tgt30_3m','weight']
#for i in range(1,1590):
#    title.append('MX'+str(i))
# 使用read_csv函数,指定chunksize参数
for chunk in pd.read_csv(file_path, chunksize=chunksize,encoding='ansi'):
    # 在这里可以对每一个chunk数据进行处理
    processed_chunk = chunk[title]   # 假设不需要处理,直接赋值
    # 将 -9999 和 -9990 替换为 np.nan
    processed_chunk.replace([-9999, -9990], np.nan, inplace=True)
    
    for col in title:
        # 查看每列的数据类型
        print('数据类型',col,processed_chunk[col].dtypes)
        string_mask = processed_chunk[col].apply(lambda x: isinstance(x, str))
        if string_mask.any():
            print(f'Column {col} contains string values.')
            # 将字符串值替换为np.nan
            processed_chunk.loc[string_mask, col] = np.nan


    # 对数值特征的缺失值进行简单имп纳值
    numeric_features = processed_chunk.select_dtypes(include=['int64', 'float64']).columns
    numeric_imputer = SimpleImputer(strategy='mean')
    processed_chunk[numeric_features] = numeric_imputer.fit_transform(processed_chunk[numeric_features])

 
    
    # 将处理后的chunk添加到final_df
    final_df = pd.concat([final_df, processed_chunk])
    k=k+1
    print(k*chunksize)
    #if k>48:
    #    break
    del chunk
    del processed_chunk
    gc.collect()


# 现在final_df包含了整个CSV文件的内容
#print(final_df)
#final_df = final_df.replace(-9999, np.nan)
final_df.to_csv('..//OOT2//rf600_OOT2_fix.csv', index=False)