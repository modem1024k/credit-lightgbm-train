import pandas as pd
import numpy as np
import gc

# 假设有一个名为'large_data.csv'的超大CSV文件
#file_path = '..//TRN-csv//TRN_v.csv'
file_path = '..//OOT1//OOT1_v.csv'
# 指定每次读取100000行
chunksize = 10000

#sel=pd.read_csv('..//TRN-csv//rf-600-feature.csv')
sel=pd.read_csv('..//OOT1//rf-900_OOT.csv')
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
final_df = final_df.replace(-9999, np.nan)
final_df.to_csv('..//OOT1//rf900_OOT1.csv', index=False)