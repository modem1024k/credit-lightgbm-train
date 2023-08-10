import pandas as pd
import time
import tkinter as tk
from tkinter import filedialog
import xlrd
from xlrd import xldate_as_tuple
import numpy as np
import csv
from decimal import Decimal

#打开文件
def open_f():
    root = tk.Tk()   # 创建一个Tkinter.Tk()实例
    root.withdraw()  # 将Tkinter.Tk()实例隐藏
    
    # 选择多个个文件
    file_path = filedialog.askopenfilename(title='请选择多个文件', initialdir=r'.\\', filetypes=[
            ('Excel', '.xls .xlsx .csv'),('All Files', ' *')], defaultextension='.csv', multiple=False)
    
    
    print(file_path)
    return file_path

#生成dataframe
def pd_data(fname,*args):
    start= time.time()
    if fname.endswith('csv') :
        #df1 = pd.read_csv(fname,index_col=0,encoding="utf_8_sig")
        try :
            df1 = pd.read_csv(fname,index_col=0,encoding="ansi")
            print(fname,'ansi')
        except :
            df1 = pd.read_csv(fname,index_col=0)
            print(fname,'UTF8')
        #print(df1)
        #fpath=(fname.replace(fname.split('/')[-1],''))
        print ('csv参数',args[0])
        df1[args[0][2]] = df1[args[0][2]].apply(lambda x: str(Decimal(x)))
        #df1['合同号'] = df1['合同号'].fillna(0).astype(int)
        #df1['合同号'] = df1['合同号'].astype(int)
        #df1['合同号'] = df1['合同号'].astype(str)
        print(time.time() -start,fname)
        print(df1[0:3])
        #df2 = pd.read_csv(r"2.csv",index_col=0)
    if fname.endswith(('xlsx','xls')) :
        
        file = pd.ExcelFile(fname)
        print('表名',file.sheet_names)
        #df1 = file.parse(args[0],index_col = args[1])
        print ('excel参数',args[0])
        #df1 = file.parse(args[0][0],index_col = args[0][1])
        df1 = file.parse(file.sheet_names[args[0][0]],skiprows=args[0][3])  #表名
        #超长数字转文本,字段名
        #df1[args[0][2]] = df1[args[0][2]].apply(lambda x: str(Decimal(x)))
        print(df1[0: 3])
        print(time.time() -start,fname)
    return df1

#merge 两个dataframe
def pd_merge(df_1,df_2,*args):
    print (args[0])
    df = pd.merge(df_1,df_2,left_on=args[0][0],right_on=args[0][1])
    print(df)

def pd_merge_lr(df_1,df_2,*args):
    print (args[0])
    l1=len(args[0])
    print('长度',l1)
    df = pd.merge(df_1,df_2,on=args[0][0:l1-1],how=args[0][l1-1])
    print(df)

#聚合连接text
def join_text(df_1,k1,k2):
    grouped = df_1.groupby(k1)[k2].agg(lambda x: ','.join(x))
    return grouped

#供应链台账生成
def taizh(df_1,df_2):
    #df_1.loc[df_1['客户经理'].str.contains('（客户经理）'), '客户经理'].str.replace('（客户经理）','')
    df_1.loc[df_1['客户经理'].str.contains('（客户经理）'), '客户经理'] = df_1.loc[df_1['客户经理'].str.contains('（客户经理）'), '客户经理'].str.replace('（客户经理）', '')

    print(df_1[0:3])

    merged_table = pd.merge(df_1, df_2, left_on=['产品名称', '平台名称'], right_on=['产品名称2', '平台名称2'], how='left')
    merged_table.loc[merged_table['备注2'].notna(), '备注'] = merged_table['备注2']
    updated_table = merged_table.drop(columns=['产品名称2', '平台名称2', '备注2'])
    updated_table = updated_table.drop(columns=['企业规模','产品子名称','平台名称'])
    updated_table['业务发生日'] = pd.to_datetime(updated_table['业务发生日']).dt.strftime('%Y/%m/%d')
    #df['日期列名'] = pd.to_datetime(df['日期列名']).dt.strftime('%Y/%m/%d')
    #updated_table['业务发生日'] = updated_table['业务发生日'].dt.strftime('%Y/%m/%d')
    #updated_table['业务发生日'] = pd.to_datetime(updated_table['业务发生日'])
    updated_table['利率']=updated_table['利率']/100
    updated_table['币种']='人民币'
    updated_table = updated_table[updated_table['客户经理'] != '梅进健']
    updated_table = updated_table[updated_table['客户经理'] != '何洁']
    updated_table = updated_table[updated_table['客户经理'] != '徐晓光']
    updated_table = updated_table[updated_table['客户经理'] != '吴从宇']
    updated_table = updated_table[updated_table['客户经理'] != '闫晓英']

    updated_table = updated_table.sort_values('业务发生日')
    print(updated_table[0:3])
    updated_table.to_excel('temp'+time.strftime('%Y%m%d')+'.xlsx')


    return df_1,updated_table

if __name__ == "__main__":
    file1 = open_f()
    file2 = open_f()


    #定义excel表名,index字段,超长数值字段
    sh1=[0,'借据号','借据号',3]
    sh2=[0,'借据号','借据号',0]

    df1=pd_data(file1,sh1)  
    df2=pd_data(file2,sh2) 
    
    taizh(df1,df2)

    #df1=join_text(df1,'客户证件号','借据号')
    #df1.to_excel('合并借据.xlsx')

    #
    ##筛选记录
    #df_sel=df1.loc[(df1['转让合同号'] == 'ZRHT120004534575') & (df1['渠道号'] == '借钱优选'), ['转让合同号', '合同号']]
    #print(df_sel)
#
    #df_filtered = df1[df1['转让合同号'].str.contains('ZRHT')]
    #print(df_filtered)
#
    #df_filtered = df1[~df1['转让合同号'].str.contains('ZRHT')]
    #print(df_filtered)
    #
    #
#
    #df_filtered = df1[pd.to_datetime(df1['日期']) >= pd.to_datetime('2022-01-01')]
    #print(df_filtered)
    #print('#################')
    #
   #
#
    #file2 = open_f()
    #sh2=['Sheet1','借据号','合同号1']
    #df2 = pd_data(file2,sh2)
    
    #定义两个dataframe 的不同名关联字段
    #pd_merge(df1,df2,(['转让合同号','合同号'],['转让合同号','合同号1']))
    #print('################') 
    #
    ##定义两个dataframe 的同名关联字段,方向
    #pd_merge_lr(df1,df2,['转让合同号','left'])



    