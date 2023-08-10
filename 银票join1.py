import pandas as pd
import time
import tkinter as tk
from tkinter import filedialog
from decimal import Decimal

pd.options.display.float_format = '{:.2f}'.format  #设置两位小数
#打开文件
def open_f():
    root = tk.Tk()   # 创建一个Tkinter.Tk()实例
    root.withdraw()  # 将Tkinter.Tk()实例隐藏
    
    # 选择多个个文件
    file_path = filedialog.askopenfilename(title='请选择多个文件', initialdir=r'.\\', filetypes=[
            ('Excel', '.xls .xlsx .csv'),('All Files', ' *')], defaultextension='.csv', multiple=False)
    
    
    print(file_path)
    return file_path

'''
        df2['发票号码'] = df2['发票号码'].astype(str) 
        #去除空格
        df2['发票号码'] = df2['发票号码'].str.strip()
        #print(df2)
#del data['Unnamed: 0']

# on中填写根据哪个字段来进行连接，how为left代表left join
df = pd.merge(df1,df2, on='发票号码',how='left')
df = df.astype({'发票号码':'str'}) 
df.to_excel(fpath+'\\比对结果'+time.strftime("%m%d")+'.xlsx',sheet_name='data')

# 只取出这些列，组成新的DataFrame，带有col_name
df3 = df[['中登号码']]
# 将列进行重命名
new_col = ['中登编号']
df3.columns = new_col
#df.to_csv(path_or_buf=r"3.csv", encoding="utf_8_sig")

'''


if __name__ == "__main__":
    
    file1 = open_f()
    file = pd.ExcelFile(file1)
    shname=file.sheet_names[0]
    df1 = file.parse(shname)
    #print(df1)
    df1['银票累计'] = df1['合同金额'].cumsum()
    print(df1['银票累计'])
    
    file2 = open_f()
    file3 = pd.ExcelFile(file2)
    shname1=file3.sheet_names[0]
    df2=file3.parse(shname1)
    #df2['发票累计']=df2['发票金额'].cumsum().round(2).apply(lambda x: str(Decimal(x)))
    df2['发票累计']=df2['发票金额'].cumsum()
    print(df2['发票累计'])
    print(len(df2))
    fp=0
    mark=[]
    yp=[]
    for i in range(len(df1)):
        fphm=''
        fphm1=''
        fpje=''
        fpje1=''
        for j in range(fp,len(df2)):
            row1=[]
            if df1['银票累计'][i]>=df2['发票累计'][j]:
                #try :
                    #fphm=fphm+df2['发票号码'][j].astype(str)+';'
                fphm=str(df2['发票号码'][j])
                fpje=str(df2['发票金额'][j])
                row1=df1.iloc[i].tolist()
                
                row1.append(fphm)
                row1.append(fpje)
                print(row1)
                yp.append(row1)
                fp=fp+1
                #except :
                #    fphm=fphm+df2['发票号码'][j]+';'
                #    fp=fp+1    
            else :
                #try :
                fphm1=str(df2['发票号码'][j])
                fpje1=str(df2['发票金额'][j])
                row1=df1.iloc[i].tolist()
                print(row1)
                row1.append(fphm1)
                row1.append(fpje1)
                yp.append(row1)
                
                #except :
                #    fphm1=df2['发票号码'][j]    
                break    
        
        
        #print (i,fphm,fphm1) 
        #mark.append(fphm+fphm1)
    #print (yp)
    headtitles = df1.columns.tolist()
    headtitles.append('发票号码')
    headtitles.append('发票金额')
    df3=pd.DataFrame(yp,columns=headtitles)  #生成新的DF

    #df1['单据编号']=mark
    #print(df1)   
    df3.to_excel('银票配对'+time.strftime('%Y%m%d')+'.xlsx',index=False) 

