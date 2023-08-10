import os, sqlite3
import time
time_start=time.time()
db_file = os.path.join(os.path.dirname(__file__), r'D:\供应链\蔡杰敏-顾悦悦\供应链.db')
print (db_file)
conn = sqlite3.connect(db_file)

cursor = conn.cursor()
sqld='delete from 日均余额'
cursor.execute(sqld)
cursor.close()
conn.commit()

cursor = conn.cursor()
rq='2021-01-01'
for i in range(365):
    sql1="insert into 日均余额(余额,备注,日期,中心) select printf('%.2f',sum(金额)) as 余额,备注,strftime('%Y-%m-%d',"+"'"+rq+"','"+str(i)+" days') as 日期,max(所属中心) as 中心 from 供应链 where strftime('%Y-%m-%d',业务发生日)<=strftime('%Y-%m-%d',"+"'"+rq+"','"+str(i)+" days') group by 备注"
    #print (sql1)
    cursor.execute(sql1)
    print (i)
     
cursor.close()
conn.commit()

time_end=time.time()
print('totally cost',time_end-time_start)
