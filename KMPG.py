# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:02:54 2021

@author: sRv
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from datetime import date
import matplotlib as plt

os.chdir("D:\Temp")
data = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx')
df_trans = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name = 1, index_col=2)
df_NCL = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name = 2)
df_CD = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name = 3, index_col=0)
df_CA = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name = 4, index_col=0)


#TASK: 1 Data Quality Assessment

#1. Transaction

df_trans["product_first_sold_date"] = pd.to_datetime(df_trans["product_first_sold_date"]).dt.date
df_trans.describe()
df_trans.isnull().sum()
df_trans.dropna()
df_trans.columns
df_trans['online_order'].value_counts()
df_trans['order_status'].value_counts()
df_trans['brand'].value_counts()
df_trans['product_line'].value_counts()
df_trans['product_class'].value_counts()
df_trans['product_size'].value_counts()
df_trans['product_first_sold_date'].value_counts()
dup = df_trans.duplicated()
df_trans[dup].sum()
df_trans.to_csv(r"D:\Temp\Transaction.csv", index=False)

#2. NewCustomerList

df_NCL.info()
df_NCL.columns
dcols=['Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20']
df_NCL = df_NCL.drop(dcols, axis=1, inplace = True)
df_NCL.describe()
df_NCL.isnull().sum()
df_NCL.duplicated().sum()
df_NCL.columns
df_NCL["gender"].value_counts()
df_NCL["gender"] = df_NCL["gender"].replace("U","Unidentified")
df_NCL["job_industry_category"].value_counts()
df_NCL["wealth_segment"].value_counts()
df_NCL["deceased_indicator"].value_counts()
df_NCL["owns_car"].value_counts()
df_NCL["country"].value_counts()
df_NCL.to_csv(r"D:\Temp\NewCustomerList.csv", index=False)

#3. CustomerDemographic

df_CD.info()
df_CD = df_CD.drop("default", axis=1)
df_CD.describe()
df_CD.isnull().sum()
df_CD.duplicated().sum()
df_CD["gender"].value_counts()
df_CD["gender"] = df_CD["gender"].replace("Femal","Female").replace("F","Female").replace("M","Male").replace("U","Unidentified")
df_CD.columns
df_CD.to_csv(r"D:\Temp\CustomerDemographic.csv", index=False)

#4. CustomerAddress

df_CA.info()
df_CA.describe()
df_CA.isnull().sum()
df_CA.duplicated().sum()
df_CA.columns
df_CA["state"].value_counts()
df_CA["country"].value_counts()
df_CA.to_csv(r"D:\Temp\CustomerAddress.csv", index=False)




# TASK: 2 Data Insights

#1. Transaction

df_trans["product_first_sold_date"] = pd.to_datetime(df_trans["product_first_sold_date"]).dt.date
df_trans.describe()
df_trans.isnull().sum()
df_trans.info()
df_trans["online_order"] = df_trans["online_order"].fillna(df_trans["online_order"].median())
df_trans["Profit"] = df_trans["list_price"] - df_trans["standard_cost"]
df_trans = df_trans.reset_index()
df_trans["Transaction_Month"] = df_trans["transaction_date"].dt.month_name()
df_trans["Recency"] = 0
df_trans["transaction_date"] = df_trans["transaction_date"].dt.date
df_trans["transaction_date"].unique()
for i in range(0,len(df_trans),1):
     df_trans["Recency"][i] = date(2017, 12, 31) - df_trans["transaction_date"][i]
df_trans.set_index("customer_id", inplace=True)
df_trans.sort_index(inplace = True)
df_trans_cleaned = df_trans.dropna()

#3. CustomerDemographic

df_CD.info()
df_CD = df_CD.drop("default", axis=1)
df_CD.describe()
df_CD.isnull().sum()
df_CD.duplicated().sum()
df_CD["gender"].value_counts()
df_CD["gender"] = df_CD["gender"].replace("Femal","Female").replace("F","Female").replace("M","Male").replace("U","Unidentified")
df_CD["tenure"] = df_CD["tenure"].fillna(df_CD["tenure"].mean())
df_CD["Age"] = ""
df_CD["Age_Category"] = ""
for i in range(1,len(df_CD)+1,1):
    df_CD["Age"][i] = date.today().year - df_CD["DOB"][i].year
    if df_CD["Age"][i] <= 20:
        df_CD["Age_Category"][i] = "Child(<20)"
    elif df_CD["Age"][i] <= 30:
        df_CD["Age_Category"][i] = "Young(21-30)"
    elif df_CD["Age"][i] <= 40:
        df_CD["Age_Category"][i] = "Younger Middle Age(31-40)"
    elif df_CD["Age"][i] <= 50:
        df_CD["Age_Category"][i] = "Older Middle Age(41-50)"
    elif df_CD["Age"][i] <= 60:
        df_CD["Age_Category"][i] = "Elder(51-60)"
    else:
        df_CD["Age_Category"][i] = "Old(>60)"   
df_CD["Age"] = pd.to_numeric(df_CD["Age"])
df_CD["Age"] = df_CD["Age"].fillna(df_CD["Age"].mean())

#4. CustomerAddress

df_CA.info()
df_CA.describe()
df_CA.isnull().sum()
df_CA.duplicated().sum()
df_CA["state"] = df_CA["state"].replace('NSW','New South Wales').replace('VIC','Victoria').replace("M","Male")

# Combined Data

df_CA_CD = pd.merge(df_CD, df_CA, on = "customer_id", how = "left")
df = pd.merge(df_CA_CD, df_trans, on = "customer_id", how = "left")
df.isnull().sum()
df.info()
df.describe()
df.reset_index(inplace=True)

# Pivot Table

df_pivot_F = pd.DataFrame(df.pivot_table(index = "customer_id", values = ["product_id"], aggfunc = "count"))
df_pivot_R = pd.DataFrame(df.pivot_table(index = "customer_id", values = ["Recency"], aggfunc = "min", dropna = False))
df_pivot_M = pd.DataFrame(df.pivot_table(index = "customer_id", values = ["Profit"], aggfunc = "sum", dropna = False))
df_pivot = df_pivot_R.merge(df_pivot_F, on = "customer_id").merge(df_pivot_M, on = "customer_id")
df_pivot.dropna(inplace = True)
df_pivot.describe()
df_pivot["R_score"] = 0
df_pivot["F_score"] = 0
df_pivot["M_score"] = 0
df_pivot["Recency"] = df_pivot["Recency"].dt.days.astype('int64')
df_pivot.reset_index(inplace=True)
for i in range(0,len(df_pivot),1):
    if df_pivot["Recency"][i] >= 86:
       df_pivot["R_score"][i] = 1
    elif df_pivot["Recency"][i] >= 45:
       df_pivot["R_score"][i] = 2
    elif df_pivot["Recency"][i] >= 18:
       df_pivot["R_score"][i] = 3
    else:
       df_pivot["R_score"][i] = 4

for i in range(0,len(df_pivot),1):
    if df_pivot["product_id"][i] >= 7:
       df_pivot["F_score"][i] = 4
    elif df_pivot["product_id"][i] >= 5:
       df_pivot["F_score"][i] = 3
    elif df_pivot["product_id"][i] >= 3:
       df_pivot["F_score"][i] = 2
    else:
       df_pivot["F_score"][i] = 1

for i in range(0,len(df_pivot),1):
    if df_pivot["Profit"][i] >= 3947.950000:
       df_pivot["M_score"][i] = 4
    elif df_pivot["Profit"][i] >= 2583.670000:
       df_pivot["M_score"][i] = 3
    elif df_pivot["Profit"][i] >= 1278.360000:
       df_pivot["M_score"][i] = 2
    else:
       df_pivot["M_score"][i] = 1
    
df_pivot["RFM_value"] = 100*df_pivot["R_score"] + 10*df_pivot["F_score"] + df_pivot["M_score"]
df_pivot["RFM_value"].describe()
df_pivot["Customr_Category"] = ""
df_pivot.reset_index(inplace=True)
for i in range(0,len(df_pivot),1):
    if df_pivot["RFM_value"][i] >= 344:
       df_pivot["Customr_Category"][i] = "Platinum"
    elif df_pivot["RFM_value"][i] >= 244:
       df_pivot["Customr_Category"][i] = "Gold"
    elif df_pivot["RFM_value"][i] >= 144:
       df_pivot["Customr_Category"][i] = "Silver"
    else:
       df_pivot["Customr_Category"][i] = "Bronze"
df_Customer_RFM = pd.DataFrame(df_pivot.pivot_table(index = "Customr_Category", values = ["RFM_value"], aggfunc = "count"))
df_Customer_RFM.plot.bar()
df_Customer_Age = pd.DataFrame(df.pivot_table(index = 'Age_Category', columns = ['wealth_segment'], values = ['Profit'], aggfunc = "sum"))  
df_Customer_Age.plot.bar()
df_Customer_Cars = pd.DataFrame(df_CA_CD.pivot_table(index = 'state', columns= 'owns_car', values = 'Age_Category', aggfunc = "count"))    
df_Customer_Cars.plot.bar()
df_Customer_Bikes = pd.DataFrame(df_CA_CD.pivot_table(index = 'gender', values = 'past_3_years_bike_related_purchases' ,aggfunc = "sum"))    
df_Customer_Bikes.plot.bar()
df_Customer_IndustrySector = pd.DataFrame(df.pivot_table(index = 'job_industry_category', values = ['Profit','past_3_years_bike_related_purchases'],aggfunc = "sum"))
df_Customer_IndustrySector.plot.bar()
df_Customer_Gender = pd.DataFrame(df.pivot_table(index = 'gender', values = ['Profit'] ,aggfunc = "sum"))    
df_Customer_Gender.plot.bar()
df_CD.reset_index(inplace=True)
df_Cust = df_CA_CD.merge(df_pivot, on = "customer_id")
df_Cust.drop(['last_name', 'gender',
       'past_3_years_bike_related_purchases', 'DOB', 'job_title',
       'deceased_indicator', 'Age_Category', 'address', 'postcode',
       'tenure', 'country', 'property_valuation', 'index', 'Recency',
       'product_id', 'Profit', 'R_score', 'F_score', 'M_score',
       'Customr_Category'], axis = 1, inplace = True)
df_Cust.set_index("customer_id", inplace = True)
df_Cust.sort_values("RFM_value", ascending = False, inplace=True)
df_Cust.dropna(inplace=True)
df_Cust.reset_index(inplace = True)
df_Cust.to_csv(r"D:\Temp\KMPG_Customer.csv", index=False)
cid = df_Cust["customer_id"].tolist()
cid=cid[:1000]
cid=str(cid)





# TASK: 3 Data Visualisation

writer = pd.ExcelWriter(r"D:\Temp\KMPG_3.xlsx", engine = "xlsxwriter")
data.to_excel(writer, sheet_name = "Introduction")
df_trans.to_excel(writer, sheet_name = "Transaction")
df_CA.to_excel(writer, sheet_name = "CustomerAdress")
df_CD.to_excel(writer, sheet_name = "CustomerDemographic")
df_NCL.to_excel(writer, sheet_name = "NewCustomerList", index = False)
df_CA_CD.to_excel(writer, sheet_name = "Customer")
df.to_excel(writer, sheet_name = "Data", index = False)
writer.save()



