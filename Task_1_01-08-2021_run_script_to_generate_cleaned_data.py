#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.style as style
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

# Input cell where all the required data must be entered.

# Reading the .csv input file.
master_data = pd.read_csv("E:/CV/Internship/job/task-1/manuf_data_input.csv")

file_name ="cleaned_data" # enter the file name with which the furnished data must be stored

path_to_save = "E:/CV/Internship/job/task-1/" #enter file path wo to speichern


# Creating the copy of the master file
manufacturing_one_hr = master_data.copy()

# Calculating the percentage of missing values
null_values = manufacturing_one_hr.isnull().sum()
percentage_missing_values_df = pd.DataFrame(round(100*(manufacturing_one_hr.isnull().sum()/len(manufacturing_one_hr.index)), 2))
print("Percentage_missing_values_in_each_column =", percentage_missing_values_df.sort_values(by = 0, ascending=False))

# Printing the datatypes of the features.
print("data_types_of_each_feature =", manufacturing_one_hr.dtypes)

# Splitting the coloumn "timestamp" into two columns as "furnished_timestamp" and "col_sep_from_timestamp".
manufacturing_one_hr["furnished_timestamp"] = manufacturing_one_hr["timestamp"].str.split(';', expand=True)[0]
manufacturing_one_hr["col_sep_from_timestamp"] = manufacturing_one_hr["timestamp"].str.split(';', expand=True)[1]
manufacturing_one_hr = manufacturing_one_hr.drop(['timestamp'], axis=1)

# Rearranging the oreder of the columns as it was in original dataset.
cols = manufacturing_one_hr.columns.tolist()
for rearrange in range(0,2):
    cols = cols[-1:] + cols[:-1]    
manufacturing_one_hr = manufacturing_one_hr[cols]

# Getting the list of index value where "col_sep_from_timestamp" isnot None and "part_id" is null.
list_index_true_condn = pd.DataFrame(manufacturing_one_hr[~manufacturing_one_hr["col_sep_from_timestamp"].isna()]["part_id"].isna()).index.tolist()
list_index_true_condn

# Below line of codes represent the procedure to shift the elements of the row if the  above condition is satisfied
for index_no in list_index_true_condn:
    #shifting all the elements of that particular row to right side by a step of 1.
    manufacturing_one_hr.loc[[index_no]] = manufacturing_one_hr.loc[[index_no]].shift(periods=1, axis="columns")
    # interchanging the values of the cols "furnished_timestamp" & "col_sep_from_timestamp" with each other.
    manufacturing_one_hr.loc[index_no,['furnished_timestamp','col_sep_from_timestamp']] = manufacturing_one_hr.loc[index_no,['col_sep_from_timestamp','furnished_timestamp']].values
    
manufacturing_one_hr

# Function developed to verify if the given two columns are similar anf if they are then we drop one of the duplicate column from the dataframe.
def compare_cols(manufacturing_one_hr, list_cols):
    compare_column = np.where(manufacturing_one_hr[list_cols[0]] == manufacturing_one_hr[list_cols[1]], True, False)
    if np.unique(compare_column) == True:
        df = manufacturing_one_hr.drop([list_cols[1]], axis=1)
    return df

# Comparing columns "station_id" vs "station_id.1".
df_remove_staion_id = compare_cols(manufacturing_one_hr, ["station_id", "station_id.1"])

# Comparing columns "prod_id" vs "prodid".
df_remove_prodid = compare_cols(df_remove_staion_id, ["prod_id", "prodid"])

# converting the data type of column "furnished_timestamp" to datetime type.
df_remove_prodid["furnished_timestamp"] = pd.to_datetime(df_remove_prodid["furnished_timestamp"])

# converting the data type of column "prod_id" to string type.
df_remove_prodid['prod_id']= df_remove_prodid['prod_id'].astype('str')

# converting the data type of column "station_id" to integer type.
df_remove_prodid['station_id']= df_remove_prodid['station_id'].astype(int)

# converting the data type of column "sensor" to string type.
df_remove_prodid['sensor']= df_remove_prodid['sensor'].astype("category")

# converting the data type of column "part_id" to string type.
df_remove_prodid['part_id']= df_remove_prodid['part_id'].astype('str')

# The column "value" contains different data types like for example:
# "OK" = str and "-0.7995458628749204" = float , thus in order to make format the same column with different data types we used the following lines of code.
index = df_remove_prodid.index

condition_mit_status = df_remove_prodid["sensor"] == "STATUS"
condition_ohne_status = df_remove_prodid["sensor"] != "STATUS"

condition_mit_status_indices = index[condition_mit_status].tolist()
condition_ohne_status_indices = index[condition_ohne_status].tolist()

for mit_status in condition_mit_status_indices:
    df_remove_prodid['value'][mit_status] = str(df_remove_prodid['value'][mit_status])
    
for ohne_status in condition_ohne_status_indices:
    df_remove_prodid['value'][ohne_status] = float(df_remove_prodid['value'][ohne_status])
    
# Dropping the dummy column "col_sep_from_timestamp". 
df_remove_prodid = df_remove_prodid.drop(['col_sep_from_timestamp'], axis=1)

# Furnished data file.
furnished_file = df_remove_prodid.copy()

# Code to save the furnished file.
path = (path_to_save +str(file_name)+"_"+str(datetime.now())[:19].replace(":","").replace(":","")+".csv")
furnished_file.to_csv(path)

