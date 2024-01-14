#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyodbc
from sqlalchemy import create_engine
import numpy as np
import hashlib
import time
from cryptography.fernet import Fernet
import pandas as pd
import seaborn as sns
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

def encrypt_password(password):
    # Use a hashing algorithm (e.g., SHA-256) to encrypt the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return hashed_password

# database_password = input("Please enter DB Password: ")
server = input("Please enter SQL Server address: ")
encrypted_password = input("Please enter DB Password: ")

# dyncrypt_password =dyncrypt_password(encrypted_password)
a = Fernet.generate_key()

key = Fernet(a)

encrypt = key.encrypt(encrypted_password.encode())

decrypt = key.decrypt(encrypt).decode()

    # Define variables for connection parameters

database = 'SSISRND'
uid = 'ExcelVBA'
password = decrypt

# Establish the connection using variables
cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={uid};PWD={password}')
cursor = cnxn.cursor()

# SQL query
sql = """SELECT * FROM [SSISRND].[dim].[tblSupplierInfo] WHERE [Unit ID] = 144"""
sql2 ="""SELECT *  FROM [SSISRND].[pro].[tblProcurementInfo] where [Unit Id] = 144"""
# Execute the query and read the results into a DataFrame
Sup_Info = pd.read_sql(sql, cnxn)
Pro_Info = pd.read_sql(sql2,cnxn)

while True:
    
    
    # SQL query
    sql = """SELECT * FROM [SSISRND].[dim].[tblSupplierInfo] WHERE [Unit ID] = 144"""
    sql2 ="""SELECT *  FROM [SSISRND].[pro].[tblProcurementInfo] where [Unit Id] = 144"""
    # Execute the query and read the results into a DataFrame
    Sup_Info = pd.read_sql(sql, cnxn)
    Pro_Info = pd.read_sql(sql2,cnxn)

    # Check if the user wants to exit
    if server == 'exit':
        print("Exiting the loop. Goodbye!")
        break  # Exit the loop if the user enters 'exit'

    try:
        ## skLearn Function for Remove Duplicate

        from sklearn.base import BaseEstimator, TransformerMixin

        # class RemoveDuplicate(BaseEstimator, TransformerMixin):
        #     def fit(self, X, y=None):
        #         return self

        #     def transform(self, X):
        #         return X.drop_duplicates(['Supplier ID'],inplace=True)

        class RemoveDuplicate(BaseEstimator, TransformerMixin):
            def __init__(self, column_name):
                self.column_name = column_name

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                # Drop duplicates based on the specified column
                if self.column_name in X.columns:
                    X.drop_duplicates(subset=[self.column_name], inplace=True)
                return X



        class NameDroper(BaseEstimator, TransformerMixin):
            def __init__(self, column_name):
                self.column_name = column_name

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                # Drop the specified column in place
                if self.column_name in X.columns:
                    X.drop(columns=[self.column_name], inplace=True)
                return X
                    # Short Function Name
        dropper = NameDroper(column_name='BIN')
        dup=RemoveDuplicate(column_name='Supplier ID')

        # Assuming 'X' is your DataFrame
        dropper.fit_transform(Sup_Info)
        dup.fit_transform(Sup_Info)

        #df_raw = Pro_Info[Pro_Info['Item Type Id']==1]
        df_raw = Pro_Info[(Pro_Info['Item Type Id'] == 1) & (Pro_Info['Purchase Organization Id'] == 11)]
        df_raw
        df_raw['Purchase Order Type'].unique()
        PO_Type = df_raw.groupby('Purchase Organization Id')['Purchase Organization'].unique().reset_index()
        PO_Type
        df = df_raw[['Supplier Id','Supplier Name', 'Purchase Order No','Purchase Order Id','Item Id', 'Item Name','Currency Code','PO Net Value',
         'Total Amount', 'Total Qty','UoM Name', 'Base Price', 'Order Qty','Final Price','Reference Qty','Receive Qty', 
          'Rest Qty','Purchase Order Date','Issue Date', 'Transaction Quantity','Transaction Value','Lead Days',
           ]]
        item_Sup_freq = df.groupby(['Supplier Id','Item Id']).agg({'Purchase Order Id': 'count'}).reset_index()
        item_Sup_freq = item_Sup_freq.rename(columns={'Purchase Order Id': 'item_Sup_freq'})
        item_Sup_freq.head()
        # total order count (Total_Sup_freq)
        Total_Sup_freq = df.groupby('Supplier Id').agg({'Purchase Order Id': 'count'}).reset_index()
        Total_Sup_freq = Total_Sup_freq.rename(columns={'Purchase Order Id': 'Total_Sup_freq'})
        Total_Sup_freq.head()
        data = df[['Supplier Id','Item Id','Base Price','Order Qty','Receive Qty','Purchase Order Date','Issue Date','Lead Days']]
        data = pd.merge(data,item_Sup_freq[['Supplier Id','Item Id','item_Sup_freq']],on=['Supplier Id','Item Id'],how='inner')
        data = pd.merge(data, Total_Sup_freq[['Supplier Id','Total_Sup_freq']], on='Supplier Id', how='inner')
        data['Cost_lac'] = round(((data['Base Price']*data['Receive Qty'])/100000),2)
        Data = pd.merge(data, Sup_Info[['Supplier ID', 'RegistrationDate']], left_on='Supplier Id', right_on='Supplier ID', how='inner')

        # Find the index of the newest and oldest Registration Dates
        newest_index = Data['RegistrationDate'].idxmax()
        oldest_index = Data['RegistrationDate'].idxmin()

        # Get the corresponding Supplier IDs
        newest_supplier_id = Data.loc[newest_index, 'Supplier ID']
        oldest_supplier_id = Data.loc[oldest_index, 'Supplier ID']

#         # Print the results
#         print("The newest Supplier ID:", newest_supplier_id)
#         print("Supplier R_Date  :", Data.loc[newest_index, 'RegistrationDate'].date())

#         print("\nThe oldest Supplier ID:", oldest_supplier_id)
#         print("Supplier R_Date  :", Data.loc[oldest_index, 'RegistrationDate'].date())

        Data['RegistrationDate'] = pd.to_datetime(Data['RegistrationDate'])
        # Calculate the age in months
        current_date = datetime.now()
        Data['AgeInMonths'] = ((current_date - Data['RegistrationDate']) / pd.Timedelta(days=30)).astype(int)

        to_drop = ["Purchase Order Date", "Supplier ID", "RegistrationDate"]

        for col in to_drop:
            try:
                Data.drop(col, axis=1, inplace=True)
            except KeyError:
                print(f"Column '{col}' not found. Skipping...")


        Data['Standard Lead Days']=12
        Data['LeadDeviance'] = Data['Standard Lead Days']-Data['Lead Days']
        Data['QTY_Deviance'] = Data['Order Qty']-Data['Receive Qty']
        df =round(Data.groupby(['Supplier Id','Item Id']).agg({
            'Base Price': 'mean',
            'Receive Qty': 'sum',
            'Issue Date': 'max',
            'Lead Days': 'mean',
            'item_Sup_freq':'mean',
            'Total_Sup_freq': 'mean',
            'Cost_lac': 'sum',
            'AgeInMonths': 'mean',
            'QTY_Deviance': 'mean'
        }).reset_index(),2)

        df['Issue Date'] = pd.to_datetime(df['Issue Date'], format='%Y-%m-%d')
        current_date = datetime.now()
        df['Recency'] = (current_date - df['Issue Date']).dt.days
        # df['Issue Date'].dtype
        data = df.copy()


        data['Time_to_Deliver'] = data['Lead Days']

        # Encode categorical variables
        label_encoder = LabelEncoder()
        data['Supplier Id'] = label_encoder.fit_transform(data['Supplier Id'])
        data['Item Id'] = label_encoder.fit_transform(data['Item Id'])

        # Define features and target variable
        features = ['Base Price','Receive Qty', 'Time_to_Deliver','item_Sup_freq','Total_Sup_freq','Cost_lac',
                    'AgeInMonths','QTY_Deviance','Recency']
        target = 'Supplier_Rank'  # You can create this column later

        # Create target variable based on your criteria (lower is better for all criteria)
        data['Supplier_Rank'] = (
            data.groupby('Item Id')
            .apply(
                lambda group: (
                    -group['Base Price'].rank(ascending=True) +
                    -group['Time_to_Deliver'].rank(ascending=True) +
                    group['Receive Qty'].rank(ascending=True) +
                    group['item_Sup_freq'].rank(ascending=True) +
                    group['Total_Sup_freq'].rank(ascending=True) +
                    group['Cost_lac'].rank(ascending=True) +
                    group['AgeInMonths'].rank(ascending=True) +
                    -group['QTY_Deviance'].rank(ascending=True) +
                    -group['Recency'].rank(ascending=True) 
                )
            )
            .values
        )

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

        #Random Forest Regression



        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict supplier rankings
        data['Predicted_Rank'] = model.predict(data[features])

        # Rank suppliers within each item based on predicted ranking
        data['Supplier_Rank_Predicted'] = data.groupby('Item Id')['Predicted_Rank'].rank(ascending=True)

        # Display the results
        r2_RFR = r2_score(data['Supplier_Rank'], data['Predicted_Rank'])
#         print(f'R-squared (RandomForestRegressor): {r2_RFR}')

        result_RFR = data[['Item Id', 'Supplier Id', 'Supplier_Rank', 'Supplier_Rank_Predicted']].sort_values(['Item Id', 'Supplier_Rank'])
        #print(result_df)




        # Linear Regression



        # Train the model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        # Predict supplier rankings
        data['Predicted_Rank_Linear'] = linear_model.predict(data[features])

        # Rank suppliers within each item based on predicted ranking
        data['Supplier_Rank_Predicted_Linear'] = data.groupby('Item Id')['Predicted_Rank_Linear'].rank(ascending=True)

        # Display the results
        result_df_linear = data[['Item Id', 'Supplier Id', 'Supplier_Rank', 'Supplier_Rank_Predicted']].sort_values(['Item Id', 'Supplier_Rank'])

        # Calculate R-squared
        r2_linear = r2_score(data['Supplier_Rank'], data['Predicted_Rank_Linear'])
#         print(f'R-squared (Linear Regression): {r2_linear}')

        #print(result_df_linear)



        # Decision Tree 

        # Train the model
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(X_train, y_train)

        # Predict supplier rankings
        data['Predicted_Rank_DT'] = dt_model.predict(data[features])

        # Rank suppliers within each item based on predicted ranking
        data['Supplier_Rank_Predicted_DT'] = data.groupby('Item Id')['Predicted_Rank_DT'].rank(ascending=True)

        # Display the results
        result_df_dt = data[['Item Id', 'Supplier Id', 'Supplier_Rank', 'Supplier_Rank_Predicted']].sort_values(['Item Id', 'Supplier_Rank'])

        # Calculate MSE
        r2_dt = r2_score(data['Supplier_Rank'], data['Predicted_Rank_DT'])
#         print(f'Mean Squared Error (Decision Tree): {r2_dt}')


        # Gradient Boosting


        # Train the model
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)

        # Predict supplier rankings
        data['Predicted_Rank_GB'] = gb_model.predict(data[features])

        # Rank suppliers within each item based on predicted ranking
        data['Supplier_Rank_Predicted_GB'] = data.groupby('Item Id')['Predicted_Rank_GB'].rank(ascending=True)

        # Display the results
        result_df_gb = data[['Item Id', 'Supplier Id', 'Supplier_Rank', 'Supplier_Rank_Predicted']].sort_values(['Item Id', 'Supplier_Rank'])

        # Calculate R-squared
        r2_gb = r2_score(data['Supplier_Rank'], data['Predicted_Rank_GB'])
#         print(f'R-squared (Gradient Boosting): {r2_gb}')

        #print(result_df_gb)


        # Create a DataFrame to store model results and R-squared scores
        model_results = pd.DataFrame({
            'Model': ['RandomForestRegressor', 'LinearRegression', 'DecisionTreeRegressor', 'GradientBoostingRegressor'],
            'R-squared': [r2_RFR, r2_linear, r2_dt, r2_gb]
        })

        # Display the R-squared scores for each model
        # print(model_results)

        # Find the best model based on the highest R-squared score
        best_model = model_results.loc[model_results['R-squared'].idxmax()]
        print(f"\nBest Model:\n{best_model}")

        # Assuming you have these DataFrames: result_df_dt, result_df_gb, result_df_linear, result_RFR

        # Calculate R-squared scores for each model
        r2_scores = {
            'DecisionTreeRegressor': r2_dt,
            'GradientBoostingRegressor': r2_gb,
            'LinearRegression': r2_linear,
            'RandomForestRegressor': r2_RFR
        }

        # Find the model with the highest R-squared score
        best_model = max(r2_scores, key=r2_scores.get)
        best_r2_score = r2_scores[best_model]

        # Display the best model and its associated R-squared score
        print(f"The best model based on R-squared score is: {best_model} with R-squared: {best_r2_score}")

        # Display the results of the best model (assuming you want to output the results of the best model)
        if best_model == 'DecisionTreeRegressor':
            best_result = result_df_dt
            best_Model_File = dt_model
        elif best_model == 'GradientBoostingRegressor':
            best_result = result_df_gb
            best_Model_File = gb_model
        elif best_model == 'LinearRegression':
            best_result = result_df_linear
            best_Model_File = linear_model
        elif best_model == 'RandomForestRegressor':
            best_result = result_RFR
            best_Model_File = model

        pickle.dump(best_Model_File, open('model.pkl', 'wb'))

        merged_df = pd.merge(df, best_result, left_index=True, right_index=True, how='inner')
        merged_df.sort_values('Supplier_Rank_Predicted')
        merged_df
        merged_df = pd.merge(df, best_result, left_index=True, right_index=True, how='inner')
        merged_df.sort_values('Supplier_Rank_Predicted')

        rank_Supp =merged_df.groupby(['Item Id_x','Supplier Id_x']).agg({
            'Supplier_Rank_Predicted':'mean',
            'Recency': 'mean'
        }).reset_index()

        top_two_ranked_suppliers = rank_Supp[rank_Supp['Supplier_Rank_Predicted'].isin([1,2,3])]
        top_two_ranked_suppliers.sort_values('Supplier_Rank_Predicted',ascending=True)

        top_two_ranked_suppliers.rename(columns={
            'Item Id_x': 'intItemID',
            'Supplier Id_x': 'intSupplierID',
            'Supplier_Rank_Predicted': 'Supplier_Rank',
            'Recency': 'Recency'
        }, inplace=True)
        
        
        

        
        
        import pyodbc
        from datetime import datetime

        # Replace placeholders with your database connection details

        database_name = 'SSISRND'
        table_name = 'pre.tblSupplierRank'
        username = 'ExcelVBA'

        # Establish a connection to the SQL Server database
        conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database_name};UID={username};PWD={password}'
        cnxn = pyodbc.connect(conn_str)
        cursor = cnxn.cursor()

        # Get current date/time
        current_datetime = datetime.now()
        intUnitID=144
        cursor.execute(f"TRUNCATE TABLE {table_name}")
        # Iterate through each row in the DataFrame and perform insert or update (replace) using MERGE statement
        # Iterate through each row in the DataFrame and perform insert or update (replace) using MERGE statement
        for index, row in top_two_ranked_suppliers.iterrows():
            # Check if data exists in the table based on intItemID and intSupplierID
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE intItemID = ? AND intSupplierID = ?", (row['intItemID'], row['intSupplierID']))
            count = cursor.fetchone()[0]

            if count == 0:
                # Data doesn't exist, perform INSERT
                
                cursor.execute(f"""
                    INSERT INTO {table_name} (intUnitID,intItemID, intSupplierID, Supplier_Rank, Recency, dteInsertDate)
                    VALUES (?,?, ?, ?, ?, ?)
                    """, (intUnitID,row['intItemID'], row['intSupplierID'], row['Supplier_Rank'], row['Recency'], current_datetime))
            else:
                # Data exists, perform UPDATE (replace)
                cursor.execute(f"""
                    UPDATE {table_name}
                    SET Supplier_Rank = ?, Recency = ?, dteInsertDate = ?, intUnitID= ?
                    WHERE intItemID = ? AND intSupplierID = ?
                    """, (row['Supplier_Rank'], row['Recency'], current_datetime, intUnitID, row['intItemID'], row['intSupplierID']))

            cnxn.commit()  # Commit after each insert/update

        # Close cursor and connection
#         cursor.close()
#         cnxn.close()

        print("Data insertion or replacement completed.")
    
        print("Sleeping for 2 hours...")
        time.sleep(2 * 60 * 60)  # 2 hours in seconds


    except ValueError:
        # Handle invalid input
        print("Invalid input. Please enter a valid number or 'exit'")
        



# In[ ]:





# In[ ]:




