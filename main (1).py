import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score , mean_squared_error
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


df=pd.read_csv("Gold Price (2013-2023).csv")


# Clean numerical columns by removing commas and converting to float
columns = ["Price", "Open", "High", "Low"]
for col in columns:
    df[col] = df[col].replace({",": ""}, regex=True)
    df[col] = df[col].astype("float64")

# Convert Date format 
df['Date']=pd.to_datetime(df['Date'])
df.sort_values(by="Date",ascending=True,inplace=True)
df.reset_index(drop=True,inplace=True)
df["date_year"]= df["Date"].dt.year
df["date_month"]= df["Date"].dt.month
df["date_day"]= df["Date"].dt.day


# Replace 'K' with '000' and convert 'Vol.' to float
df['Vol.'] = df['Vol.'].str.replace("K", "000", regex=True)
df['Vol.'] = df['Vol.'].astype("float64")

# Remove '%' sign from 'Change %' and convert to float
df['Change %'] = df['Change %'].str.replace("%", "", regex=True)
df['Change %'] = df['Change %'].astype("float64")

# Assuming 'df' is your DataFrame and it's already loaded and preprocessed

cols_to_shift = ["Open", "High", "Low", 'Vol.', "Change %"]
df_shifted = df[['Date']].copy()

for col in cols_to_shift:
    df[f'Yesterday_{col}'] = df[col].shift(1)

df = df.merge(df_shifted, on='Date', how='left')

# Remove rows where the previous day's date does not match
df['Yesterday_Date'] = df['Date'].shift(1)
df = df[df['Date'] - pd.Timedelta(days=1) == df['Yesterday_Date']]

# Drop the temporary 'Yesterday_Date' column
df = df.drop(columns=['Yesterday_Date'])

# Drop the first row with NaN values due to shifting
df = df.dropna().reset_index(drop=True)

# Display the first few rows to verify the new columns

print(df.head())

   
# Clean numerical columns by removing commas and converting to float
columns = ["Yesterday_Open", "Yesterday_High", "Yesterday_Low", "Yesterday_Vol.", "Yesterday_Change %"]

for i in columns :
    df[i]=df[i].replace({",":""},regex=True)
    df[i]=df[i].astype("float64")
print(df.info())

print(df.isna().sum())    



# removing outliers 
yesterday_columns = ["Yesterday_Open", "Yesterday_High", "Yesterday_Low", "Yesterday_Vol.", "Yesterday_Change %"]

for i in columns :
    Q1= df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR= Q3-Q1
    lower_bound= Q1 - 1.5 * IQR
    upper_bound= Q3 + 1.5 * IQR 
    print(len(df[(df[i] > upper_bound) | (df[i] < lower_bound)]))
    df = df[(df[i]>= lower_bound) & (df[i]<= upper_bound)]
    
# Spliting x and y 
    
col = df.columns.drop(['Price', 'Date',"Open", "High", "Low", 'Vol.', "Change %"])
x=df[col]
y=df["Price"]
print(x.head())
print(y.head())

# Training the model Stage 
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.30)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# modeling 

linreg= LinearRegression()

linreg.fit(x_train,y_train)

y_pridict= linreg.predict(x_test)



# Plot predictions vs actual values
plt.scatter(y_test,y_pridict)
plt.plot(y_test, y_test, color='red')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Calculate R² score
r2 = r2_score(y_test,y_pridict)
print(f'R^2 Score: {r2}')

print("Mean Squared Error: ", mean_squared_error(y_test, y_pridict))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pridict)))




# RandomForest using GridSearch 

rand= RandomForestRegressor()

param_grid={
    "n_estimators": [100,200,300],
    "max_depth": [None,10,20,30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    
}

GS=GridSearchCV(estimator=rand, param_grid=param_grid, scoring='r2',verbose=0,n_jobs=-1)
GS.fit(x_train,y_train)

# printing the best parameter and score for the model 
print("Best parameters:", GS.best_params_)
best_rf = GS.best_estimator_

print("Best Score:", GS.best_score_)

