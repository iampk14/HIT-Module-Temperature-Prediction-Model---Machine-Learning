# Importing all required libraries
from tkinter import HIDDEN
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# Ignoring the warnings

#region region_sample
import warnings
warnings.filterwarnings("ignore")
#endregionuu


# Path selection
df = pd.read_csv(r"C:\Users\adity\Documents\Final Year Project\filteredData_train.csv")
df_test = pd.read_csv(r"C:\Users\adity\Documents\Final Year Project\filteredData_test.csv")

# module temperature
x_mod = df[['Gt', 'WS', 'Tamb']]
y_mod = df[['Temp_Mid_avg']]




# module temperature
x_mod_test = df_test[['Gt', 'WS', 'Tamb']]
y_mod_test= df_test[['Temp_Mid_avg']]

# Training Data
X_train, X_test, y_train, y_test = train_test_split(x_mod, y_mod, test_size=0.2)
regressor = RandomForestRegressor(n_estimators=100)

# Fitting data into model 
regressor.fit(X_train, y_train.values.ravel())

# Predicting Data 
y_pred_test = regressor.predict(x_mod_test)
daf=pd.DataFrame({'Actual':y_mod_test.values.flatten(), 'Predicted':y_pred_test, 'Error':abs(y_pred_test-y_mod_test.values.flatten())})
print(daf)

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print("-----------------------------")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_pred_test, y_mod_test)))

# plotting Actual Value and Predicted value in Graph
plt.figure(figsize=(5, 7))
ax = sns.distplot(y_mod_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred_test, hist=False, color="b", label="Predicted Values" , ax=ax)
plt.title('Actual vs Predicted values')
plt.show()



# Code for Maximum Power

x_pmax = df[['Ipmax', 'Vpmax','Gt','Voc','Isc']]
y_pmax = df[['Pmax']]

x_pmax_test= df_test[['Ipmax', 'Vpmax','Gt','Voc','Isc']]
y_pmax_test = df_test[['Pmax']]

# Training Data 
X_train_pmax, X_test_pmax, y_train_pmax, y_test_pmax = train_test_split(x_pmax, y_pmax, test_size=0.2)
regressor_pmax = RandomForestRegressor(n_estimators=100)

# Fitting data into model
regressor_pmax.fit(X_train_pmax, y_train_pmax)

# Predicting Pmax
y_pred_pmax_test = regressor_pmax.predict(x_pmax_test)
daf_pmax=pd.DataFrame({'Actual':y_pmax_test.values.flatten(), 'Predicted':y_pred_pmax_test,'Error':abs(y_pred_pmax_test-y_pmax_test.values.flatten())})
print("-----------------------------")
print(daf_pmax)

print("-----------------------------")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_pmax_test, y_pred_pmax_test)))

# Plotting Actual and predicted value of Pmax
plt.figure(figsize=(5, 7))
ax = sns.distplot(y_pmax_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred_pmax_test, hist=False, color="y", label="Predicted Values" , ax=ax)
plt.title('Actual vs Predicted values')
plt.show()

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 3,figsize=(12,12))

# For Ipmax vs Pmax
axis[0, 0].scatter(y_pmax, df[['Ipmax']], color= 'red')
axis[0, 0].set_title("Ipmax vs Pmax")
  
# For vpmax vs Pmax
axis[0, 1].scatter(y_pmax, df[['Vpmax']], color= 'green')
axis[0, 1].set_title("Vpmax vs Pmax")

# For Ws vs Tm
axis[1, 0].scatter(y_mod, df[['WS']], color= 'blue')
axis[1, 0].set_title("WS vs Tm")
  
# For Gt vs Tm
axis[1, 1].scatter(y_mod, df[['Gt']], color= 'green')
axis[1, 1].set_title("Gt vs Tm")

# For Tamb vs Tm
axis[1, 2].scatter(y_mod, df[['Tamb']], color= 'red')
axis[1, 2].set_title("Tamb vs Tm")
  
# Combine all the operations and display
plt.show()
