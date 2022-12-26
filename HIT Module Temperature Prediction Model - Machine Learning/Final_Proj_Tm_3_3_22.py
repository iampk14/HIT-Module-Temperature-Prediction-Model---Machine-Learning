# Importing all required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Path selection
df = pd.read_csv(r"C:\Users\adity\Documents\Final Year Project\filteredData_train.csv")
# module temperature

x_mod = df[['Gt', 'WS', 'Tamb']]
y_mod = df[['Temp_Mid_avg']]

X_train, X_test, y_train, y_test = train_test_split(
    x_mod, y_mod, test_size=0.2)
regressor = RandomForestRegressor(n_estimators=100, random_state=1)
regressor.fit(X_train, y_train.values.ravel())

y_pred = regressor.predict(X_test)
daf=pd.DataFrame({'Actual':y_test.values.flatten(), 'Predicted':y_pred})
print(daf)

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(5, 7))


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Predicted Values" , ax=ax)


plt.title('Actual vs Predicted values')


plt.show()
plt.close()




# Code for Maximum Power

x_pmax = df[['Ipmax', 'Vpmax']]
y_pmax = df[['Pmax']]

X_train_pmax, X_test_pmax, y_train_pmax, y_test_pmax = train_test_split(
    x_pmax, y_pmax, test_size=0.2)
regressor_pmax = RandomForestRegressor(n_estimators=100, random_state=0)
regressor_pmax.fit(X_train_pmax, y_train_pmax)

y_pred_pmax = regressor_pmax.predict(X_test_pmax)
daf_pmax=pd.DataFrame({'Actual':y_test_pmax.values.flatten(), 'Predicted':y_pred_pmax})
print(daf_pmax)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_pmax, y_pred_pmax)))

plt.figure(figsize=(5, 7))


ax = sns.distplot(y_test_pmax, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred_pmax, hist=False, color="y", label="Predicted Values" , ax=ax)


plt.title('Actual vs Predicted values')


plt.show()
plt.close()