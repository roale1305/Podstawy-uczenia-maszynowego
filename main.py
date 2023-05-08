import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Salary Data.csv')
df.reset_index()
df.head()
df.shape

df.isna().sum()

df.dropna(axis=0, inplace=True)

df.isna().sum()

X = df['Years of Experience'].values.reshape(-1,1)
y = df['Salary'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

'''
print(f"\nModel score: {model.score(X_test, y_test)}" )
plt.scatter(X_train, y_train, c='g')
plt.plot(np.linspace(0,25,100).reshape(-1,1), model.predict(np.linspace(0,25,100).reshape(-1,1)), 'm')
plt.title('Training set')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


plt.scatter(X_test, y_test, c='g')
plt.plot(np.linspace(0,25,100).reshape(-1,1), model.predict(np.linspace(0,25,100).reshape(-1,1)), 'm')
plt.title('Test set')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
'''
#print(model.predict(np.array([2.34]).reshape(-1,1)))

#print(model.predict(np.array([0.5]).reshape(-1,1)))


#Zmiana Gender  female: 0 male: 1
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

#Lasso
#Wybór zmiennych objaśniających i zmiennej objaśnianej:
X = df[['Age', 'Gender', 'Years of Experience']]
y = df['Salary']



#Podział danych na zbiór treningowy i testowy:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Budowanie modelu LASSO
lasso = Lasso(alpha=0.1)

#Trenowanie modelu na danych treningowych:
lasso.fit(X_train, y_train)

#Wykonanie predykcji na danych testowych:
y_pred_test = lasso.predict(X_test)
y_pred_train = lasso.predict(X_train)

#Obliczenie wartości MSE i R²:
mse_lasso_test = mean_squared_error(y_test, y_pred_test)
mse_lasso_train = mean_squared_error(y_train, y_pred_train)
r2_lasso = r2_score(y_test, y_pred_test)
print('Lasso: ')
print('MSE test: ', mse_lasso_test)
print('MSE train: ', mse_lasso_train)
print('R²: ', r2_lasso , '\n')


#Regresja wielomianowa stopnia 3

#Tworzenie cech wielomianowych stopnia 3:
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

#Podział danych na zbiór treningowy i testowy:
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)

#Budowanie modelu regresji wielomianowej
    #Definiowanie modelu:

poly_reg = LinearRegression()

    #Trenowanie modelu na danych treningowych:

poly_reg.fit(X_train, y_train)

#Ocena modelu
    #Wykonanie predykcji na danych testowych:

y_pred_wielomianowa_test = poly_reg.predict(X_test)
y_pred_wielomianowa_train = poly_reg.predict(X_train)

    #Obliczenie wartości MSE i R²:
mse_wielomianowa_test = mean_squared_error(y_test, y_pred_wielomianowa_test)
mse_wielomianowa_train = mean_squared_error(y_train, y_pred_wielomianowa_train)
r2_wielomianowa = r2_score(y_test, y_pred_wielomianowa_test)
print('Regresja wielomianowa 3 stopnia: ')
print('MSE test: ', mse_wielomianowa_test)
print('MSE train: ', mse_wielomianowa_train)
print('R²: ', r2_wielomianowa, '\n')


#Budowanie modelu regresji drzewa decyzyjnego

    #Definiowanie modelu i ustawienie parametru max_depth:
dt_reg = DecisionTreeRegressor(max_depth=5)
    #Trenowanie modelu na danych treningowych:
dt_reg.fit(X_train, y_train)
    #Wykonanie predykcji na danych testowych:
y_pred_dt_test = dt_reg.predict(X_test)
y_pred_dt_train = dt_reg.predict(X_train)
    #Obliczenie wartości MSE i R²:
mse_dt_test = mean_squared_error(y_test, y_pred_dt_test)
mse_dt_train = mean_squared_error(y_train, y_pred_dt_train)
r2_dt = r2_score(y_test, y_pred_dt_test)
print('Regresja drzewa decyzyjnego: ')
print('MSE test: ', mse_dt_test)
print('MSE train: ', mse_dt_train)
print('R²: ', r2_dt)

frame = {'Typ regresji: ' : ['Lasso', 'Wielomianowa', 'Decision tree'],
         'MSE test: ' : [mse_lasso_train, mse_wielomianowa_test, mse_dt_test],
         'MSE train: ' : [mse_lasso_test, mse_wielomianowa_train, mse_dt_train],
         'R2' : [r2_lasso, r2_wielomianowa, r2_dt]}

dataFrame = pd.DataFrame(data = frame)

print('\n', dataFrame)