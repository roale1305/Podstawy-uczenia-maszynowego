import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#zaczytaj dane z pliku csv
df_train = pd.read_csv("train.csv" , sep = "," , encoding = 'utf-8')
#sprawdź liczbę kolumn i wierszy
df_train.shape
df_train.info()
#wyświetl część tabeli
df_train.head()
#usuń wiersze z duplikatami id
df_train.drop_duplicates(subset="ID", inplace=True)


#opisz statystyki danych
df_train.describe()
#zlicz różne wartości danych
for i in df_train.columns:
    print(df_train[i].value_counts())
    print('*'*50)

#zastąp błędne dane

df_train.info()
#zmień dane na numeryczne
FeaturesToConvert = ['Age', 'Annual_Income','Num_of_Loan', 'Num_of_Delayed_Payment','Changed_Credit_Limit',
                      'Outstanding_Debt','Amount_invested_monthly', 'Monthly_Balance']

# ale najpierw sprawdź czy nie ma błędów w danych
for feature in FeaturesToConvert:
    uniques = df_train[feature].unique()
# usuń zbędne znaki '-’ , '_'
for feature in FeaturesToConvert:
    df_train[feature] = df_train[feature].str.strip('-_')
# puste kolumny zastąp NAN
for feature in FeaturesToConvert:
    df_train[feature] = df_train[feature].replace({'':np.nan}) # zmien typ zmiennych ilościowych for feature in FeaturesToConvert:
    df_train[feature] = df_train[feature].astype('float64')

#uzupełnij braki średnią
df_train['Monthly_Inhand_Salary']= df_train['Monthly_Inhand_Salary'].fillna(method='pad')

# stwórz obiekt enkodera
label = LabelEncoder()
df_train.Occupation = label.fit_transform(df_train.Occupation)
# sprawdź transformacje
df_train.head()

# Tworzenie zmiennych kategorycznych
cat = ['Month', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']
df_cat = df_train[cat]

df_cat =df_cat.apply(LabelEncoder().fit_transform)

df_cat.head()

# Wykres
plt.figure(figsize = (20, 18))

sns.heatmap(df_train[FeaturesToConvert].corr(), annot = True, linewidths = 0.1, cmap = 'Blues')

plt.title('Numerical Features Correlation')

plt.show()

# Standaryzacja/normalizacja
scaler = MinMaxScaler()

for x in df_train[FeaturesToConvert]:
    df_train[x]=scaler.fit_transform(df_train[[x]])

df_train.head()