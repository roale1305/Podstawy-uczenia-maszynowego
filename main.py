import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

frame = pd.read_csv("cleaned_data.csv", sep = ',', encoding= 'utf-8', index_col= 0)
frame_1 = frame.drop(['Credit_Score'], axis=1)
Credit_score = frame['Credit_Score']

#2. 1 wyświetl dane (początkowe)
print(frame.head())

#2. 2 podaj liczbę cech

print('Liczba cech {}'.format(frame.shape[1]))

#3. wybierzcie liczbę składowych – jedną z dwóch przedstawionych metod
#4. Przekształć zbiór: pca.fit_transform()

pca = PCA(svd_solver='full', n_components=0.95)
principal_components = pca.fit_transform(frame_1)
principal_df = pd.DataFrame(data=principal_components)
#4. 2 wyświetl dane (początkowe)
print(principal_df.head())
#4. 3 podaj liczbę cech
print('Liczba cech {}'.format(principal_df.shape[1]))

x_train, x_test, y_train, y_test = train_test_split(principal_df, Credit_score, test_size= 0.50, random_state= 50)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Regresja
reg = LogisticRegression(random_state=60).fit(x_train, y_train)
predict = reg.predict(x_train)

#Ocena
score = accuracy_score(y_test, predict)
print('Accuracy: {}'.format(score))

recall = recall_score(y_test, predict, average='weighted')
print('Recall score: {}'.format(recall))

precission = precision_score(y_test, predict, average='weighted')
print('Precision: {}'.format(precission))

f1_score = f1_score(y_test, predict, average='weighted')
print('F1: {}'.format(f1_score))


#Macierz
matrix = confusion_matrix(y_test, predict)

disp = ConfusionMatrixDisplay(matrix).plot(cmap='viridis')
disp.plot()
plt.show()
