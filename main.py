#from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

#Import danych, ',' jako decimal
df = pd.read_csv('countries of the world.csv', decimal = ',')
print(df.head())

#Nadanie każdemu regionowi unikalnego numeru, usunięcie linii z pustymi wartosciami i kolumny country
df['Region'] = pd.factorize(df['Region'])[0] + 1
df = df.dropna()
df = df.drop('Country', axis=1)
print(df.head())

# Inicjalizacja modelu K-means z k=3
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=10)

# Dopasowanie modelu do danych
kmeans.fit(df)

# Przewidywanie przynależności do klastrów
labels = kmeans.labels_

print(labels)




#Sprawdzenie sugerowanych k metoda elbow
# Inicjalizacja listy do przechowywania sumy kwadratów odległości
sum_of_squared_distances = []

# Przeprowadzenie analizy dla różnych wartości k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    sum_of_squared_distances.append(kmeans.inertia_)

# Wyświetlanie wykresu "łokcia"
plt.plot(range(1, 11), sum_of_squared_distances, 'bx-')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('Suma kwadratów odległości')
plt.title('Metoda łokcia')
plt.show()


#Zbuduj dendrogram

# Obliczanie macierzy łączenia
Z = linkage(df, method='ward')

# Rysowanie dendrogramu
plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode='level', p=3)
plt.title('Dendrogram')
plt.xlabel('Indeksy punktów')
plt.ylabel('Odległość')
plt.show()
