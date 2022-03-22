# %% [markdown]
# # HOJA DE TRABAJO 5 REDES BAYESIANAS

# Raul Jimenez 19017

# Donaldo Garcia 19683

# Oscar Saravia 19322

# link al repo: https://github.com/raulangelj/HT5_REDES_BAYESIANAS

# %%
# from re import U
from statsmodels.graphics.gofplots import qqplot
import numpy as np
import pandas as pd
# import pandasql as ps
import matplotlib.pyplot as plt
# import scipy.stats as stats
import statsmodels.stats.diagnostic as diag
# import statsmodels.api as sm
import seaborn as sns
# import random
import sklearn.cluster as cluster
# import sklearn.metrics as metrics
import sklearn.preprocessing
# import scipy.cluster.hierarchy as sch
import pyclustertend
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import normaltest
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from yellowbrick.regressor import ResidualsPlot
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import sklearn.mixture as mixture
# from sklearn import datasets
# from sklearn.cluster import DBSCAN
# from numpy import unique
# from numpy import where
# from matplotlib import pyplot
# from sklearn.datasets import make_classification
# from sklearn.cluster import Birch
# from sklearn.mixture import GaussianMixture


# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# %% [markdown]
# ## 1. Use los mismos conjuntos de entrenamiento y prueba que utilizó en las dos hojas anteriores.

# %%
train = pd.read_csv('./train.csv', encoding='latin1')
train.head()

# %%
usefullAttr = ['SalePrice', 'LotArea', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF',
               '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'PoolArea', 'Neighborhood', 'OverallQual']


# %%
data = train[usefullAttr]
data.head()

# %%
sns.pairplot(data[['SalePrice', 'LotArea', 'TotalBsmtSF',
             'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']], hue='SalePrice')
plt.show()

# %%
plt.subplots(figsize=(8, 8))
sns.heatmap(data[['SalePrice', 'LotArea', 'TotalBsmtSF',
                  'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']].corr(), annot=True, fmt="f").set_title("Correlación de las variables numéricas de Iris")


# %%
# NORMALIZAMOS DATOS
if 'Neighborhood' in data.columns:
    usefullAttr.remove('Neighborhood')
data = train[usefullAttr]
X = []
for column in data.columns:
    try:
        column
        if column != 'Neighborhood' or column != 'SalePrice':
            data[column] = (data[column]-data[column].mean()) / \
                data[column].std()
            X.append(data[column])
    except:
        continue
data_clean = data.dropna(subset=usefullAttr, inplace=True)
X_Scale = np.array(data)
X_Scale

# %%
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(X_Scale)
kmeans_result = kmeans.predict(X_Scale)
kmeans_clusters = np.unique(kmeans_result)
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = np.where(kmeans_result == kmeans_cluster)
    # make the plot
    plt.scatter(X_Scale[index, 0], X_Scale[index, 1])
plt.show()

# %%
data['cluster'] = kmeans.labels_
print(data[data['cluster'] == 0].describe().transpose())
print(data[data['cluster'] == 1].describe().transpose())
print(data[data['cluster'] == 2].describe().transpose())
# ## Variable clasificacion
# %%
# Clasificacion de casas en: Economias, Intermedias o Caras.
data.fillna(0)
# limit1 = data.query('cluster == 0')['SalePrice'].mean()
# limit2 = data.query('cluster == 1')['SalePrice'].mean()

minPrice = data['SalePrice'].min()
maxPrice = data['SalePrice'].max()
division = (maxPrice - minPrice) / 3
data['Clasificacion'] = data['LotArea']
# data.loc[data['SalePrice'] < limit1, 'Clasificacion'] = 'Economica'
# data.loc[(data['SalePrice'] >= limit1) & (
#     data['SalePrice'] < limit2), 'Clasificacion'] = 'Intermedia'
# data.loc[data['SalePrice'] >= limit2, 'Clasificacion'] = 'Caras'

data['Clasificacion'][data['SalePrice'] < minPrice + division] = 'Economica'
data['Clasificacion'][data['SalePrice'] >= minPrice + division] = 'Intermedia'
data['Clasificacion'][data['SalePrice'] >= minPrice + division * 2] = 'Caras'

# %% [markdown]
# #### Contamos la cantidad de casas por clasificacion

# %%
# Obtener cuantos datos hay por cada clasificacion
print(data['Clasificacion'].value_counts())

# %% [markdown]
# ## Dividmos en entrenamiento y prueba

# %% [markdown]
# # Estableciendo los conjuntos de Entrenamiento y Prueba

# %%
y = data['Clasificacion']
X = data[['SalePrice', 'LotArea', 'TotalBsmtSF',
          'GrLivArea', 'TotRmsAbvGrd', 'OverallQual']]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, train_size=0.7)
y_train

# %% [markdown]
# 70% de entrenamiento y 30% prueba

# %%
X_train.info()

# %%
X_test.info()

# %% [markdown]
# ## 2. Elabore un modelo de bayes ingenuo (naive bayes) utilizando el conjunto de entrenamiento y explique los resultados a los que llega. El experimento debe ser reproducible por lo que debe  fijar  que  los  conjuntos  de  entrenamiento  y  prueba  sean  los  mismos  siempre  que  se ejecute el código.

# %% [markdown]
# ## Creando el modelo

# %%
gaussian = GaussianNB()
modelo = gaussian.fit(X_train, y_train)

# %% [markdown]
# ## 3. El modelo debe ser de clasificación, use la variable categórica que hizo con el precio de las casas (barata, media y cara) como variable respuesta.
# %%
y_pred = gaussian.predict(X_test)
pred = list(y_pred)
print('Economicas', pred.count('Economica'))
print('Intermedias', pred.count('Intermedia'))
print('Caras', pred.count('Caras'))
# %% [markdown]
# ## 4. Utilice  el  modelo  con  el  conjunto  de  prueba  y  determine  la  eficiencia  del  algoritmo  para clasificar.
# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')
print('Accuracy: ', accuracy)
# %% [markdown]
# ## 5. Haga  un  análisis  de  la  eficiencia  del  algoritmo  usando  una  matriz  de  confusión.  Tenga  en cuenta la efectividad, donde el algoritmo se equivocó más, donde se equivocó menos y la importancia que tienen los errores.

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix for Naive Bayes\n', cm)
graf = sns.heatmap(cm, annot=True, cmap='Blues')
graf.set_title('Matriz de confusion\n\n');
graf.set_xlabel('\nPredicted Values')
graf.set_ylabel('Actual Values ');
plt.show()
p2 = """
Los resultados a los que llegamos al elaborar y analizar
la matriz de confusion utilizando el conjunto de
entrenamiento, son que la precision de la clasificacion
cuenta con alta efectividad
"""
print(p2)
# %% [markdown]
# ## 6. Analice el modelo. Explique si hay sobreajuste (overfitting) o no.
print(accuracy)
p6 = """
Para ver si existe sobreajuste o no, es necesario ver
el puntaje de Gauss obtenido, el cual es de 0.96, este
valor es algo elevado, pero al no ser tan cercano a 1
como lo es 0.99, podemos decir que  no hay sobreajuste en
el modelo
"""
print(p6)
# %% [markdown]
# ## 7. Haga  un  modelo  usando  validación  cruzada,  compare  los  resultados  de  este  con  los  del modelo anterior. ¿Cuál funcionó mejor?
#Score modelo general
score = gaussian.score(X_train, y_train)

print("Score del modelo en general:", score)
#Usando KFolds
kf = KFold(n_splits=10)
scores = cross_val_score(gaussian, X_train, y_train, cv=kf, scoring="accuracy")
print("KFolds: Metricas de la validacion cruzada:", scores)
print("KFolds: Resultado de la validacion cruzada:", scores.mean())

#Usando StratifiedKFolds
skf = StratifiedKFold(n_splits=10)
scores = cross_val_score(gaussian, X_train, y_train, cv=skf, scoring="accuracy")
print("StratifiedKFolds: Metricas de la validacion cruzada:", scores)
print("StratifiedKFolds: Resultado de la validacion cruzada:", scores.mean())

print("Dado el resultado se puede observar que el que mejor funcionó fue el de KFolds luego le sigue StratifiedKFolds y por último el modelo general.")
# %% [markdown]
# ## 8. Compare la eficiencia del algoritmo con el resultado obtenido con el árbol de decisión (el de clasificación). ¿Cuál es mejor para predecir? ¿Cuál se demoró más en procesar?
#Modelo de árbol de decisión
arbol = DecisionTreeClassifier(max_depth=4, random_state=42) 
arbol = arbol.fit(X_train, y_train) 
score = arbol.score(X_train, y_train)

print("Score arbol de decision:", score)
print("Se puede concluir que dado el score 1 el mejor fue el árbol de desición.")