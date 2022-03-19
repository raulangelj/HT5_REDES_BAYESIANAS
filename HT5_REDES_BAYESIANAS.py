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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import normaltest
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot
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
# %% [markdown]
# ## 2. Elabore un modelo de bayes ingenuo (naive bayes) utilizando el conjunto de entrenamiento y explique los resultados a los que llega. El experimento debe ser reproducible por lo que debe  fijar  que  los  conjuntos  de  entrenamiento  y  prueba  sean  los  mismos  siempre  que  se ejecute el código.
# %% [markdown]
# ## 3. El modelo debe ser de clasificación, use la variable categórica que hizo con el precio de las casas (barata, media y cara) como variable respuesta.
# %% [markdown]
# ## 4. Utilice  el  modelo  con  el  conjunto  de  prueba  y  determine  la  eficiencia  del  algoritmo  para clasificar.
# %% [markdown]
# ## 5. Haga  un  análisis  de  la  eficiencia  del  algoritmo  usando  una  matriz  de  confusión.  Tenga  en cuenta la efectividad, donde el algoritmo se equivocó más, donde se equivocó menos y la importancia que tienen los errores.
# %% [markdown]
# ## 6. Analice el modelo. Explique si hay sobreajuste (overfitting) o no.
# %% [markdown]
# ## 7. Haga  un  modelo  usando  validación  cruzada,  compare  los  resultados  de  este  con  los  del modelo anterior. ¿Cuál funcionó mejor?
# %% [markdown]
# ## 8. Compare la eficiencia del algoritmo con el resultado obtenido con el árbol de decisión (el de clasificación). ¿Cuál es mejor para predecir? ¿Cuál se demoró más en procesar?
