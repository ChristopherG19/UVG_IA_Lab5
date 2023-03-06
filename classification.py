import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
from pandas_profiling import ProfileReport
from quickda.clean_data import *

# Leer archivo csv
df = pd.read_csv('./dataset_phishing.csv')

#Se realiza la codificación de la columna status
data_encoded = pd.get_dummies(df, columns=['status'], prefix='', prefix_sep='')
df = data_encoded

df.info()

df["legitimate"] = df["legitimate"].astype('int64')
df["phishing"] = df["phishing"].astype('int64')
to_categoric = ["url"]
df = clean(df, method = 'dtypes', columns = to_categoric, dtype='category')

df.info()

df.describe()