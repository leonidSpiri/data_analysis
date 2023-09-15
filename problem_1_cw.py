from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

filename = 'data/pima-indians-diabetes.data.csv'
namesArray = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=namesArray)

print(data.head(10))

print(data.tail(10))

print(data.shape)

print(data.info())

print(data.groupby('class').size())

set_option('display.precision', 2)

print(data.describe())
