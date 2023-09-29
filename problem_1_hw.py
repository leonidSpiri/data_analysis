import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix

filename = 'data/tripadvisor_review.csv'
namesArray = ['User ID', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6',
              'Category 7', 'Category 8', 'Category 9', 'Category 10']
data = read_csv(filename, names=namesArray)
data["User ID"] = data["User ID"].replace({"User ": ""}, regex=True)

print("2.1 head")
print(data.head(10))
print("\n\n2.2 tail")
print(data.tail(10))
# всего 980 записей


print("\n\n3 dimensions")
print(data.shape)
# 980 строк, 11 столбцов

print("\n\n4 data types")
print(data.dtypes)
# все столбцы имеют тип float64

print("\n\n5 statistical summary")
print(data.info())
# нет пропущенных значений. RangeIndex: 980 entries, 0 to 979

print("\n\n6 class distribution")
for i in range(0, 10):
    print(data.groupby(data.columns[i]).size())
# большое количество значений уникально.

print("\n\n7 set accuracy of output results")
set_option('display.precision', 3)

print("\n\n8 statistical summary")
print(data.describe())

print("\n\n9 skew")
print(data.skew())
# все столбцы имеют положительную асимметрию

print("\n\n10 kurtosis")
print(data.kurtosis())
# Колонка Category 4 имеет высокий положительный коэффициент эксцессов, что может указывать на наличие выбросов или особенности
# в данных. Данные колонки могут потребовать дополнительного внимания при анализе данных

print("\n\n11 pearson correlation")
n = data.shape[0]
set_option('display.max_columns', None)
set_option('display.max_rows', None)
print(data.corr(method="pearson"))
# Колонки Category 1 и Category 2 имеют высокий коэффициент корреляции.


print("\n\n12 scatter plot matrix")
plt.figure(figsize=(8, 8))
data.hist()
plt.show()
#Все диаграммы (кроме Category 5, 6, 8) являются ассиметричными слева.

print("\n\n13 density plot matrix")
data.plot(kind='kde', subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(8, 8))
plt.show()
#

print("\n\n14 box and whisker plot")
data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(8, 8))
plt.show()

plt.figure(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True)
plt.show()

plt.figure(figsize=(8, 8))
scatter_matrix(data, figsize=(8, 8))
plt.show()
