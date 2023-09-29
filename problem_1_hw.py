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

print("\n\n3 dimensions")
print(data.shape)

print("\n\n4 data types")
print(data.dtypes)

print("\n\n5 statistical summary")
print(data.info())

print("\n\n6 class distribution")
for i in range(1, 10):
    print(data.groupby(data.columns[i]).size())

print("\n\n7 set accuracy of output results")
set_option('display.precision', 3)

print("\n\n8 statistical summary")
print(data.describe())

print("\n\n9 skew")
print(data.skew())

print("\n\n10 kurtosis")
print(data.kurtosis())

print("\n\n11 pearson correlation")
n = data.shape[0]
set_option('display.max_columns', None)
set_option('display.max_rows', None)
print(data.corr(method="pearson"))

print("\n\n12 scatter plot matrix")
plt.figure(figsize=(8, 8))
data.hist()
plt.show()

print("\n\n13 density plot matrix")
data.plot(kind='kde', subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(8, 8))
plt.show()

print("\n\n14 box and whisker plot")
data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(8, 8))
plt.show()

plt.figure(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True)
plt.show()

plt.figure(figsize=(8, 8))
scatter_matrix(data, figsize=(8, 8))
plt.show()
