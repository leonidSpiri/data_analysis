from numpy import set_printoptions
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import utils

filename = 'data/tripadvisor_review.csv'
namesArray = ['User ID', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6',
              'Category 7', 'Category 8', 'Category 9', 'Category 10']
data = read_csv(filename, names=namesArray)

data["User ID"] = data["User ID"].replace({"User ": ""}, regex=True)
data["User ID"] = data["User ID"].astype(int)

print(data.head(10))

array = data.values

print(data.shape)
print()
x = array[:, 0:8]
y = array[:, 8]

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(x)
set_printoptions(precision=3)
print(rescaledX[0:5, :])
print()

scaler = StandardScaler().fit(x)
rescaledX = scaler.fit_transform(x)
print(rescaledX[0:5, :])
print()

scaler = Normalizer().fit(x)
rescaledX = scaler.fit_transform(x)
print(rescaledX[0:5, :])
print()

binarizer = Binarizer(threshold=0.0).fit(x)
rescaledX = binarizer.fit_transform(x)
print(rescaledX[0:5, :])
print()

test = SelectKBest(score_func=f_classif, k=4)
lab_enc = preprocessing.LabelEncoder()
fit = test.fit_transform(x, y)

print(fit.scores_)
print()

features = fit.transform(x)
print(features[0:5, :])
print()

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(x, y)
print(fit.n_features_)
print()
print(fit.support_)
print()
print(fit.ranking_)
print("False")
rfe.fit_transform(x, y)
