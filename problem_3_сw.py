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

filename = 'data/pima-indians-diabetes.data.csv'
namesArray = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=namesArray)

print(data.head(10))

array = data.values

print(data.shape)

x = array[:, 0:8]
y = array[:, 8]

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(x)
set_printoptions(precision=3)
print(rescaledX[0:5, :])

scaler = StandardScaler().fit(x)
rescaledX = scaler.fit_transform(x)
print(rescaledX[0:5, :])

scaler = Normalizer().fit(x)
rescaledX = scaler.fit_transform(x)
print(rescaledX[0:5, :])

binarizer = Binarizer(threshold=0.0).fit(x)
rescaledX = binarizer.fit_transform(x)
print(rescaledX[0:5, :])

test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(x, y)
print(fit.scores_)
features = fit.transform(x)
print(features[0:5, :])

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(x, y)
print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)
rfe.fit_transform(x, y)
