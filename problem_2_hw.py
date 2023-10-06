import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from scipy.stats._morestats import ShapiroResult
from statsmodels.stats.power import TTestIndPower  #
from statsmodels.stats.power import TTestPower
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.weightstats import ztest
from scipy import stats
import pingouin as pg

filename = 'data/tripadvisor_review.csv'
namesArray = ['User ID', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6',
              'Category 7', 'Category 8', 'Category 9', 'Category 10']
data = read_csv(filename, names=namesArray)

print("2. head")
# Содержит данные о 980 пользователей, которые оценили 10 категорий отеля
print(data.head(10))

print("3. hist")
param = 'Category 1'
salary = data[param]
plt.hist(salary, bins=25)
# plt.show()

log_salary = np.log10(data[param])
plt.hist(log_salary, bins=100)
# plt.show()
# После логарифмирования столбца Category 1 распределение не стало нормальным. Возмозжно это происходит из за того,
# что исходные данные могут иметь нелинейные или более сложные структуры.


print("4. shapiro")
print(stats.shapiro(log_salary))
# Результат показывает что p-значение (p-value) < 0.05, значит распределение не является нормальным

print("5. replace")
data["User ID"] = data["User ID"].replace({"User ": ""}, regex=True)
data["User ID"] = data["User ID"].astype(int)
print(data.head(10))
# убираем слово User из столбца User ID, чтобы можно было преобразовать в числовой тип

print("6. scatter")
plt.scatter(data['Category 1'], data['Category 2'])
# plt.show()
# По графику видно, что между столбцами Category 1 и Category 2 нет линейной зависимости


print("7. plot")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# График 1: Распределение первой категории
data['Category 1'].plot(kind='kde', ax=axes[0])
axes[0].set_title('Category 1')

# График 2: Распределение второй категории
data['Category 2'].plot(kind='kde', ax=axes[1])
axes[1].set_title('Category 2')

# plt.show()

column_bc = data['Category 1']
column_ss = data['Category 2']
bc_p = stats.shapiro(column_bc).pvalue
ss_p = stats.shapiro(column_ss).pvalue
print(bc_p, ss_p)
# Результат теста показывает, что p-value в обоих случаях меньше 0.05. Нулевую гипотезу о нормальном распределении подтвердить не удалось.
# Распределение сильно отличается от нормального

pg.partial_corr(data=data, x='Category 1', y='Category 2', covar='Category 3')
