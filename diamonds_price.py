
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

all_data = pd.read_csv('c:/Users/user/strt/inne/book/diamonds/diamonds.csv')
print(all_data.head())
all_data.hist(bins=50,figsize=(40,30))
#plt.show()
#train_data.info()
#print(data['talk_time'].describe())

#podział na dane uczace i testowe
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = len(data) * test_ratio
    test_indices = shuffled_indices[:int(test_set_size)]
    train_indices = shuffled_indices[int(test_set_size):]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

all_data_with_id = all_data.reset_index()
train_set, test_set = split_train_test_by_id(all_data_with_id, 0.2, "index")
#print(len(train_set))
#print(len(test_set))
#print(train_set)

explor = train_set.copy()
explor.plot(kind="scatter", x="x", y="y")

explor['size'] = explor['x']*explor['y']*explor['z']
#sprawdzenie koleracji danych
corr_matrix = explor.corr()
print(corr_matrix['price'].sort_values(ascending=False))

#inna metoda wykresy zależności
scatter_matrix(explor, figsize=(12,8))

explor.plot(kind='scatter', x='size', y='price')

explor_tr = train_set.drop(['cut','color','clarity'], axis=1)
explor_labels = train_set['price'].copy()

lin_reg = LinearRegression()
lin_reg.fit(explor_tr, explor_labels)

some_data = explor_tr.iloc[:5]
some_labels = explor_labels.iloc[:5]
print('prognozy: ',lin_reg.predict(some_data))
print('etykiety: ',list(some_labels))

#plt.show()