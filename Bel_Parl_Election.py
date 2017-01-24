# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:52:16 2017

@author: S_Vasilev
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

#загружаем данные
data = pd.read_csv('Bel_Elec_data.csv')

#добавляем столбец возраст - age
data['age'] = data['year'] - data['born']

# почистим колонку партии - party
data['party'] = data['party'].str.replace('None', 'беспартийный')
data['party'] = data['party'].str.replace('неизвестно', 'беспартийный')
data['party'] = data['party'].str.replace('Белорусская социальнр-спортивная партия',
                                      'Белорусская социально-спортивная партия')

#добавим столбец oppoz, где 0 - беспартийный, 1- оппазиционная партия, 
#2 - неоппазиционная партия
data['oppoz'] = data['party']
vals_to_replace = {'беспартийный':'0', 
                   'Белорусская аграрная партия':'2',
                   'Белорусская партия «Зеленые»':'1',
                   'Белорусская партия левых «Справедливый мир»':'1',
                   'Белорусская патриотическая партия':'2',
                   'Белорусская социал-демократическая партия (Грамада)':'1',
                   'Белорусская социал-демократическая партия (Народная грамада)':'1',
                   'Белорусская социально-спортивная партия':'2',
                   'Коммунистическая партия Беларуси':'2',
                   'Либерально-демократическая партия':'2',
                   'Объединенная гражданская партия':'1',
                   'Партия БНФ':'1',
                   'Партия коммунистов белорусская':'1',
                   'Республиканская партия':'2',
                   'Республиканская партия труда и справедливости':'2',
                   'Социал-демократическая партия народного согласия':'2'
                   }
data['oppoz'] = data['oppoz'].map(vals_to_replace) 

#разделим ФИО на три столбца
data['fam'], data['im'], data['otch'] = zip(*data['fio'].map(lambda x: x.split(' ')))

# удаляем неиспользуемые столбцы
del data['fio'], data['okrug'], data['info'], data['vydvinut'], data['comment']

# удаляем тех, кто снялся до выборов
data = data.loc[data.status != 2]

#выполним LabelEncoding
data_encoded = data.apply(preprocessing.LabelEncoder().fit_transform)

#нормализация данных
col_for_scaling = list(data.columns)
del col_for_scaling[8], col_for_scaling[7]
for i in col_for_scaling:
    data_encoded[i] = preprocessing.scale(data_encoded[i])

#разделим на данные до 2016 и данные 2016
data00_12 = data_encoded.loc[data_encoded.year < 5]
data16 = data_encoded.loc[data_encoded.year > 5]

del data00_12['year'], data16['year'], data16['status']

#определим список признаков
features = list(data00_12.columns)
del features[7]

#разделим dta00_12 на тестовую и тренировочную части
X_train, X_test, y_train, y_test = train_test_split(data00_12[features], 
                                                    data00_12.status, 
                                                    test_size=0.20, 
                                                    random_state=123)

svc = SVC(probability = True, random_state = 42)
svc = svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print(accuracy_score(y_test, y_pred))

svc_full = SVC(probability = True, random_state = 42)
svc_full = svc_full.fit(data00_12[features], data00_12['status'])

# предсказываем результат
res_svc = svc_full.predict_proba(data16)


result2016  = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 
             0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
             0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 
             0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 
             0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 
             0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
             1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 
             0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 
             0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
             0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 
             1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 
             0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
             0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
             0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
             0, 0, 0,]
             
data2016 = data.loc[data.year > 2015]
data2016 = data2016[['n_okrug', 'fam', 'im', 'otch']]
data2016['real'] = pd.Series(result2016, index=data2016.index)
data2016['prob'] = pd.Series(res_svc[:,1], index=data2016.index)


ok_res = pd.DataFrame(columns=['n_okrug', 'fam', 'im', 'otch', 'real', 'prob'])
for i in list(set(data2016['n_okrug'])):
     okr = data2016.loc[data2016['n_okrug'] == i]
     max_okr = max(okr['prob'])
     okr1 = okr.loc[okr['prob'] == max_okr]
     ok_res = ok_res.append(okr1)
     
sum_suc = (ok_res['real'] == 1).sum()
accur = sum_suc/110
