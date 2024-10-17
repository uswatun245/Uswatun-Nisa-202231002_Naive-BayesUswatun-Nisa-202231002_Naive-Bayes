import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('kaca.csv', sep = ';')
dataset

dataset.info()

categorical_features = [feature for feature in dataset.columns
                     if dataset[feature].dtypes == "O"]
dataset[categorical_features]

for i in categorical_features:
    print(dataset[i].unique())

    from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for feature in categorical_features:
    print(dataset[feature].unique())
    dataset[feature] = label_encoder.fit_transform(dataset[feature])
    print(dataset[feature].unique())

    dataset

    x = dataset.drop(['Kelas', 'No Data'], axis = 1)
y = dataset['Kelas']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

model = GaussianNB()

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

x_baru = np.array([[2,3,44500]])
y_pred_baru = model.predict(x_baru)
y_pred_baru

from sklearn.metrics import confusion_matrix, classification_report
print('evaluasi training')
print(confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))

print('evaluasi testing')
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
