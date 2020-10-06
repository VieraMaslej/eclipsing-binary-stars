from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, LSTM, Bidirectional, GRU

import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import json
import pickle
np.random.seed(1234)

from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D
from keras.layers import Input, Dense, concatenate, Activation,LSTM,Bidirectional
from keras.models import Model
from keras.layers.merge import Concatenate


data = pd.read_csv('C:/Users/student/Vierka/spotty/df_all.csv_v2')
data.head()

data=data[data['spotty']==0]

for row in data['generic_bessell_v']:
     row = json.loads(row)
    
newData = []
for row in data['generic_bessell_v']:
    row = json.loads(row)
    minValueOfRow, maxValueOfRow = min(row), max(row)
    newRow = []
    for valueIndex in range(len(row)):
        row[valueIndex] = (row[valueIndex] - minValueOfRow) / (maxValueOfRow - minValueOfRow)
        newRow.append([row[valueIndex]])
    newData.append(newRow)
    
newData = np.array(newData)

target = []
oc, sd, dt = 0, 0, 0
for row in data['morphology']:
    if row == 'over-contact':
        target.append(0)
        oc += 1
    elif row == 'detached':
        target.append(1)
        dt += 1
        
target = np.array(target)
print("Over-contact: " + str(oc) + "\n" + "\n" + "Detached: " + str(dt))


print("Pocet kriviek: " + str(len(newData)) + "\nPocet cieloveho atributu: " + str(len(target)))
X_train, X, y_train, y = train_test_split(newData, target, test_size=0.2)
y_train = np_utils.to_categorical(y_train, 2)

print(len(X_train))

# nacitanie observacnych dat a cieloveho atributu
# vyber atributu 'flux', ktory predstavuje svietivost svetelnych kriviek a ich transformacia na numpy polia
# preskalovanie hodnot svetelnej krivky podla MinMax skalovania
# transformacia oznacenia cieloveho atributu: over-contact - 0, semi-detached - 1, detached - 2
print("loading observation data...")
processed = pd.read_pickle('C:/Users/student/Vierka/spotty/observation_data.pkl')['processed_lightcurve']
morphology = pd.read_pickle('C:/Users/student/Vierka/spotty/observation_data.pkl')['morphology']
curves = []
for j in processed:
    processed_data = eval(j)
    processed_data = np.array(processed_data['flux'])
    single_curve = []
    for i in range(len(processed_data)):
        point = []
        point_data = (processed_data[i] - processed_data.min()) / (processed_data.max() - processed_data.min())
        point.extend([point_data])
        single_curve.append(point)
    curves.append(single_curve)
    single_curve = pd.DataFrame(single_curve)
curves = np.array(curves)

target = []
for i in morphology:
    if i == 'over-contact':
        target.extend([0])
    if i == 'semi-detached':
        target.extend([1])
    if i == 'detached':
        target.extend([1])
target = np.array(target)


target = np_utils.to_categorical(target, 2)

newObservation = []
for curve in curves:
    newCurve = []
    for valueIndex in range(len(curve)):
        if valueIndex % 2 == 1:
            newCurve.append(curve[valueIndex])
    newCurve = np.array(newCurve)
    newObservation.append(newCurve)
newObservation = np.array(newObservation)

print(len(newObservation))
print(len(newObservation[0]))

inputs = Input(shape=(100,1))
a = Bidirectional(LSTM(64, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(inputs)
a = Flatten()(a)
b = Conv1D(32, kernel_size = 3, padding = "valid", input_shape=(100,1))(inputs)
b = MaxPooling1D(2)(b)
b = Conv1D(32, kernel_size = 3, padding = "valid")(b)
b = MaxPooling1D(2)(b)
b = Flatten()(b)
x = concatenate([a,b])
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output = Dense(2, activation='softmax')(x)
classifier = Model(inputs=inputs, outputs=output)
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(classifier.summary())

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


history = classifier.fit(X_train, y_train, validation_data=(newObservation, target), epochs=5, batch_size=32, verbose=1)
accuracy = classifier.evaluate(newObservation, target, batch_size=32, verbose=1)
print("Celková presnosť modelu: " + str(accuracy))

classifier.save("model.hdf5")
# vyhodnotenie modelu na syntetickych datach pomocou kontingencnej tabulky
# vypocet presnosti, navratnosti a f1 skore

y_pred = classifier.predict(newObservation)
y_pred2 = []
for i in y_pred:
    maximum = np.argmax(i)
    y_pred2 = np.append(y_pred2, maximum)
y_pred2 = np_utils.to_categorical(y_pred2, 2)

cm = confusion_matrix(target.argmax(axis=1), y_pred2.argmax(axis=1))
print("Kontingencna tablulka: \n" + str(cm))
prfs = precision_recall_fscore_support(target.argmax(axis=1), y_pred2.argmax(axis=1), average=None)
print("Presnost, navratnost, fmiera, support: " + str(prfs))


print(classification_report(target, y_pred2))
'''


print("loading model....")
from keras.models import load_model
classifier=load_model('model.hdf5')


print("predict model....")
y_pred = classifier.predict(newObservation)
print(y_pred)
y_pred2 = np.where(y_pred > 0.5, 1, 0)
print(y_pred2)
print(target)
cm = confusion_matrix(target.argmax(axis=1), y_pred2.argmax(axis=1))
print("Kontingencna tablulka: \n" + str(cm))
prfs = precision_recall_fscore_support(target.argmax(axis=1), y_pred2.argmax(axis=1), average=None)
print("Presnost, navratnost, fmiera, support: " + str(prfs))


print(classification_report(target.argmax(axis=1), y_pred2.argmax(axis=1)))
'''