from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, LSTM, Bidirectional

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

'''
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


objectt = pd.read_pickle('C:/Users/student/Vierka/spotty/observation_data.pkl')['object']

print("loading model....")
from keras.models import load_model
classifier=load_model('model_BIlstm.hdf5')


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

print(classification_report(target, y_pred2))
print(range(len(y_pred2)))

from matplotlib import pyplot as plt

for j in range(len(y_pred2)):
    if  (y_pred2[j].argmax(axis=0) == 1) and (target[j].argmax(axis=0) == 0)  :
        print(j)
        plt.plot(newObservation[j], label = objectt[j] )
plt.legend(loc='upper right')       

plt.savefig('2.png')


objekt=[]
for j in range(len(y_pred2)):
    if (y_pred2[j].argmax(axis=0) == 0) and (target[j].argmax(axis=0) == 1)   :
        plt.plot(newObservation[j], label = objectt[j])
        
plt.legend(loc='upper right')       
plt.savefig('1.png')
'''

print("loading observation data...")
from ast import literal_eval

columns = ['id', 'morphology', 'passband', 'params', 'data', 'origin', 'period', 'target', 'epoch', 'meta']
df = pd.read_csv("C:/Users/student/Vierka/spotty/observed_lc.csv", sep="|", header=None)
df.columns = columns
df = df[["target", "morphology", "data", "meta"]]

processed_data = [literal_eval(_data)["flux"] for _data in df["data"]]
morphology = df[["morphology"]].values

df["original_morphology"] = [json.loads(_data)["morphology"] for _data in df["meta"]]
morphology = df["original_morphology"]


processed_data = np.array(processed_data, dtype=np.float32)



#processed = pd.read_pickle('C:/Users/student/Vierka/spotty/observation_data.pkl')['processed_lightcurve']
#morphology = pd.read_pickle('C:/Users/student/Vierka/spotty/observation_data.pkl')['morphology']
#print(processed.head())
'''
curves = []

for j in processed:
    processed_data = eval(j)
    processed_data = np.array(processed_data['flux'])
    print(processed_data)
    single_curve = []
    for i in range(len(processed_data)):
        point = []
        point_data = (processed_data[i] - processed_data.min()) / (processed_data.max() - processed_data.min())
        point.extend([point_data])
        single_curve.append(point)
    curves.append(single_curve)
    single_curve = pd.DataFrame(single_curve)
curves = np.array(curves)

'''


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
'''
# odstranenie semi-detached kriviek
filtered_curves = []
curves=[]
for i in range(len(morphology)):
        if morphology[i] != 'semi-detached':
            filtered_curves.append(processed_data[i])
curves = filtered_curves
curves = np.array(curves, dtype=np.float32)

###################################
'''


print("loading model....")
from keras.models import load_model
classifier=load_model('model.hdf5')


print("predict model....")
y_pred = classifier.predict(processed_data)
print(y_pred)
y_pred2 = np.where(y_pred > 0.5, 1, 0)
print(y_pred2)
print(target)
cm = confusion_matrix(target.argmax(axis=1), y_pred2.argmax(axis=1))
print("Kontingencna tablulka: \n" + str(cm))
prfs = precision_recall_fscore_support(target.argmax(axis=1), y_pred2.argmax(axis=1), average=None)
print("Presnost, navratnost, fmiera, support: " + str(prfs))


print(classification_report(target.argmax(axis=1), y_pred2.argmax(axis=1)))


objectt =  df[["target"]].values

from matplotlib import pyplot as plt

for j in range(len(y_pred2)):
    if  (y_pred2[j].argmax(axis=0) == 1) and (target[j].argmax(axis=0) == 0)  :
        print(j)
        plt.plot(processed_data[j], label = objectt[j] )
plt.legend(loc='upper right')       

plt.savefig('nove/bi_a.png')
'''

objekt=[]
for j in range(len(y_pred2)):
    if (y_pred2[j].argmax(axis=0) == 0) and (target[j].argmax(axis=0) == 1)   :
        plt.plot(processed_data[j], label = objectt[j])
        
plt.legend(loc='upper right')       
plt.savefig('nove/bi_b.png')
'''