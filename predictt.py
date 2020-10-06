import numpy as np
import pandas as pd
import json
import pickle
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from ast import literal_eval

print("loading observation data...")

columns = ['id', 'morphology', 'passband', 'params', 'data', 'origin', 'period', 'target', 'epoch', 'meta']
df = pd.read_csv("C:/Users/student/Vierka/spotty/observed_lc.csv", sep="|", header=None)
df.columns = columns
df = df[["target", "morphology", "data", "meta"]]

processed_data = [literal_eval(_data)["flux"] for _data in df["data"]]
morphology = df[["morphology"]].values
objectt =  df[["target"]].values

df["original_morphology"] = [json.loads(_data)["morphology"] for _data in df["meta"]]
morphology = df["original_morphology"]

processed_data = np.array(processed_data, dtype=np.float32)

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


# delete semi-detached
'''
filtered_curves = []
curves=[]
for i in range(len(morphology)):
        if morphology[i] != 'semi-detached':
            filtered_curves.append(processed_data[i])
curves = filtered_curves
curves = np.array(curves, dtype=np.float32)
'''

print("loading model....")
classifier=load_model('model.hdf5')

print("predict model....")
y_pred = classifier.predict(processed_data)
print(y_pred)
y_pred2 = np.where(y_pred > 0.5, 1, 0)
print(y_pred2)
print(target)
cm = confusion_matrix(target.argmax(axis=1), y_pred2.argmax(axis=1))
print("Confusion matrix: \n" + str(cm))
prfs = precision_recall_fscore_support(target.argmax(axis=1), y_pred2.argmax(axis=1), average=None)
print("Precision, Recall, f1 score, support: " + str(prfs))

print(classification_report(target.argmax(axis=1), y_pred2.argmax(axis=1)))


# plot
for j in range(len(y_pred2)):
    if  (y_pred2[j].argmax(axis=0) == 1) and (target[j].argmax(axis=0) == 0)  :
        print(j)
        plt.plot(processed_data[j], label = objectt[j] )
plt.legend(loc='upper right')       

plt.savefig('image.png')

'''
objekt=[]
for j in range(len(y_pred2)):
    if (y_pred2[j].argmax(axis=0) == 0) and (target[j].argmax(axis=0) == 1)   :
        plt.plot(processed_data[j], label = objectt[j])
        
plt.legend(loc='upper right')       
plt.savefig('image2.png')
'''
