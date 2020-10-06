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
from keras.layers import Input, Dense, concatenate, Activation, LSTM, Bidirectional, Dropout
from keras.models import Model
from keras.layers.merge import Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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
X_train, X, y_train, y = train_test_split(newData, target, test_size=0.2)
y_train = np_utils.to_categorical(y_train, 2)

print(len(X_train))

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

history = classifier.fit(X_train, y_train, validation_data=(newObservation, target), epochs=5, batch_size=32, verbose=1)
accuracy = classifier.evaluate(newObservation, target, batch_size=32, verbose=1)
print("Accuracy: " + str(accuracy))

classifier.save("model.hdf5")
