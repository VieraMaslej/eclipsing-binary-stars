import numpy as np
import pandas as pd
import json
import pickle
np.random.seed(1234)
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D
from keras.layers import Input, Dense, concatenate, Activation, LSTM, Bidirectional, Dropout
from keras.models import Model
from keras.layers.merge import Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# load synthetic light curve - ELISa
data = pd.read_csv('data.csv')
data.head()

data=data[data['spotty']==0]

for row in data['generic_bessell_v']:
     row = json.loads(row)

# normalize data
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

# target by morphology
target = []
oc, dt = 0, 0
for row in data['morphology']:
    if row == 'over-contact':
        target.append(0)
        oc += 1
    elif row == 'detached':
        target.append(1)
        dt += 1      
target = np.array(target)

# split to train and validation dataset
X_train, X, y_val, y_val = train_test_split(newData, target, test_size=0.2)
y_train = np_utils.to_categorical(y_train, 2)

#biLSTM + CNN model
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

# CNN, LSTM model -- the second best model

#inputs = Input(shape=(100,1))
#b = Conv1D(32, kernel_size = 3, padding = "valid")(inputs)
#b = MaxPooling1D(2)(b)
#b = Dropout(0.2)(b)
#b = LSTM(64, return_sequences=True)(b)
#b = Flatten()(b)
#x = Dense(32, activation='relu')(b)
#output = Dense(2, activation='softmax')(x)
#classifier = Model(inputs=inputs, outputs=output)
#classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#print(classifier.summary())

# checpoint - save only best model
saved_model = "model.hdf5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=3)
callbacks_list = [checkpoint, early]

# training  
history = classifier.fit(X_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=32, verbose=1, callbacks = callbacks_list)

# evalute model - validation dataset
accuracy = classifier.evaluate(x_val, y_val, batch_size=32, verbose=1)
print("Accuracy: " + str(accuracy))

