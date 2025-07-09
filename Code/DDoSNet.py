## Import dependencies
import numpy as np
import pandas as pd
import pickle
import pandas
import re
import glob
import datetime
import tensorflow as tf
import itertools
import math
import random
from collections import Counter
from sklearn.metrics import log_loss, auc, roc_curve
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import *
from keras.engine.topology import Input
from keras.models import Model, Sequential
from keras.utils import np_utils, to_categorical
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.python.keras.optimizers import TFOptimizer, RMSprop
#from keras.optimizers import TFOptimizer, RMSprop

from sklearn import metrics

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import argparse
import os
from os import walk
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

from pylab import rcParams
import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from numpy.random import seed
seed(7)
#from tensorflow import set_random_seed
#set_random_seed(11)
tf.random.set_seed(11)
from sklearn.model_selection import train_test_split
from sklearn import decomposition

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from numpy import array

import matplotlib.pyplot as plt
import seaborn as sns

from pylab import rcParams

import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed
seed(7)
#from tensorflow import set_random_seed
#set_random_seed(11)
tf.random.set_seed(11)
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

from keras.layers import Input, Dropout, Dense
from keras.utils.data_utils import get_file

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras import regularizers
from numpy.random import seed
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
SEED = 123  #used to help randomly select the data points

#from tensorflow import set_random_seed
# set_random_seed(11)
tf.random.set_seed(11)
from sklearn.model_selection import train_test_split
# Define early_stopping_monitor
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)

## Import dependencies

import re
import glob
import datetime
import tensorflow as tf
import itertools
import math
import random
from collections import Counter
from sklearn.metrics import log_loss, auc, roc_curve
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import *
from keras.engine.topology import Input
from keras.models import Model, Sequential
from keras.utils import np_utils, to_categorical

## Set random seeds for reproducibility
np.random.seed(123)
random.seed(123)

from sklearn.metrics import accuracy_score

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, mean_squared_error,
                             mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import scikitplot as skplt
## Set a working directory and point to your data
## Set a working directory and point to your data
wd = "G:\\TUI\\MasterThesisMain\\Datasets\\CICDDOS2019\\Secondday\\CSV-01-12\\syntraffic\\"
cicids_files = "*.csv"

##Load the data

## Load the data
print("Reading CICIDS2017 data...")
files = glob.glob(wd + cicids_files)
cicids_data = []
for ff in files:
    cicids_data.append(pandas.read_csv(ff, encoding="Latin1"))
cicids_data = pandas.concat(cicids_data)

#cicids_data = cicids_data.drop(['SimillarHTTP'], axis = 1)
#cicids_data = cicids_data.dropna(axis=0, how='any')

dataframe = cicids_data.copy()

dataframe = dataframe.dropna(axis=0, how='any')

for col in dataframe.columns:
    print(col)

print('Label distribution Training set:')
print(dataframe[' Label'].value_counts())
print()

labeltrain = dataframe[' Label']

newlabel_train = labeltrain.replace({
    'DoS_Portmap': 'Attack',
    'DoS_UDP-lag': 'Attack',
    'DoS_UDP': 'Attack',
    'BENIGN': 'BENIGN',
    'DoS_LDAP': 'Attack',
    'DoS_SNMP': 'Attack',
    'DoS_UDP': 'Attack',
    'DoS_UDP': 'Attack',
    'DoS_UDP': 'Attack',
    'DoS_MSSQL': 'Attack',
    'DoS_DNS': 'Attack',
    'DoS_TFTP': 'Attack',
    'DoS_NTP': 'Attack',
    'DoS_NetBIOS': 'Attack',
    'DoS_SSDP': 'Attack',
    'DoS_Syn': 'Attack',
    'DDoS_Web': 'Attack'
})

dataframe['label'] = newlabel_train

print('Label distribution Training set:')
print(dataframe['label'].value_counts())
print()

COLUMN_TO_STANDARDIZE = [
    ' Bwd Packet Length Std', ' URG Flag Count', ' Active Std',
    ' ACK Flag Count', ' Min Packet Length', ' Fwd Avg Packets/Bulk',
    ' CWE Flag Count', ' Packet Length Mean', ' Bwd URG Flags',
    ' Bwd IAT Mean', ' Bwd IAT Max', ' Bwd IAT Min', ' Idle Std',
    ' act_data_pkt_fwd', ' SYN Flag Count', ' Fwd URG Flags', ' Idle Max',
    ' RST Flag Count', ' Max Packet Length', ' Down/Up Ratio',
    ' Total Length of Bwd Packets', ' Fwd IAT Mean', ' Flow IAT Min',
    ' Bwd PSH Flags', 'Bwd Avg Bulk Rate', 'Bwd Packet Length Max',
    ' PSH Flag Count', ' Fwd Packet Length Std', ' Bwd Packet Length Min',
    ' Total Backward Packets', ' Idle Min', ' Fwd Packet Length Min',
    'Active Mean', ' Init_Win_bytes_backward', ' Bwd Header Length',
    ' Subflow Fwd Bytes', ' Bwd IAT Std', ' Flow IAT Max',
    ' Subflow Bwd Bytes', 'Fwd IAT Total', 'Total Length of Fwd Packets',
    ' Bwd Avg Packets/Bulk', ' Avg Fwd Segment Size', ' Fwd IAT Std',
    ' Fwd IAT Max', ' Avg Bwd Segment Size', ' Bwd Packets/s',
    'Init_Win_bytes_forward', ' Protocol', 'FIN Flag Count',
    ' min_seg_size_forward', ' Fwd Avg Bulk Rate', ' ECE Flag Count',
    ' Flow IAT Mean', ' Active Min', ' Total Fwd Packets',
    ' Fwd Packet Length Max', 'Subflow Fwd Packets', 'Fwd PSH Flags',
    ' Packet Length Std', 'Bwd IAT Total', 'Fwd Avg Bytes/Bulk',
    ' Bwd Avg Bytes/Bulk', ' Fwd Header Length.1', 'Fwd Packets/s',
    ' Active Max', 'Idle Mean', ' Bwd Packet Length Mean', ' Flow IAT Std',
    ' Packet Length Variance', ' Average Packet Size', ' Subflow Bwd Packets',
    ' Fwd IAT Min', ' Fwd Header Length', ' Fwd Packet Length Mean'
]

Flow_Packets = dataframe[' Flow Packets/s']
#scaler training
###########'Flow Bytes/s', 'Flow Packets/s' contains infinity or a value too large for dtype('float64').
#scaler training
###########'Flow Bytes/s', 'Flow Packets/s' contains infinity or a value too large for dtype('float64').

dataframe[COLUMN_TO_STANDARDIZE] = preprocessing.StandardScaler(
).fit_transform(dataframe[COLUMN_TO_STANDARDIZE])

dataframe = dataframe.drop([
    'Flow ID', ' Source IP', ' Source Port', ' Destination IP',
    ' Destination Port', ' Timestamp'
],
                           axis=1)
'''
from sklearn.utils import shuffle
dataframe = shuffle(dataframe)
dataframe = shuffle(dataframe)
dataframe = shuffle(dataframe)
dataframe = shuffle(dataframe)
dataframe = shuffle(dataframe)


dataframe = dataframe.reset_index()
del dataframe['index']
'''

#input_X = dataframe.loc[:, dataframe.columns != 'Label'].values  # converts the df to a numpy array
#input_y = cicids_data['Label'].values

#training_df, testing_df = train_test_split(dataframe, test_size=0.2)

training_df = dataframe.copy()

# Repeat the previous steps again but change the directory folder to refer to testing files, then run the folloing command
testing_df = dataframe.copy()
'''
x_train,y_train=training_df,training_df.pop('label').values
x_train=x_train.values
x_train = np.array(x_train).astype(np.float32) 

y_train1 = pd.get_dummies(y_train)
y_train1=y_train1.values
y_train1 = np.array(y_train1).astype(np.float32)  


# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

'''
'''
training_df= training_df1.iloc[:-179000, :]
testing_df= dataframe.iloc[-179000:,:]
valid_df= training_df1.iloc[-179000:,:]
'''
'''
print('Label distribution Training set:')
print(training_df['Label'].value_counts())
print()
print(testing_df['Label'].value_counts())
print()

print(valid_df['Label'].value_counts())
print()
'''

training_df = training_df.drop(['SimillarHTTP'], axis=1)

testing_df = testing_df.drop(['SimillarHTTP'], axis=1)

x_train, y_train = training_df, training_df.pop('label').values
x_train = x_train.values
x_train = np.array(x_train).astype(np.float32)

y_train1 = pd.get_dummies(y_train)
y_train1 = y_train1.values
y_train1 = np.array(y_train1).astype(np.float32)

x_test, y_test = testing_df, testing_df.pop('label').values
x_test = x_test.values
x_test = np.array(x_test).astype(np.float32)

y_test1 = pd.get_dummies(y_test)
y_test1 = y_test1.values
y_test1 = np.array(y_test1).astype(np.float32)


def flatten(X):
    flattened_X = np.empty(
        (X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)


n_features = 77
timesteps = 1

x_train_3d = x_train.reshape(x_train.shape[0], timesteps, n_features)
y_train1_3d = y_train1.reshape(y_train1.shape[0], timesteps, y_train1.shape[1])

x_test_3d = x_test.reshape(x_test.shape[0], timesteps, n_features)
y_test1_3d = y_test1.reshape(y_test1.shape[0], timesteps, y_test1.shape[1])


def calculate_losses(x, preds):
    losses = np.zeros(len(x))
    for i in range(len(x)):
        losses[i] = ((preds[i] - x[i])**2).mean(axis=None)

    return losses


lr = 0.0001
adam = optimizers.Adam(lr)

####SimpleRNN
inputs = Input(shape=(timesteps, n_features))
encoded1 = SimpleRNN(64, activation='relu', return_sequences=True)(inputs)
encoded2 = SimpleRNN(32, activation='relu', return_sequences=True)(encoded1)
encoded3 = SimpleRNN(16, activation='relu', return_sequences=True)(encoded2)
encoded4 = SimpleRNN(8, activation='relu', return_sequences=True)(encoded3)
Latent = SimpleRNN(units=8, activation='relu',
                   return_sequences=False)(encoded4)

decoded1 = RepeatVector(timesteps)(Latent)
decoded2 = SimpleRNN(8, activation='relu', return_sequences=True)(decoded1)
decoded3 = SimpleRNN(16, activation='relu', return_sequences=True)(decoded2)
decoded4 = SimpleRNN(32, activation='relu', return_sequences=True)(decoded3)
decoded5 = SimpleRNN(64, activation='relu', return_sequences=True)(decoded4)
decoded6 = SimpleRNN(n_features, activation='relu',
                     return_sequences=True)(decoded5)

autoencoder = Model(inputs, decoded6)
encoder = Model(inputs, Latent)

autoencoder.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
autoencoder.summary()

history = autoencoder.fit(x_train_3d,
                          x_train_3d,
                          epochs=5,
                          batch_size=64,
                          validation_split=0.2,
                          verbose=2).history

#####################################################################In case you
output = Dense(2, activation='softmax')(Latent)

autoencoder = Model(inputs, output)

autoencoder.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

history = autoencoder.fit(x_train_3d,
                          y_train1,
                          epochs=10,
                          batch_size=64,
                          validation_split=0.2,
                          verbose=2).history

#####################################################################In case you train the Softemax with the encoded layers
preds = autoencoder.predict(x_test_3d)

y_test1 = flatten(y_test1_3d)

pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test1, axis=1)
print(metrics.classification_report(true_lbls, pred_lbls))

#####################################################################the output of the decoder(full input)
output = Dense(2, activation='softmax')(decoded6)

autoencoder = Model(inputs, output)

autoencoder.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

history = autoencoder.fit(x_train_3d,
                          y_train1,
                          epochs=5,
                          batch_size=64,
                          validation_split=0.2,
                          verbose=2).history

preds = autoencoder.predict(x_test_3d)

y_test1 = flatten(y_test1_3d)

pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test1, axis=1)
print(metrics.classification_report(true_lbls, pred_lbls))

#####################################################################Save the model
from keras.models import model_from_json
# serialize model to JSON
model_json = autoencoder.to_json()
with open("autoencoder.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("autoencoder.h5")
print("Saved model to disk")
#####################################################################
# later...

# load json and create model
json_file = open('autoencoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("autoencoder.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])
score = loaded_model.evaluate(x_test_3d, y_test1, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

preds = loaded_model.predict(x_test_3d)

pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test1, axis=1)
print(metrics.classification_report(true_lbls, pred_lbls))
#####################################################################Evaluation metrices

accuracy = accuracy_score(true_lbls, pred_lbls)
recall = recall_score(true_lbls, pred_lbls)
precision = precision_score(true_lbls, pred_lbls)
f1 = f1_score(true_lbls, pred_lbls)
print("Performance over the testing data set \n")
print("Accuracy : {} , Recall : {} , Precision : {} , F1 : {}\n".format(
    accuracy, recall, precision, f1))

###########confusion_matrix

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(true_lbls,
                                    pred_lbls,
                                    cmap=plt.cm.Blues,
                                    figsize=(10, 10),
                                    normalize=True,
                                    text_fontsize=20,
                                    title=' confusion matrix',
                                    title_fontsize=20)

classNames = ['Attack', 'Normal']
plt.title('Confusion Matrix', size=20)
plt.ylabel('True label', size=20)
plt.xlabel('Predicted label', size=20)
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
plt.autoscale(enable=True, axis='y')
#s = [['TN','FP'], ['FN', 'TP']]
#for i in range(2):
#    for j in range(2):
#        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), size = 12)
plt.savefig('confusion_matrix.png')
plt.savefig('confusion_matrix.pdf')

from pandas_ml import ConfusionMatrix

binary_confusion_matrix = ConfusionMatrix(true_lbls, pred_lbls)
binary_confusion_matrix.plot()
plt.show()

#################################################ROC Curve and AUC##########
###########

#ROC Curve and AUC
fig, ax = plt.subplots()
#size
fig.set_size_inches((6, 6))
#ROC Curve and AUC
false_pos_rate, true_pos_rate, thresholds = roc_curve(true_lbls, pred_lbls)
roc_auc = auc(
    false_pos_rate,
    true_pos_rate,
)
plt.plot(false_pos_rate,
         true_pos_rate,
         linewidth=5,
         label='AUC = %0.3f' % roc_auc)
#plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.title('Receiver operating characteristic curve (ROC)', size=15)
plt.ylabel('True Positive Rate', size=20)
plt.xlabel('False Positive Rate', size=20)
plt.legend(fontsize=20, loc=4)
plt.savefig('ROC_AUC.png')
plt.savefig('ROC_AUC.pdf')
plt.show()

##############################Classic##################################################score
trainlabel = np.array(y_train)
testlabel = np.array(y_test)
expected = testlabel
testdata = np.array(x_test)

model = LogisticRegression()
model.fit(x_train, trainlabel)
# make predictions
predicted = model.predict(x_test)
print(metrics.classification_report(expected, predicted))

cm = metrics.confusion_matrix(expected, predicted)

#  Gradient Boosting function for classification

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(x_train, trainlabel)

predicted = model.predict(x_test)
print(metrics.classification_report(expected, predicted))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, trainlabel)
predicted = model.predict(x_test)
print(metrics.classification_report(expected, predicted))

trainlabel = np.array(y_train)
testlabel = np.array(y_test)
expected = testlabel
testdata = np.array(x_test)

model = LogisticRegression()
model.fit(x_train, trainlabel)
# make predictions
predicted = model.predict(x_test)
print(metrics.classification_report(expected, predicted))

# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(x_train, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
print(metrics.classification_report(expected, predicted))
# summarize the fit of the model

model = DecisionTreeClassifier()
model.fit(x_train, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))

#%% SVM
from sklearn import svm

svc_model = svm.SVC(kernel='rbf', C=1.0, gamma=0.1)
svc_model.fit(x_train, trainlabel)
predicted = svc_model.predict(testdata)
print(metrics.classification_report(expected, predicted))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred_svm))
print("Accuracy Score : ", accuracy_score(y_test, y_pred_svm))
print("Classification Report: \n", classification_report(y_test, y_pred_svm))

#=============andrews_curves=============

import matplotlib
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates
from sklearn import preprocessing as ps
from pandas.plotting import radviz
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
sample = dataframe.sample(int(dataframe.shape[0] * .025))  # 10% of total data
sample.to_pickle("tsne_sample.pkl")
sample = pd.read_pickle("tsne_sample.pkl")

x_tsne = sample.iloc[:, :-1]
y_tsne = sample.iloc[:, -1]

from sklearn.decomposition import SparsePCA
pca_analysis = SparsePCA(n_components=20)
x_tsne_pca = pca_analysis.fit_transform(x_tsne)

#pd.DataFrame(x_tsne_pca).to_pickle("dataset/tsne_pca_df.pkl")
x_tsne_pca = pd.read_pickle("tsne_pca_df.pkl").values

x_tsne_pca_df = pd.DataFrame(x_tsne_pca)

codes_to_attack = {1: "Attack", 0: "BENIGN"}

y_tsne_cta = y_tsne.map(lambda x: codes_to_attack[x])
x_tsne_pca_df['is'] = y_tsne.values

import matplotlib.cm as cm
from collections import OrderedDict

cmaps = OrderedDict()

cmaps['Sequential (2)'] = [
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring',
    'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot',
    'gist_heat', 'copper'
]

cmaps['Diverging'] = [
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn',
    'Spectral', 'coolwarm', 'bwr', 'seismic'
]

cmaps['Miscellaneous'] = [
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot',
    'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    'nipy_spectral', 'gist_ncar'
]

plt.figure(figsize=(8, 8))
andrews_curves(x_tsne_pca_df, "is", colormap='prism')
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.title('Andrews curve', size=20)
plt.legend(fontsize=20, loc=1)
plt.savefig('andrews_curves1.png')
plt.savefig('andrews_curves1.pdf')
plt.show()
