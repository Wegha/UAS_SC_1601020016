"""
Created on Wed Feb 10 09:57:20 2019
#UAS SOFT COMPUTING
#I WAYAN WEGHA NANDA KUSUMA
#PRODI IF 2016
#1601020016
@author: wegha
"""
# Import Package-Package
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

#PROSES IMPORT DAN PROSES DATA
#=======================================
#membaca dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#X adalah data pada semua dataset kecuali 3 kolom pertama
X = dataset.iloc[:, 3:-1].values
#Y adalah data pada semua dataset kecuali kolom terakhir
Y = dataset.iloc[:, 13].values


#memanggil 2 fungsi dari sklear yaitu OneHotEncoder dan LabelEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_negara_X = LabelEncoder()
labelencoder_jkelamin_X = LabelEncoder()

#merubah nama negara pada kolom index ke-1 menjadi bentuk angka (string->int)
#merubah jenis kelamin pada kolom index 2 menjadi bentuk angka (string->int)
X[:, 1] = labelencoder_negara_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_jkelamin_X.fit_transform(X[:, 2])

#kemudian angkat tadi dirubah lagi menjadi bentuk binary dengan ONeHotEncoder
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()

#data pada kolom index 0 tidak perlu dipakai
X = X[:, 1:]

#Kemudian data tadi dibagi dalam 4 variabel, 2 variabel untuk 
#data training dan 2 variabel untuk data testing yaitu X_train, X_test, y_train, dan y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#scalling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#PROSES MELATIH DATA
#=======================================
#data dilatih sampai iterasi ke 10 dan epoch 100
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# MEMPREDIKSI HASIL TEST
# ======================================
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# MEMBUAT CONFUSION MATRIX UNTUK MELIHAT HASIL PROSES DIATAS
  # ======================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
