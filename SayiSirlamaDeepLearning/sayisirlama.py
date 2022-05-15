import os
import pickle

import cv2
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from warnings import filterwarnings
filterwarnings("ignore")

yol ="myData"

data = os.listdir("myData")
SınıfSayısı = len(data)
print(len(data))

X_resim=[]
Y_siniflar=[]

for i in range(SınıfSayısı):
    MyimageList =os.listdir(yol + "\\" +str(i))
    for y in MyimageList:
        img = cv2.imread(yol + "\\" + str(i) + "\\" + y)
        img =cv2.resize(img,(32,32))
        X_resim.append(img)
        Y_siniflar.append(i)

X_resimArrAy=np.array(X_resim)
Y_siniflarArrAy =np.array(Y_siniflar)

#Veri ayırma

X_train,X_test,Y_train,Y_test = train_test_split(X_resimArrAy,Y_siniflarArrAy,test_size=0.5,random_state=42)
X_train,X_val,Y_train,Y_val =train_test_split(X_train,Y_train,test_size=0.5,random_state=42)


#Preprocces

def preProcces(img):
    img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img =cv2.equalizeHist(img)
    img =img/255

    return img



X_train =np.array(list(map(preProcces,X_train)))
X_test =np.array(list( map(preProcces,X_test)))
X_val =np.array(list(map(preProcces,X_val)))

X_train =X_train.reshape(-1,32,32,1)
X_test =X_test.reshape(-1,32,32,1)
X_val =X_val.reshape(-1,32,32,1)



#data generate

dataGen =ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1
                            ,zoom_range=0.3,rotation_range=10)


dataGen.fit(X_train)



#Y değişkenlerimizi catorical hale getircen (one hot encoding)

Y_val=to_categorical(Y_val,SınıfSayısı)
Y_test=to_categorical(Y_test,SınıfSayısı)
Y_train=to_categorical(Y_train,SınıfSayısı)


model =Sequential()

model.add(Conv2D(input_shape=(32,32,1),filters=8,kernel_size=(5,5),activation="relu",padding="same"))
model.add(MaxPool2D())

model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPool2D())

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=SınıfSayısı,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer=("Adam"),metrics=["accuracy"])

batch_size =25

hist =model.fit_generator(dataGen.flow(X_train,Y_train,batch_size=batch_size),validation_data=(X_val,Y_val),epochs=14,steps_per_epoch=X_train.shape[0]//batch_size,shuffle=1)

"""pickle_out=open("model_traiiin.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()"""

model.save_weights("deneme.h5")

'val_loss', 'val_accuracy', 'loss', 'val_accuracy'
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

plt.plot(hist.history["accuracy"],label="Train accuracy")
plt.plot(hist.history["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()
















