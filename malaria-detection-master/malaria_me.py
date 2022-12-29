from PIL import Image
import numpy as np 
import os
import cv2
import keras
from keras.utils import np_utils
from keras.models import Sequential,load_model
#from keras.models import MaxPooling2D,Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Dense,Flatten,Dropout
#import pandas as pd 


val=os.path.isfile('Cells.npy')
if val==False:
	data=[]
	labels=[]
	parasitized=os.listdir('cell_images/Parasitized/')
	for a in parasitized:
		try:
			image=cv2.imread('cell_images/Parasitized/'+a)
			image=Image.fromarray(image,'RGB')
			image=image.resize((50,50))
			data.append(np.array(image))
			labels.append(0)
		except AttributeError:
			print(" ")

	uninfected=os.listdir('cell_images/Uninfected/')
	for a in uninfected:
		try:
			image=cv2.imread('cell_images/Uninfected/'+a)
			image=Image.fromarray(image,'RGB')
			image=image.resize((50,50))
			data.append(np.array(image))
			labels.append(1)
		except AttributeError:
			print(" ")

	Cells=np.array(data)
	labels=np.array(labels)

	np.save("Cells",Cells)
	np.save("labels",labels)
	print("dataset saved")

Cells=np.load("Cells.npy")
labels=np.load("labels.npy")
print("dataset loaded")




	




#shuffle before splitting
a=np.arange(Cells.shape[0])
#making s an array with 0 to cells.shape[0]-1 numbers
np.random.shuffle(a)
#shuffling those numbers
Cells=Cells[a]
labels=labels[a]



num_classes=len(np.unique(labels))
len_data=len(Cells)

#0.1 percent is test
x_train,x_test=Cells[(int(0.1*len_data)):],Cells[:(int(0.1*(len_data)))]
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

y_train,y_test=labels[(int(0.1*len_data)):],labels[:(int(0.1*(len_data)))]

#onehot encoding will change the labels in binary format.
#2 can be [1,0] if 2 neurons and [0,0,1,0] if 4 neurons.

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))
#last has 2 neurons and softmax output

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)

accuracy=model.evaluate(x_test,y_test,verbose=1)

print("Test accuracy: ",accuracy[1])

val=os.isfile("cells.h5")
if val==False:
	model.save('cells.h5')
