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


val=os.path.isfile('places_train.npy')
if val==False:
	data_train=[]
	labels_train=[]
	count=-1
	er=0
	if os.path.exists('dataset/seg_train'):
		place_list=os.listdir('dataset/seg_train')
		for i in place_list:
			if os.path.exists('dataset/seg_train/'+i):
				place=os.listdir('dataset/seg_train/'+i)
				count=count+1
				for a in place:
					try:
						image=cv2.imread('dataset/seg_train/'+i+'/'+a)
						image=Image.fromarray(image,'RGB')
						image=image.resize((50,50))
						data_train.append(np.array(image))
						labels_train.append(count)
					except AttributeError:
						er=er+1
			else:
				print("not opening folder")
	else:
		print("not opening folder")			
	print("files not opened: "+str(er))
	places_train=np.array(data_train)
	labels_train=np.array(labels_train)

	np.save("places_train",places_train)
	np.save("labels_train",labels_train)
	print("dataset saved")

places_train=np.load("places_train.npy")
labels_train=np.load("labels_train.npy")
print("train dataset loaded")

#shuffle before splitting
a=np.arange(places_train.shape[0])
#making s an array with 0 to cells.shape[0]-1 numbers
np.random.shuffle(a)
#shuffling those numbers
places_train=places_train[a]
labels_train=labels_train[a]



num_classes=len(np.unique(labels_train))
len_data=len(places_train)

#0.1 percent is test
x_train=places_train[:]
x_train=x_train.astype('float32')/255

y_train=labels_train[:]

#onehot encoding will change the labels in binary format.
#2 can be [1,0] if 2 neurons and [0,0,1,0] if 4 neurons.
y_train=keras.utils.to_categorical(y_train,num_classes)


#preparing test images
val=os.path.isfile('places_test.npy')
if val==False:
	data_test=[]
	labels_test=[]
	count=-1
	er=0
	if os.path.exists('dataset/seg_test'):
		place_list=os.listdir('dataset/seg_test')
		for i in place_list:
			if os.path.exists('dataset/seg_test/'+i):
				place=os.listdir('dataset/seg_test/'+i)
				count=count+1
				for a in place:
					try:
						image=cv2.imread('dataset/seg_test/'+i+'/'+a)
						image=Image.fromarray(image,'RGB')
						image=image.resize((50,50))
						data_test.append(np.array(image))
						labels_test.append(count)
					except AttributeError:
						er=er+1
			else:
				print("not opening folder")
	else:
		print("not opening folder")			
	print("files not opened: "+str(er))
	places_test=np.array(data_test)
	labels_test=np.array(labels_test)

	np.save("places_test",places_test)
	np.save("labels_test",labels_test)
	print("Test dataset saved")

places_test=np.load("places_test.npy")
labels_test=np.load("labels_test.npy")
print("Test dataset loaded")

a=np.arange(places_test.shape[0])
#making s an array with 0 to cells.shape[0]-1 numbers
np.random.shuffle(a)
#shuffling those numbers
places_test=places_test[a]
labels_test=labels_test[a]



num_classes=len(np.unique(labels_test))
len_data=len(places_test)

#0.1 percent is test
x_test=places_test[:]
x_test=x_test.astype('float32')/255

y_test=labels_test[:]

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
model.add(Dense(6,activation="softmax"))
#last has 2 neurons and softmax output

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)

#accuracy=model.evaluate(x_test,y_test,verbose=1)

#print("Test accuracy: ",accuracy[1])

val=os.path.isfile("places_save.h5")
if val==False:
	model.save('places_save.h5')


#do the prediction
data_pred=[]
er=0
if os.path.exists('dataset/seg_pred'):
	place=os.listdir('dataset/seg_pred')
	for a in place:
		try:
			image=cv2.imread('dataset/seg_pred/'+a)
			image=Image.fromarray(image,'RGB')
			image=image.resize((50,50))
			data_pred.append(np.array(image))
		except AttributeError:
			er=er+1
else:
	print("not opening folder")			
#print("files not opened: "+str(er))
places_pred=np.array(data_pred)

model=load_model('places_save.h5')
prediction=model.predict(places_pred,verbose=1)



#sigmoid with relu works good
#otherwise use sigmoid in the last layer as in sigmoid if 1 goes up other goes down.Also it depends on others

