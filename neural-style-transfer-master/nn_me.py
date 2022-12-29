import os
import sys
import scipy.io 
#to read and write data to a variety of file formats
import scipy.misc
#other scipy classes
#import matplotlib.pyplot as plt 
from PIL import Image
import cv2
from nst_utils import *
import numpy as np 
import tensorflow as tf 

model = load_vgg_model("pretrained_model/imagenet-vgg-verydeep-19.mat")
#model is a python dictionary containing variable and its value

for key,value in model.items():
	print(str(key)+str(" ")+str(value))
	print("\n")

#The above lines gives the details of what the vgg model looks like.

content_image=scipy.misc.imread("images/louvre.jpg")
cv2.imshow('r',content_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def compute_content_cost(a_C,a_G):
	m,n_H,n_W,n_C=a_G.get_shape().as_list()
	#get shape is from tensorflow and as_list will give you the shape as a list of numbers
	a_C_unrolled=tf.transpose(tf.reshape(a_C,[-1]))
	#we are converting the a_C to a single 1D vector.
	a_G_unrolled=tf.transpose(tf.reshape(a_G,[-1]))
	J_content=tf.reduce_sum((a_C_unrolled-a_G_unrolled)**2)/(4*n_H*n_W*n_C)
	#above statement will minus each ele and square and then add that to others.
	return J_content

style_image=scipy.misc.imread("images/monet_800600.jpg")
cv2.imshow("r",style_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def gram_matrix(A):
	GA=tf.matmul(A,tf.transpose(A))
	#matmul is dot product
	return GA

def compute_layer_style_cost(a_S,a_G):
	m,n_H,n_W,n_C=a_G.get_shape().as_list()
	a_S=tf.reshape(a_S,[n_H*n_W,n_C])
	a_G=tf.reshape(a_G,[n_H*n_W,n_C])
	G_S=gram_matrix(tf.transpose(a_S))
	G_G=gram_matrix(tf.transpose(a_G))
	J_style_layer=tf.reduce_sum((G_S-G_G)**2)/(4*n_C**2*(n_W*n_H)**2)
	return J_style_layer

#we are going to combine the style of diff layers.

STYLE_LAYERS=[
	('conv1_1',0.2),
	('conv2_1',0.2),
	('conv3_1',0.2),
	('conv4_1',0.2),
	('conv5_1',0.2),
]

#if gen image will softly follow the style then use larger weights for deeper and smaller for first layers.

def compute_style_cost(model,STYLE_LAYERS):
	#model is that dict
	J_style=0
	for layer_name,coeff in STYLE_LAYERS:
		out=model[layer_name]
		a_S=sess.run(out)
		a_G=out
		J_style_layer=compute_layer_style_cost(a_S,a_G)
		J_style+=coeff*J_style_layer
	return J_style

def total_cost(J_content,J_style,alpha=10,beta=40):
	J=alpha*J_content+beta*J_style
	#alpha is a parameter weighting the importance of the content cost.
	return J

#all the cost part is done!!

tf.get_default_graph()

sess=tf.InteractiveSession()

content_image=scipy.misc.imread("images/louvre_small.jpg")
content_image=reshape_and_normalize_image(content_image)

#reshape means matching the i/p of VGG16
#subtract the mean to match the expected i/p of VGG16

style_image=scipy.misc.imread("images/monet.jpg")
style_image=reshape_and_normalize_image(style_image)

generated_image=generate_noise_image(content_image)
#Make it closer to content so that we get a good image quickly.

cv2.imshow("r",generated_image[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

g=tf.Graph()
with g.as_default():
	sess.run(model['input'].assign(content_image))

out=model['conv4_2']

a_C=sess.run(out)

a_G=out

J_content=compute_content_cost(a_C,a_G)

sess.run(model['input'].assign(style_image))

J_style=compute_style_cost(model,STYLE_LAYERS)

J=total_cost(J_content,J_style,alpha=10,beta=40)

optimizer=tf.train.AdamOptimizer(2.0)

train_step=optimizer.minimize(J)

def model_nn(sess,input_image,num_iterations=200):
	sess.run(tf.global_variables_initializer())
	sess.run(model['input'].assign(input_image))
	#feeding the model dict this way with the image
	for i in range(num_iterations):
		_=sess.run(train_step)
		generated_image=sess.run(model['input'])
		if i%20==0:
			Jt,Jc,Js=sess.run([J,J_content,J_style])
			print("iteration "+str(i)+" :")
			print("total cost ="+str(Jt))
			print("content cost = "+str(Jc))
			print("style cost = "+str(Js))
			save_image("outputs/"+str(i)+".png",generated_image)
	save_image('outputs/generated_image.jpg',generated_image)
	return generated_image

model_nn(sess,generated_image)



