#ksize is somewhat similar to the filter size

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist=input_data.read_data_sets('data',one_hot=True)
#print(mnist)
learning_rate=0.001
batch_size=128
num_classes=10
num_features=784
dropout=0.8


X=tf.placeholder(dtype=tf.float32,shape=[None,num_features])#initial x
Y=tf.placeholder(dtype=tf.float32,shape=[None,num_classes])#final Y for a training example
keepprob=tf.placeholder(tf.float32)

def conv2d(X,W,b,strides=1):
	conv=tf.nn.conv2d(X,W,strides=[1,strides,strides,1],padding='SAME')#strides?
	conv=tf.nn.bias_add(conv,b)
	conv=tf.nn.relu(conv)
	return conv

def maxpool(X,k=2):
	return tf.nn.max_pool(X,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def overall(X,weights,biases,dropout):
	X=tf.reshape(X,shape=[-1,28,28,1])#how?#batch size height width and channel
	conv1=conv2d(X,weights['w1'],biases['b1'])
	conv1=maxpool(conv1)
	conv2=conv2d(conv1,weights['w2'],biases['b2'])
	conv2=maxpool(conv2)
	'''
	conv3=conv2d(conv2,weights['w5'],biases['b5'])
	conv3=maxpool(conv3)
	conv4=conv2d(conv3,weights['w6'],biases['b6'])
	conv4=maxpool(conv4)

	conv5=conv2d(conv4,weights['w7'],biases['b7'])
	conv5=maxpool(conv5)
	conv6=conv2d(conv5,weights['w8'],biases['b8'])
	conv6=maxpool(conv6)
	conv7=conv2d(conv6,weights['w9'],biases['b9'])
	conv7=maxpool(conv7)
	conv8=conv2d(conv7,weights['w10'],biases['b10'])
	conv8=maxpool(conv8)
	print(conv8.shape)
	'''

	fc1=tf.reshape(conv2,[-1,weights['w3'].get_shape().as_list()[0]])
	fc1=tf.add(tf.matmul(fc1,weights['w3']),biases['b3'])
	fc1=tf.nn.relu(fc1)
	fc1=tf.nn.dropout(fc1,dropout)#2nd parameter is the keepprob
	fc2=tf.add(tf.matmul(fc1,weights['w4']),biases['b4'])
	return fc2

weights={
	'w1':tf.Variable(tf.random_normal([5,5,1,32])),
	'w2':tf.Variable(tf.random_normal([5,5,32,64])),
	'w3':tf.Variable(tf.random_normal([7*7*64,1024])),
	'w4':tf.Variable(tf.random_normal([1024,num_classes])),
	'w5':tf.Variable(tf.random_normal([5,5,64,128])),
	'w6':tf.Variable(tf.random_normal([5,5,128,64])),

	'w7':tf.Variable(tf.random_normal([5,5,64,64])),
	'w8':tf.Variable(tf.random_normal([5,5,64,128])),
	'w9':tf.Variable(tf.random_normal([5,5,128,256])),
	'w10':tf.Variable(tf.random_normal([5,5,256,64]))
}

biases={
	'b1':tf.Variable(tf.random_normal([32])),
	'b2':tf.Variable(tf.random_normal([64])),
	'b3':tf.Variable(tf.random_normal([1024])),
	'b4':tf.Variable(tf.random_normal([num_classes])),
	'b5':tf.Variable(tf.random_normal([128])),
	'b6':tf.Variable(tf.random_normal([64])),

	'b7':tf.Variable(tf.random_normal([64])),
	'b8':tf.Variable(tf.random_normal([128])),
	'b9':tf.Variable(tf.random_normal([256])),
	'b10':tf.Variable(tf.random_normal([64]))
}



#practice

logits=overall(X,weights,biases,keepprob)
#logits1=overall(X,weights,biases,dropout=1)

prediction=tf.nn.softmax(logits)
#prediction1=tf.nn.softmax(logits)

correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))#axis=0 for rows and 1 for the columns
#correct_pred1=tf.equal(tf.argmax(prediction1,1),tf.argmax(Y,1))
#return the largest value accross the axis and the 2nd parameter is the axis specifier.the tf.equal returns a type of bool
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
#accuracy1=tf.reduce_mean(tf.cast(correct_pred1,tf.float32))

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		logits=logits,
		labels=Y
	))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)#initialize the optimizer
train_op=optimizer.minimize(cost)#after one step of grad descent when we get the o/p
print(type(train_op))

init=tf.global_variables_initializer()
costs=[]
acu=[]
with tf.Session() as sess:
	sess.run(init)
	for i in range(0,1600):
		batch_x,batch_y=mnist.train.next_batch(batch_size)#treat it as one training example
		sess.run([train_op],feed_dict={
				X:batch_x,
				Y:batch_y,
				keepprob:0.8
			})
		if i%10==0 and i!=0:
			accu1,cost1=sess.run([accuracy,cost],feed_dict={
					X:batch_x,
					Y:batch_y,
					keepprob:0.8
				})
			accu1=accu1*100
			print("training accuracy after "+str(i)+" iterations is "+str(accu1))
			print("loss after "+str(i)+" iterations is "+str(cost1))
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per 10 iterations)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	plt.plot(np.squeeze(acu))
	plt.ylabel('accuracy')
	plt.xlabel('iterations (per 10 iterations)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	print("testing accuracy is "+str(sess.run(accuracy,feed_dict={
			X:mnist.test.images[:256],
			Y:mnist.test.labels[:256],
			keepprob:1.0
		})))
