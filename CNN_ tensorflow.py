import cv2
print (cv2.__version__)

import numpy as np
print (np.__version__)

import os
import math

import tensorflow as tf
print("Tensorflow version "+tf.__version__)


#######################################################################
IMG_SIZE=28
training_iters = 50 # number of time the network is train
learning_rate = 0.0005

#######################################################################
#Load the data
#######################################################################

train_data = np.load('../CNN_tensorflow/train_data.npy', allow_pickle = True)
#train = train_data[:-50] si on a besoin de prendre des images dans tests 
#test = train_data[-50:] si on a besoin de prendre des images dans tests 
print ('fichier binaire loaded')
print ("Train data shape: {shape}".format(shape=train_data.shape))
train_X=np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)/255
train_Y=np.array([i[1] for i in train_data])
print ("image shape: {shape}".format(shape=train_X.shape))
print ("label shape: {shape}".format(shape=train_Y.shape))


#######################################################################
#Preparation du modele
#######################################################################
batch_size = 64
nb_classes = 5
rate = 0.75
K = 24  # first convolutional layer output depth 8
L = 48  # second convolutional layer output depth 16 
M = 64  # third convolutional layer 32
N = 300 # fully connected layer 300

X = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])

Y_ = tf.placeholder(tf.float32, [None, nb_classes])

def conv2d(X,W,b,strides):
	X = tf.nn.relu(tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding='SAME') + b)
	return X

# function maxpooling si on décide de maintenir la variables strides à 1 dans la fonction conv2d
#def maxpool2d(x,k=2)
#	return tf.nn.max_pool(x, ksize=[1, k,  k, 1], strides=[1, k, k, 1], padding='SAME')

# définition des variables Poids

weights = {
	'W1': tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1)),
	'W2': tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1)),
	'W3': tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1)),
	'W4': tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1)),
	'W5': tf.Variable(tf.truncated_normal([N, nb_classes], stddev=0.1)),

	#'wc1': tf.get_variable('w0',shape=(3,3,1,32), initializer = tf.contrib.layers.xavier_initializer()), kernem 3x3 appliqué 32 fois
}

# définition des variables Biais

biaises = {
	'B1': tf.Variable(tf.constant(0.1, tf.float32, [K])),
	'B2': tf.Variable(tf.constant(0.1, tf.float32, [L])),
	'B3': tf.Variable(tf.constant(0.1, tf.float32, [M])),
	'B4': tf.Variable(tf.constant(0.1, tf.float32, [N])),
	'B5': tf.Variable(tf.constant(0.1, tf.float32, [nb_classes])),

}


def convolution_network(X, weights, biaises):
	conv1 = conv2d(X, weights['W1'], biaises['B1'],1)

	conv2 = conv2d(conv1, weights['W2'], biaises['B2'],2)

	conv3 = conv2d(conv2, weights['W3'], biaises['B3'],2)
#reshape avant de faire la suite
	full_connected_layer = tf.reshape(conv3, shape=[-1, 7 * 7 * M])

	full_connected_layer = tf.nn.relu(tf.matmul(full_connected_layer,weights['W4'])+biaises['B4'])

	#full_connected_layer = tf.nn.dropout(full_connected_layer,rate)
	output = tf.matmul(full_connected_layer,weights['W5'])+biaises['B5']
	#output=tf.nn.softmax(output)
	return output


#######################################################################
#######################################################################
Y = convolution_network (X, weights, biaises)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#######################################################################
# here we check if the index of the maximum value of the calculated image is equal to actual labelled image
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
# calculate accuracy across all the given umages and average them
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#######################################################################

init = tf.global_variables_initializer() # initialise toutes les variables de types Variables in Tensorflow
with tf.Session () as sess:
	sess.run(init)
	train_loss = []
	test_loss = []
	train_accuracy = []
	test_accuracy = []
#	summary_writer = tf.summary.FimeWriter('./Output',sess.graph)
	for i in range(training_iters):
		for batch in range(len(train_X)//batch_size):
			batch_X = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
			batch_Y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]
			opt = sess.run(optimizer, feed_dict={X: batch_X, Y_: batch_Y})
			loss, acc = sess.run([cost, accuracy],feed_dict={X: batch_X, Y_: batch_Y})
		print("Iteration " + str(i) + ", Loss = " + "{:.6f}".format(loss) + ",training accuracy = " + "{:.5f}".format(acc))
		print("Optimazation finished")

# A ajouter si l'on a pris les images de tests train_data[-50:]

		# Calculate accuracy for the test images
#		test_acc, valid_loss = sess.run([cost, accuracy],feed_dict={x: test_X, y: test_Y})
#		train_loss.append(loss)
#		test_loss.append(valid_loss)
#		train_accuracy.append(acc)
#		test_accuracy.append(test_acc)
#		print("Testing Accuracy:","{:.5f}".format(test_acc))
#	summary_writer.close()













