#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf
import numpy as np
import face as f
import math


input_layer_size    =   361 # 19x19 pixel images flattened
hidden_layer_size   =   200 # 30 hidden units
num_label           =   2 #output layer - output is a face 1 or not a face 0

sess = tf.InteractiveSession()

face = f.make_Face()
x = tf.placeholder(tf.float32, [None,361])
W_1 = tf.Variable(tf.truncated_normal([361,hidden_layer_size],stddev=1./math.sqrt(361)))
b_1 = tf.Variable(tf.random_normal([hidden_layer_size]))
W_2 = tf.Variable(tf.random_normal([hidden_layer_size,2],stddev=1./math.sqrt(hidden_layer_size)))
b_2 = tf.Variable(tf.random_normal([2]))

layer_1 = tf.add(tf.matmul(x,W_1), b_1)
layer_1 = tf.nn.relu(layer_1)

y = tf.nn.softmax(tf.matmul(layer_1,W_2) + b_2)

y_ = tf.placeholder(tf.float32, [None,2]);

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for j in range(80000):
    batch_xs, batch_ys = face.next_batch(10)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_set =  np.loadtxt("svm.test.normgrey")
print(str(sess.run(accuracy,feed_dict={x: face.training_xs, y_: face.training_ys})) + " Training Accuracy")
print(str(sess.run(accuracy,feed_dict={x: face.test_data, y_: face.test_labels})) + " Test Accuracy")
