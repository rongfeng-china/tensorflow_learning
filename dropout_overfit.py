import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .3)

def add_layer(inputs,in_size,out_size,layer_name,activation_fun=None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size])+.1)
	Wx_plus_b = tf.matmul(inputs,Weights)+biases
	if activation_fun is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_fun(Wx_plus_b)
	tf.summary.histogram(layer_name+'/outputs',outputs)
	return outputs

with tf.name_scope('all_inputs'):
	xs = tf.placeholder(tf.float32,[None,64],name='x_input')
	ys = tf.placeholder(tf.float32,[None,10],name='y_input')

l1 = add_layer(xs,64,100,'l1',activation_fun =tf.nn.tanh)
prediction = add_layer(l1,100,10,'l2',activation_fun = tf.nn.softmax)

# cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)

sess.run(init)

for i in range(1000):
	sess.run(train_step,feed_dict={xs:X_train,ys:y_train})
	if i%50 == 0:
		train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train})
		test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test})
		train_writer.add_summary(train_result,i)
		test_writer.add_summary(test_result,i)





