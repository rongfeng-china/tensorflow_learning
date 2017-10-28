import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs, in_size, out_size, activation_fun=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_fun is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fun(Wx_plus_b,)
    return outputs


def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_pre = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

xs = tf.placeholder(tf.float32,[None,784]) # 28*28
ys = tf.placeholder(tf.float32,[None,10])

mnist = input_data.read_data_sets('MNIST_data',one_hot=True) #55000*784
'''x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - .5 + noise'''

prediction = add_layer(xs,784,10, activation_fun = tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)


for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
         print (compute_accuracy(mnist.test.images,mnist.test.labels))





