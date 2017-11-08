import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetWork:
    def __init__(self, n_actions, n_features, learning_rate = 0.01, reward_decay=.9,e_greedy=.9,replace_target_iter=300,memorty_size=500,batch_size=32,e_greedy_increment=None,output_graph=False):
        self.n_actions  = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        if e_greedy_increment is not None:
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_max 

        self.learn_step_counter = 0
        # memory [s,a,r,s_]
        self.memory = np.zeros((self.memory_size, n_features * 2+2))

        ## parameters for network
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params, e_params)]

        if output_graph:
            tf.summary.FileWriter("logs/",self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        ############################## build eval_net ############################3
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name = 's')
        self.q_target = tf.placeholder(tf.float32, [None,self.n_actions], name = 'Q_target')
        
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(0.,.3),tf.constant_initializer(.1)

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1, initializer = w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer = b_initializer, collections = c_name)
                l1 = tf.nn.relu(tf.matmul(self.s, w1)+b1)

            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions],initializer= w_initializer, collections=c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions], initializer = w_initializer, collections = c_names)
                self.q_eval = tf.matmul(l1,w2)+b2)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        ############################### build target_net ############################
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], names='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
 
            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1, initializer = w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer = b_initializer, collections = c_name)
                l1 = tf.nn.relu(tf.matmul(self.s, w1)+b1)
            
            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions],initializer= w_initializer, collections=c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions], initializer = w_initializer, collections = c_names)
                self.q_next = tf.matmul(l1,w2)+b2)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter - 0
