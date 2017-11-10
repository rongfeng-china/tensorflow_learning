import numpy as np
import pandas as pd
import tensorflow as tf


# Policy Gradient
class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.99,
            output_graph = True
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # memory
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):        
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observations')  
            self.tf_acts = tf.placeholder(tf.int32,[None,],name='actions_num')
            self.tf_vt = tf.placeholder(tf.float32,[None],name='action_value')

            # fc1
            layers = tf.layers.dense(
		inputs = self.tf_obs,
		units = 10,
		activation = tf.nn.tanh,
                kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.3),
                bias_initializer = tf.constant_initializer(0.1),
                name='fc1')

            all_act = tf.layers.dense(
                inputs = layers,
                units = self.n_actions,
                activation = None,
                kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.3),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'fc2')

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.variable_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels = self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob*self.tf_vt)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis,:]})
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        
	# train
        self.sess.run(self.train_op, feed_dict={
	    self.tf_obs: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm})
        
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]

        return discounted_ep_rs_norm 

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
