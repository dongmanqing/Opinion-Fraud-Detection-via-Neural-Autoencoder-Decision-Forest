# ======================Loading packages ============================== #
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import csv
import os
import time
from sklearn import preprocessing
import pickle

# ======================functions================================ #
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict


def reconstructlabel(label):
    length = len(label)
    newlabel = []
    for index in range(length):
        if label[index] == 0:
            newlabel.append([0, 1])
        else:
            newlabel.append([1, 0])
    return newlabel


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# constructing neural random forest
def model(input_layer, w_t_e, w_d_e, w_l_e, p_keep_hidden):
    assert (len(w_t_e) == len(w_d_e))
    assert (len(w_t_e) == len(w_l_e))
    decision_p_e = []
    leaf_p_e = []
    for w_t, w_d, w_l in zip(w_t_e, w_d_e, w_l_e):
        tree_layer = tf.nn.relu(tf.matmul(input_layer, w_t))
        tree_layer = tf.nn.dropout(tree_layer, p_keep_hidden)

        decision_p = tf.nn.sigmoid(tf.matmul(tree_layer, w_d)) 
        leaf_p = tf.nn.softmax(w_l) 

        decision_p_e.append(decision_p)
        leaf_p_e.append(leaf_p)
    return decision_p_e, leaf_p_e


def neural_random_forest(decision_p_e, leaf_p_e, n_depth, n_leaf, n_batch):
    flat_decision_p_e = []

    for decision_p in decision_p_e:
        decision_p_comp = tf.subtract(tf.ones_like(decision_p), decision_p)
        decision_p_pack = tf.stack([decision_p, decision_p_comp])
        flat_decision_p = tf.reshape(decision_p_pack, [-1])
        flat_decision_p_e.append(flat_decision_p)

    batch_0_indices = \
        tf.tile(tf.expand_dims(tf.range(0, n_batch * n_leaf, n_leaf), 1), [1, n_leaf])

    # The routing probability computation
    in_repeat = int(n_leaf / 2)
    out_repeat = n_batch

    batch_complement_indices = \
        np.array([[0] * in_repeat, [n_batch * n_leaf] * in_repeat]
                 * out_repeat).reshape(n_batch, n_leaf)

    mu_e = []
    # iterate over each tree
    for i, flat_decision_p in enumerate(flat_decision_p_e):
        mu = tf.gather(flat_decision_p, tf.add(batch_0_indices, batch_complement_indices))
        mu_e.append(mu)

    # from the second layer to the last layer, we make the decision nodes
    for d in range(1, n_depth + 1):
        indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
        tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                          [1, 2 ** (n_depth - d + 1)]), [1, -1])
        batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [n_batch, 1]))

        in_repeat = int(in_repeat / 2)
        out_repeat = int(out_repeat * 2)

        batch_complement_indices = \
            np.array([[0] * in_repeat, [n_batch * n_leaf] * in_repeat]
                     * out_repeat).reshape(n_batch, n_leaf)

        mu_e_update = []
        for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
            mu = tf.multiply(mu, tf.gather(flat_decision_p, tf.add(batch_indices, batch_complement_indices)))
            mu_e_update.append(mu)

        mu_e = mu_e_update

    return mu_e


def probability_y_x(mu_e, leaf_p_e):
    py_x_e = []
    py_x_leaf_e = []
    for mu, leaf_p in zip(mu_e, leaf_p_e):
        # average all the leaf p
        py_x_tree = tf.reduce_sum(
            tf.multiply(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]),
                        tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])), 1)
        a_tree = tf.multiply(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]),
                             tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1]))
        py_x_e.append(py_x_tree)
        py_x_leaf_e.append(a_tree)

    py_x_e = tf.stack(py_x_e)
    py_x = tf.reduce_sum(py_x_e, 0)
    final = tf.nn.softmax(py_x)
    # py_x_leaf_e = tf.stack(py_x_leaf_e)
    return final


def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))


# ======================loading================================== #
the_data = unpickle('dataset.p')
head = the_data[0,:]
body = the_data[1:,:]

# Notes: different features
'''
fea_history = all_fea[:,[1,12,19,35]]   # 4
fea_rating = all_fea[:,[0,2,10,11,13,14,15,16,17,18,28,29,30,31,32]]  # 15
fea_feedback = all_fea[:,[3,4,5,6,21,22,23,24,25,26,33,34]]  # 12
fea_time = all_fea[:,[7,8,9,20,27]] # 5
fea_products = all_fea[:,36:41] # 5
fea_review = all_fea[:,41:] # 7
'''

feature = body[:,1:]
label = body[:,0]

N_feature = 48

# Scalling
# feature_normalized = feature
feature_normalized=preprocessing.scale(feature)
# feature_normalized=preprocessing.minmax_scale(feature,feature_range=(0,1))
# feature_normalized=feature/sum(feature)

# Set training and testing
trX = feature_normalized[0:4000]
teX = feature_normalized[4000:]

newlabel=reconstructlabel(label.astype(int))
trY = np.array(newlabel[0:4000])
teY = np.array(newlabel[4000:])
#
trX = trX.reshape(-1, N_feature)
teX = teX.reshape(-1, N_feature)

# Parameter Settings
DEPTH   = 3                 # Depth of a tree
N_LEAF  = 2 ** (DEPTH + 1)  # Number of leaf node 
N_LABEL = 2                # Number of classes
N_TREE  = 5                 # Number of trees (ensemble)
N_BATCH = 50             # Number of data points per mini-batch

# Input X, output Y
x = tf.placeholder("float", [N_BATCH, N_feature])
y = tf.placeholder("float", [N_BATCH, N_LABEL])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# ====================Autoencoder========================================#
num_hidden_1 = 64  
num_hidden_2 = 96  
# num_hidden_3 = 64
# num_hidden_4 = 96
num_input = N_feature  

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([num_input, num_hidden_1], stddev=0.1)),
    'encoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2], stddev=0.1)),
    # 'encoder_h3': tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_3], stddev=0.1)),
    # 'encoder_h4': tf.Variable(tf.truncated_normal([num_hidden_3, num_hidden_4], stddev=0.1)),
    # 'decoder_h1': tf.Variable(tf.truncated_normal([num_hidden_4, num_hidden_3], stddev=0.1)),
    # 'decoder_h2': tf.Variable(tf.truncated_normal([num_hidden_3, num_hidden_2], stddev=0.1)),
    'decoder_h1': tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_1], stddev=0.1)),
    'decoder_h2': tf.Variable(tf.truncated_normal([num_hidden_1, num_input], stddev=0.1))
}

biases = {
    'encoder_b1': tf.Variable(tf.constant(0.1, shape=[num_hidden_1])),
    'encoder_b2': tf.Variable(tf.constant(0.1, shape=[num_hidden_2])),
    # 'encoder_b3': tf.Variable(tf.constant(0.1, shape=[num_hidden_3])),
    # 'encoder_b4': tf.Variable(tf.constant(0.1, shape=[num_hidden_4])),
    # 'decoder_b1': tf.Variable(tf.constant(0.1, shape=[num_hidden_3])),
    # 'decoder_b2': tf.Variable(tf.constant(0.1, shape=[num_hidden_2])),
    'decoder_b1': tf.Variable(tf.constant(0.1, shape=[num_hidden_1])),
    'decoder_b2': tf.Variable(tf.constant(0.1, shape=[num_input]))
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    # layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
    #                                biases['encoder_b3']))
    # layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
    #                                biases['encoder_b4']))

    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))

    # layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
    #                                biases['decoder_b3']))
    # layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
    #                                biases['decoder_b4']))
    return layer_2

# Construct model
encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = x

# ====================Neural decision forest=================================#
num_out = 96
num_output = 128
N_nodes = 256

W_hidden = weight_variable([num_out, num_output])
b_hidden = weight_variable([num_output])
tree_input = tf.nn.relu(tf.matmul(encoder_op, W_hidden) + b_hidden)
tree_input = tf.nn.dropout(tree_input, p_keep_conv)

w_t_ensemble = []
w_d_ensemble = []
w_l_ensemble = []
for i in range(N_TREE):
    w_t_ensemble.append(weight_variable([num_output, N_nodes])) 
    w_d_ensemble.append(init_prob_weights([N_nodes, N_LEAF], -1, 1)) 
    w_l_ensemble.append(
        init_prob_weights([N_LEAF, N_LABEL], -2, 2))  

decision_p_e, leaf_p_e = model(tree_input, w_t_ensemble, w_d_ensemble, w_l_ensemble, p_keep_hidden)
mu_e = neural_random_forest(decision_p_e, leaf_p_e, DEPTH, N_LEAF, N_BATCH)
py_x = probability_y_x(mu_e, leaf_p_e) 

# loss function
cost = tf.add(tf.reduce_mean(-tf.multiply(tf.log(py_x), y)),tf.reduce_mean(tf.pow(y_true - y_pred, 2)))
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict = tf.argmax(py_x, 1)

# =============== training ========================== #
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(400):
    # One epoch
    for start, end in zip(range(0, len(trX), N_BATCH), range(N_BATCH, len(trX), N_BATCH)):
        sess.run(train_step, feed_dict={x: trX[start:end], y: trY[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})
    results = []
    for start, end in zip(range(0, len(teX), N_BATCH), range(N_BATCH, len(teX), N_BATCH)):
        results.extend(np.argmax(teY[start:end], axis=1) ==
                        sess.run(predict, feed_dict={x: teX[start:end], p_keep_conv:1.0, p_keep_hidden: 1.0}))
    print('Epoch: %d, Test Accuracy: %f' % (i + 1, np.mean(results)))


