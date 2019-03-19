import sys, time
import csv
import pandas as pd
import re
import numpy as np
from sklearn.metrics import *
from config import *
import tensorflow as tf


def fully_conacation(input, haddin_size, keep_rate=1.0, activation='leaky_relu'):
    dense_out = tf.layers.dense(inputs=input, units=haddin_size, use_bias=True)
    if activation is 'relu':
        dense_relu = tf.nn.relu(dense_out)
        dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
        return dense_relu
    elif activation is 'leaky_relu':
        dense_relu = tf.nn.leaky_relu(dense_out)
        dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
        return dense_relu
    elif activation is 'sigmoid':
        dense_relu = tf.nn.sigmoid(dense_out)
        dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
        return dense_relu


def conv2D(inputs, kernel_shape, bias_shape, strides, padding, kernel_name, bias_name, activation='leaky_relu',dropuot_rate=None):
    kernel = tf.get_variable(dtype=tf.float32, shape=kernel_shape, name=kernel_name)
    bias = tf.get_variable(dtype=tf.float32, shape=bias_shape, name=bias_name)
    conv_output = tf.nn.conv2d(input=inputs, filter=kernel, strides=strides, padding=padding) + bias
    if activation is 'relu':
        conv_output = tf.nn.relu(conv_output)
    elif activation is 'leaky_relu':
        conv_output = tf.nn.leaky_relu(conv_output)
    if dropuot_rate is not None:
        conv_output = tf.nn.dropout(conv_output, keep_prob=dropuot_rate)
    return conv_output

def dense_block(input,nb_layer,strides,padding,name):
    x = input
    for i in range(nb_layer):
        conv_out = conv2D(inputs=x,kernel_shape=[3,3,x.shape[3],x.shape[3]],strides=strides,bias_shape=x.shape[3],padding=padding,kernel_name=name+'conv'+str(i),bias_name=name+'bias'+str(i))
        x = tf.concat([x,conv_out],axis=-1)
    return x

def transition_block(input,output_channel,kernel_name,bias_name):
    x = conv2D(inputs=input,kernel_shape=[1,1,input.shape[3],output_channel],strides=[1,1,1,1],bias_shape=output_channel,padding='VALID',kernel_name=kernel_name,bias_name=bias_name)
    x_output = tf.nn.max_pool(value=x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    return x_output

def co_attention(s1,s2):
    # attention构造
    # 计算cosin匹配矩阵
    matrix_1 = tf.matmul(s1, tf.transpose(s2, perm=[0, 2, 1]))
    matrix_2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(s1), axis=-1)), axis=-1)
    matrix_3 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(s2), axis=-1)), axis=1)
    cosin_matrix_s1 = tf.div(matrix_1, tf.matmul(matrix_2, matrix_3))
    # 计算相似矩阵权重
    cosin_matrix_s1 = tf.nn.softmax(cosin_matrix_s1, dim=-1)
    print(cosin_matrix_s1)
    cosin_matrix_s2 = tf.transpose(cosin_matrix_s1, perm=[0, 2, 1])
    cosin_matrix_s2 = tf.nn.softmax(cosin_matrix_s2, dim=-1)
    a_s1 = tf.matmul(cosin_matrix_s1,s2)
    a_s2 = tf.matmul(cosin_matrix_s2,s1)
    return a_s1, a_s2

def Dynamic_LSTM(s1_f,s2_f,input_s1,input_s2,name):
    with tf.variable_scope("lst_"+str(name)+"_1"):
        cell_f_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,reuse=tf.AUTO_REUSE)
        cell_b_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,reuse =tf.AUTO_REUSE)
        lstm_output_s1, _ = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,
                                                                   inputs=input_s1,
                                                                   dtype=tf.float32)
        print(lstm_output_s1)
        lstm_fw_s1, lstm_bw_s1 = lstm_output_s1
        lstm_output_s1 = tf.concat([lstm_fw_s1, lstm_bw_s1], axis=-1)
    # with tf.variable_scope("lst_"+str(name)+"_2"):
        lstm_output_s2, _ = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,
                                                                   inputs=input_s2,
                                                                   dtype=tf.float32)
        print(lstm_output_s2)
        lstm_fw_s2, lstm_bw_s2 = lstm_output_s2
        lstm_output_s2 = tf.concat([lstm_fw_s2, lstm_bw_s2], axis=-1)
    attention_s1,attention_s2 = co_attention(lstm_output_s1,lstm_output_s2)
    concat_s1 = tf.concat([s1_f, lstm_output_s1,attention_s1], axis=-1)
    concat_s2 = tf.concat([s2_f, lstm_output_s2,attention_s2], axis=-1)
    encoder_s1_1 = fully_conacation(concat_s1,haddin_size=200,keep_rate=0.8)
    encoder_s2_1 = fully_conacation(concat_s2,haddin_size=200,keep_rate=0.8)
    decoder_s1_1 = fully_conacation(encoder_s1_1,haddin_size=concat_s1.shape[-1],keep_rate=0.8)
    decoder_s2_1 = fully_conacation(encoder_s2_1,haddin_size=concat_s2.shape[-1],keep_rate=0.8)
    loss_encoder_s1 = tf.losses.mean_squared_error(concat_s1, decoder_s1_1)
    loss_encoder_s2 = tf.losses.mean_squared_error(concat_s2, decoder_s2_1)
    tf.add_to_collection('losses',loss_encoder_s1)
    tf.add_to_collection('losses', loss_encoder_s2)
    return lstm_output_s1,lstm_output_s2,encoder_s1_1,encoder_s2_1

temp5 = tf.Variable(tf.random_normal(shape=(batch_size, batch_len, 100), mean=0, stddev=2, dtype=tf.float32))
temp = tf.slice(temp5,[0, 25, 0], [-1, 1, -1])
print(temp)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(temp)

