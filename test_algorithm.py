# -*- coding:utf-8 -*-
import csv
import numpy as np
import re

# with open('./miss.txt','r',encoding='utf-8') as f,open('./s1.txt','r',encoding='utf-8') as s1_index_old,open('./s2.txt','r',encoding='utf-8') as s2_index_old,open('./QQP/s1.txt','r',encoding='utf-8') as s1_char_old,open('./QQP/s2.txt','r',encoding='utf-8') as s2_char_old,open('./delete/s1_index.txt','w+',encoding='utf-8') as s1_index,open('./delete/s2_index.txt','w+',encoding='utf-8') as s2_index,open('./delete/s1_char.txt','w+',encoding='utf-8') as s1_char,open('./delete/s2_char.txt','w+',encoding='utf-8') as s2_char:#读文件
#     file = f.readlines()#按行分开
#
#     dic = {}
#     for item in file:#按照行读，每次读一行
#         label,miss = item.split()#按照每行切分开
#         if str(miss) not in dic:
#             dic[str(miss)] = 1
#         else:
#             dic[str(miss)] += 1
#     for item in dic.items():
#         print(str(item[0])+': '+str(item[1]))

def get_char(filepath):
    with open(filepath,'r',encoding='utf-8',errors='ignore') as f1,open('./MSRP/s1_test.txt','w+',encoding='utf-8',errors='ignore') as s1,open('./MSRP/s2_test.txt','w+',encoding='utf-8',errors='ignore') as s2:
        data_lines = f1.readlines()
        r1 = "[\�\!\.\_,$\"%^*(+]+|[+！\”\“，\‘。？、~@#￥%&*（）;?:`>><()]+"
        count = 0
        for i in data_lines:
            i = i.strip().split('	')
            ss1 = i[3]
            ss2 = i[4]
            ss1 = re.sub(r1, '', ss1)
            ss2 = re.sub(r1, '', ss2)
            ss1 = ss1.split()
            ss2 = ss2.split()
            if len(ss1) <= 2:
                continue
            if len(ss2) <= 2:
                continue
            if len(i[0]) == 0:
                continue
            count+=1
            print(count)
            s1.write(' '.join(ss1) + '\n')
            s2.write(' '.join(ss2)+'\n')
# get_char('./MSRP/test.txt')
import tensorflow as tf
## prepare the original data
with tf.name_scope('data'):
     x_data = np.random.rand(100).astype(np.float32)
     y_data = 0.3*x_data+0.1
##creat parameters
with tf.name_scope('parameters'):
    with tf.name_scope('weights'):
        weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
        tf.summary.histogram('weight',weight)
    with tf.name_scope('biases'):
        bias = tf.Variable(tf.zeros([1]))
        tf.summary.histogram('bias',bias)
##get y_prediction
with tf.name_scope('y_prediction'):
     y_prediction = weight*x_data+bias
##compute the loss
with tf.name_scope('loss'):
     loss = tf.reduce_mean(tf.square(y_data-y_prediction))
     tf.summary.scalar('loss',loss)
##creat optimizerpip
optimizer = tf.train.GradientDescentOptimizer(0.5)
#creat train ,minimize the loss
with tf.name_scope('train'):
     train = optimizer.minimize(loss)
#creat init
with tf.name_scope('init'):
     init = tf.global_variables_initializer()
##creat a Session
sess = tf.Session()
#merged
merged = tf.summary.merge_all()
##initialize
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs, step)