import numpy as np
import re
from config import *
# -*- coding: utf-8 -*-
import csv
import collections
# file = open('./test.txt','w+',encoding='utf-8')

def get_data(filepath_tran,filepath_test):
    with open(filepath_tran,'r',encoding='utf-8',errors='ignore') as data_train,open(filepath_test,'r',encoding='utf-8',errors='ignore') as data_test,open('./s1.txt','w+', encoding='ANSI',errors='ignore') as s1, open('./s2.txt', 'w+', encoding='ANSI',errors='ignore') as s2, open('./label.txt', 'w+', encoding='ANSI',errors='ignore') as label,open('./s1_test.txt','w+', encoding='ANSI',errors='ignore') as s1_test, open('./s2_test.txt', 'w+', encoding='ANSI',errors='ignore') as s2_test, open('./label_test.txt', 'w+', encoding='ANSI',errors='ignore') as label_test, open('./word2index.txt','w+',encoding='ANSI',errors='ignore') as word_index,open('./vector.txt','w+',encoding='utf-8',errors='ignore') as vector,open('./out_of_vecab.txt','w+',encoding='utf-8',errors='ignore') as out_vecab:
        data_train_lines = data_train.readlines()
        data_test_lines = data_test.readlines()
        word2index = collections.OrderedDict()
        embedding = collections.OrderedDict()
        out_of_vocab = collections.OrderedDict()
        embedding_table = load_vector('./glove.840B.300d.txt')
        r1 = "[\�\!\.\_,$\"%^*(+]+|[+！\”\“，\‘。？、~@#￥%&*（）;\’?:`>><()]+"
        s = 0
        for i in data_train_lines:
            s += 1
            print('第', s, '行')
            i = i.strip().split('	')
            ss1 = i[3]
            ss2 = i[4]
            ss1 = re.sub(r1, '', ss1)
            ss2 = re.sub(r1, '', ss2)
            ss1 = ss1.split()
            ss2 = ss2.split()
            if len(ss1)<=2:
                continue
            if len(ss2)<=2:
                continue
            if len(i[0]) == 0:
                continue
            else:
                label.write(i[0]+'\n')
            ss2.insert(0,'[SEP]')

            while len(ss1)>batch_len:
                ss1.pop(len(ss1)-1)
            while len(ss1)<batch_len:
                ss1.append(',')

            while len(ss2)>batch_len+1:
                ss2.pop(len(ss2)-1)
            while len(ss2)<batch_len+1:
                ss2.append(',')

            for j in range(len(ss1)):
                if ss1[j] not in embedding_table:
                    if ss1[j] not in out_of_vocab:
                        out_of_vocab[ss1[j]] = len(out_of_vocab)
                    embedding[ss1[j]] = np.random.uniform(-0.25000,0.25000,300)
                else:
                    embedding[ss1[j]] = embedding_table[ss1[j]]
            for j in range(len(ss2)):
                if ss2[j] not in embedding_table:
                    if ss2[j] not in out_of_vocab:
                        out_of_vocab[ss2[j]] = len(out_of_vocab)
                    embedding[ss2[j]] = np.random.uniform(-0.25000,0.25000,300)
                else:
                    embedding[ss2[j]] = embedding_table[ss2[j]]

            for item in embedding.items():
                if item[0] not in word2index:
                    word2index[item[0]] = len(word2index)
            for j in range(len(ss1)):
                if ss1[j] in word2index:
                    ss1[j] = str(word2index[ss1[j]])
            for j in range(len(ss2)):
                if ss2[j] in word2index:
                    ss2[j] = str(word2index[ss2[j]])
            s1.write(' '.join(ss1) + '\n')
            s2.write(' '.join(ss2) + '\n')

        s = 0
        for i in data_test_lines:
            s += 1
            print('第', s, '行')
            i = i.strip().split('	')
            ss1 = i[3].lower()
            ss2 = i[4].lower()
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
            else:
                label_test.write(i[0]+'\n')
            ss2.insert(0, '[SEP]')

            while len(ss1) > batch_len:
                ss1.pop(len(ss1) - 1)
            while len(ss1) < batch_len:
                ss1.append(',')

            while len(ss2) > batch_len + 1:
                ss2.pop(len(ss2) - 1)
            while len(ss2) < batch_len + 1:
                ss2.append(',')

            for j in range(len(ss1)):
                if ss1[j] not in embedding_table:
                    if ss1[j] not in out_of_vocab:
                        out_of_vocab[ss1[j]] = len(out_of_vocab)
                    embedding[ss1[j]] = np.random.uniform(-0.25000,0.25000,300)
                else:
                    embedding[ss1[j]] = embedding_table[ss1[j]]
            for j in range(len(ss2)):
                if ss2[j] not in embedding_table:
                    if ss2[j] not in out_of_vocab:
                        out_of_vocab[ss2[j]] = len(out_of_vocab)
                    embedding[ss2[j]] = np.random.uniform(-0.25000,0.25000,300)
                else:
                    embedding[ss2[j]] = embedding_table[ss2[j]]

            for item in embedding.items():
                if item[0] not in word2index:
                    word2index[item[0]] = len(word2index)
            for j in range(len(ss1)):
                if ss1[j] in word2index:
                    ss1[j] = str(word2index[ss1[j]])
            for j in range(len(ss2)):
                if ss2[j] in word2index:
                    ss2[j] = str(word2index[ss2[j]])
            s1_test.write(' '.join(ss1) + '\n')
            s2_test.write(' '.join(ss2) + '\n')

        print(len(out_of_vocab))
        print(len(word2index))
        print(len(embedding))
        for k, v in embedding.items():
            for i in range(len(v)):
                if i == 299:
                    vector.write(str(v[i])+'\n')
                else:
                    vector.write(str(v[i])+' ')
        for item in word2index.items():
            word_index.write(item[0]+' '+str(item[1])+'\n')
        for item in out_of_vocab.items():
            out_vecab.write(item[0]+' '+str(item[1])+'\n')

def get_batch(batch_size, s, label):
    random_int = np.random.randint(0,len(s)-1,batch_size)
    batch_s = np.asarray(s)[random_int]
    batch_label = np.asarray(label)[random_int]

    return batch_label, batch_s,random_int

def read_file(s1path, s2path,s1_test,s2_test,labeltest, labelpath,re_vector):
    with open(s1path, 'r', encoding='utf-8') as f_s1,open(s2path, 'r', encoding='utf-8') as f_s2,open(labelpath, 'r', encoding='utf-8') as f_quality,open(re_vector,'r',encoding='utf-8') as f_vector,open(s1_test, 'r', encoding='utf-8') as s1test,open(s2_test, 'r', encoding='utf-8') as s2test,open(labeltest,'r',encoding='utf-8') as label_test:
        train_s = []
        test_s = []
        test_label = []
        train_label = []
        vector_lines = []
        s1 = f_s1.readlines()
        s2 = f_s2.readlines()
        test_s1 = s1test.readlines()
        test_s2 = s2test.readlines()
        for index in range(len(s1)):
            temp1 = [0]*batch_len
            line = s1[index].strip('\n').split()
            for i in range(len(line)):
                if i < batch_len:
                    temp1[i] = int(line[i])

            temp2 = [0] * (batch_len+1)
            line = s2[index].strip('\n').split()
            for i in range(len(line)):
                if i < (batch_len+1):
                    temp2[i] = int(line[i])
            for i in range(len(temp2)):
                temp1.append(temp2[i])

            train_s.append(temp1)

        for index in range(len(test_s1)):
            temp1 = [0] * batch_len
            line = test_s1[index].strip('\n').split()
            for i in range(len(line)):
                if i < batch_len:
                    temp1[i] = int(line[i])

            temp2 = [0] * (batch_len + 1)
            line = test_s2[index].strip('\n').split()
            for i in range(len(line)):
                if i < (batch_len + 1):
                    temp2[i] = int(line[i])
            for i in range(len(temp2)):
                temp1.append(temp2[i])

            test_s.append(temp1)

        for line in f_quality.readlines():
            line = line.strip('\n')
            train_label.append(int(line))

        for line in label_test.readlines():
            line = line.strip('\n')
            test_label.append(int(line))

        for line in f_vector.readlines():
            temp = []
            for vector in line.split(' '):
                temp.append(float(vector))
            vector_lines.append(temp)
        return train_s,test_s,vector_lines,train_label,test_label

def load_vector(filepath):
    with open(filepath,'r',encoding='utf-8') as glove_vector:
        embadding_table = collections.OrderedDict()
        count = 0
        for line in glove_vector.readlines():
            print(count)
            count+=1
            temp = []
            line = line.split(' ')
            for j in range(len(line)):
                if j != 0:
                        temp.append(float(line[j]))
            if line[0] not in embadding_table:
                embadding_table[line[0]] = temp
    return embadding_table

def get_char(s1_path,s2_path,s1_test,s2_test):
    with open(s1_path,'r',encoding='utf-8',errors='ignore') as f1,open(s2_path,'r',encoding='utf-8',errors='ignore') as f2,open(s1_test,'r',encoding='utf-8',errors='ignore') as f1_test,open(s2_test,'r',encoding='utf-8',errors='ignore') as f2_test,open('./char2index.txt','w+',encoding='utf-8',errors='ignore') as char2index:
        data_lines_s1 = f1.readlines()
        data_lines_s2 = f2.readlines()
        data_lines_s1_test = f1_test.readlines()
        data_lines_s2_test = f2_test.readlines()
        s1 = []
        s2 = []
        s1test = []
        s2test = []
        char_index = collections.OrderedDict()
        char_index[','] = 0
        for i in range(len(data_lines_s1)):
            ss1 = data_lines_s1[i]
            ss2 = data_lines_s2[i]
            ss1 = ss1.strip().split()
            ss2 = ss2.strip().split()
            s1_word = []
            s2_word = []
            temp = [',']*batch_len
            for j in range(len(ss1)):
                if j < batch_len:
                    temp[j] = ss1[j]
            for j in temp:
                temp_char = [0]*word_length
                word_char = []
                for k in j:
                    word_char.append(k)
                    if k not in char_index:
                        char_index[k] = len(char_index)
                for k in range(len(word_char)):
                    if k < word_length:
                        word_char[k] = char_index[word_char[k]]
                        temp_char[k] = word_char[k]
                s1_word.append(temp_char)
            s1.append(s1_word)

            temp = [','] * batch_len
            for j in range(len(ss2)):
                if j < batch_len:
                    temp[j] = ss2[j]
            for j in temp:
                temp_char = [0] * word_length
                word_char = []
                for k in j:
                    word_char.append(k)
                    if k not in char_index:
                        char_index[k] = len(char_index)
                for k in range(len(word_char)):
                    if k < word_length:
                        word_char[k] = char_index[word_char[k]]
                        temp_char[k] = word_char[k]
                s2_word.append(temp_char)
            s2.append(s2_word)

        for i in range(len(data_lines_s1_test)):
            ss1 = data_lines_s1_test[i]
            ss2 = data_lines_s2_test[i]
            ss1 = ss1.strip().split()
            ss2 = ss2.strip().split()
            s1_word = []
            s2_word = []
            temp = [',']*batch_len
            for j in range(len(ss1)):
                if j < batch_len:
                    temp[j] = ss1[j]
            for j in temp:
                temp_char = [0]*word_length
                word_char = []
                for k in j:
                    word_char.append(k)
                    if k not in char_index:
                        char_index[k] = len(char_index)
                for k in range(len(word_char)):
                    if k < word_length:
                        word_char[k] = char_index[word_char[k]]
                        temp_char[k] = word_char[k]
                s1_word.append(temp_char)
            s1test.append(s1_word)

            temp = [','] * batch_len
            for j in range(len(ss2)):
                if j < batch_len:
                    temp[j] = ss2[j]
            for j in temp:
                temp_char = [0] * word_length
                word_char = []
                for k in j:
                    word_char.append(k)
                    if k not in char_index:
                        char_index[k] = len(char_index)
                for k in range(len(word_char)):
                    if k < word_length:
                        word_char[k] = char_index[word_char[k]]
                        temp_char[k] = word_char[k]
                s2_word.append(temp_char)
            s2test.append(s2_word)

        for item in char_index.items():
            char2index.write(item[0] + ' ' + str(item[1]) + '\n')
        return s1, s2, s1test,s2test

# def get_epoch():
# s1_char_train,s1_char_test, s2_char_train,s2_char_test = get_char('./first questions.csv')
# s1_char_train,s1_char_test, s2_char_train,s2_char_test = get_char('./MSRP/s1_train.txt','./MSRP/s2_train.txt','./MSRP/s1_test.txt','./MSRP/s2_test.txt')
# get_data('./MSRP/train.txt','./MSRP/test.txt')