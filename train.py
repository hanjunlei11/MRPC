from model_test import *
from tool import *
from config import *
import tensorflow as tf
from sklearn.metrics import *
train_s,test_s,vector_lines,train_label,test_label = read_file(s1path='./s1.txt',s2path='./s2.txt',labelpath='./label.txt',re_vector='./vector.txt',s1_test='./s1_test.txt',s2_test='./s2_test.txt',labeltest='./label_test.txt')
Model = Model(embedding_re=vector_lines)
print('1、构造模型完成')
opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Model.losses)
print('2、load data完成')
saver = tf.train.Saver(max_to_keep=3)
miss = [0]*5803
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init_op)
    # ckpt = tf.train.get_checkpoint_state('./ckpt/')
    # saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    print('3、初始化完成')
    print('4、开始训练')
    max_acc=0
    for i in range(100000):
        batch_label, batch_s, random_int = get_batch(batch_size=batch_size,s = train_s,label=train_label)
        feed_dic = {Model.input_s: batch_s, Model.batch_label: batch_label,
                    Model.embedding_keep_rate:0.8,Model.keep_rate:0.7,Model.is_traning:True}
        _,rs,loss, acc,prediction=sess.run([opt_op,merged,Model.losses,Model.acc,Model.max_index],feed_dict=feed_dic)
        writer.add_summary(rs, i)
        F1 = f1_score(batch_label, prediction)
        for j in range(batch_size):
            if prediction[j]!=batch_label[j]:
                miss[random_int[j]]+=1
        print(i+1,'次训练 ','loss: ','%.7f'%loss,'acc: ','%.7f'%acc,'max:','%.7f'%max_acc,' F1: ','%.7f'%F1)
        if (i+1)%100==0:
            all_acc = 0
            all_loss = 0
            all_F1 = 0
            for j in range(20):
                batch_label, batch_s, random_int= get_batch(batch_size=batch_size,s=test_s,label=test_label)
                feed_dic = {Model.input_s: batch_s, Model.batch_label: batch_label,
                            Model.embedding_keep_rate:1.0,Model.keep_rate:1.0,Model.is_traning:False}
                loss, acc, prediction= sess.run([Model.losses, Model.acc,Model.max_index], feed_dict=feed_dic)
                F1 = f1_score(batch_label,prediction)
                all_F1+=F1
                all_acc+=acc
                all_loss+=loss
                for k in range(batch_size):
                    if prediction[k] != batch_label[k]:
                        miss[random_int[k]+4076] += 1
                print('test losses: ','%.7f'%loss,' ','accuracy: ','%.7f'%acc,' F1: ','%.7f'%F1)
            all_F1 = all_F1/20.0
            all_acc = all_acc/20.0
            all_loss = all_loss/20.0
            if all_acc > max_acc:
                max_acc = all_acc
                saver.save(sess, save_path='ckpt/model.ckpt', global_step=i + 1)
            print('第',int((i+1)/100),'次测试 ','losses: ','%.7f'%all_loss, 'accuracy: ', '%.7f'%all_acc,' F1: ','%.7f'%all_F1)
            with open('miss.txt','w+',encoding='utf-8') as file:
                for k in range(len(miss)):
                    if k == len(miss):
                        file.write(str(k + 1) + ' ' + str(miss[k]))
                    else:
                        file.write(str(k+1)+' '+str(miss[k])+'\n')