# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import shutil

'''
딥러닝
@since 2019.10.20
'''

# 신경망 파라미터
learning_rate = 0.001
training_epochs = 10000
answer = []

os.chdir("C:/Project/AI/mayml/database/")
train_loc = "./train-data/0f62c"

print(train_loc)
# 트레이닝 데이터
data = np.loadtxt(train_loc, delimiter=',', dtype=str).astype(np.float32)
input = data[:, :-1]
output = data[:, -1].reshape([len(data), 1])
features = input.shape[1]

# 테스트 데이터
# test_loc = "./test-data/0f62c"
# print(test_loc)
# test_data = np.loadtxt(test_loc, delimiter=',', dtype=str).astype(np.float32)
# test_data = test_data.reshape([-1, features])

print(input.shape, output.shape)

# 트레이닝 placeholders 지정
X = tf.placeholder(tf.float32, shape=[None, features], name='x')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

# 히든레이어의 노드 수
num_hidden1 = 20
num_hidden2 = 20

# drop-out 몇퍼센트 할 건지 받을 변수
keep_prob = tf.placeholder(tf.float32)

# layer1
W1 = tf.Variable(tf.random_normal([features, num_hidden1], dtype='float'), name='weight1')
b1 = tf.Variable(tf.random_normal([num_hidden1], dtype='float'), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

# drop out 사용여부
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# layer2
W2 = tf.Variable(tf.random_normal([num_hidden1, num_hidden2], dtype='float'), name='weight2')
b2 = tf.Variable(tf.random_normal([num_hidden2], dtype='float'), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

# drop out 사용여부
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# layer3
W3 = tf.Variable(tf.random_normal([num_hidden2, 1], dtype='float'), name='weight3')
b3 = tf.Variable(tf.random_normal([1], dtype='float'), name='bias3')
hypothesis = tf.matmul(L2, W3) + b3

# Construct a linear model
h = tf.identity(hypothesis, name='h')

# cost를 Mean squared error로 지정
cost = tf.reduce_mean(tf.square(hypothesis - Y), name='cost')

# 정확도 측정
acc = tf.equal(tf.round(hypothesis), Y)
acc = tf.reduce_mean(tf.cast(acc, tf.float32), name='accuracy')

# learning_rate 값을 Gradient Descent 옵티마이저에 적용
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

# 세션 생성
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습 시작
# write result
tempStr = "";
for step in range(training_epochs):
    # sess.run(train, feed_dict={X: input, Y: output})
    # drop out 쓰려면 아래 주석을 푸시면 됩니다.
    sess.run(train, feed_dict={X: input, Y: output, keep_prob: 0.8})
    cost_val, hy_val = sess.run([cost, hypothesis], feed_dict={X: input, Y: output, keep_prob: 1.0})
    if (step % 100 == 0):
        # cost_val, hy_val = sess.run([cost, hypothesis], feed_dict={X: input, Y: output, keep_prob: 1.0})
        if step == 0:
            print("[", step + 1, "] Cost:", cost_val)
        else:
            print("[", step, "] Cost:", cost_val)
    tempStr += str(cost_val) + "\n"

f = open("C:/Project/AI/mayml/database/test-result/test1.txt", 'w')
f.write(tempStr)
f.close()

print("=============TRAIN END==============")
cost_val, acc_val = sess.run([cost, acc], feed_dict={X: input, Y: output, keep_prob: 1.0})
print(" Cost:", cost)
print(" Acc:", acc_val)

# write result
f = open("C:/Project/AI/mayml/database/test-result/test1.txt", 'a')
tempStr = "\n" + 'sincerity,' + str(acc_val)
f.write(tempStr)
f.close()

print("=============PREDICT==============")
# for i in range(test_data.shape[0]):
# pre = sess.run(hypothesis, feed_dict={X: test_data[i].reshape(1,features)})
# drop out 쓰려면 아래 주석을 푼다
# pre = sess.run(hypothesis, feed_dict={X: test_data[i].reshape(1,features), keep_prob: 1.0})
# print(test_data[i],"==> ", pre[0])

model_path = "C:/Project/AI/mayml/model/java/0f62c"
if (os.path.exists(model_path)):
    shutil.rmtree(model_path)
builder = tf.saved_model.builder.SavedModelBuilder(model_path)
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save(True)

saver = tf.train.Saver()
save_path = saver.save(sess, "C:/Project/AI/mayml/model/python/0f62c/model.chpk")
