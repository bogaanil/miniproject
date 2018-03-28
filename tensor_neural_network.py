import numpy as np
import tensorflow as tf
import os

cwd = os.getcwd()

hog_files=os.path.join(cwd,'hog_files')
class_list=os.listdir(hog_files)

neg_features=np.zeros([1218,3360])
pos_features=np.zeros([1218,3360])

for i in range(1218):
	neg_features[i,:]=np.loadtxt(os.path.join(hog_files+"/"+class_list[0],str(i)+'.txt'))

for i in range(1218):
	pos_features[i,:]=np.loadtxt(os.path.join(hog_files+"/"+class_list[1],str(i)+'.txt'))


features_train = np.zeros([2000, 3360])
labels_train = np.zeros(2000, dtype='uint8')
features_test = np.zeros([436, 3360])
labels_test = np.zeros(436, dtype='uint8')

for i in range(1000):
	features_train[2*i,:]=neg_features[i,:]
	features_train[2*i+1,:]=pos_features[i,:]
	labels_train[2*i]=0
	labels_train[2*i+1]=1

for i in range(218):
	features_test[2*i,:] = neg_features[2000+i,:]
	features_test[2*i+1,:] = pos_features[2000+i,:]
	labels_test[2*i]=0
	labels_test[2*i+1]=1


n_nodes_hl1 = 500
n_nodes_hl2 = 10
# n_nodes_hl3=100
n_classes = 2
batch_size = 100
D = 3360
x = tf.placeholder('float', [None, D])
y = tf.placeholder('float')

hidden_1_layer = {'weights':tf.Variable(tf.random_normal([D, n_nodes_hl1])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

# hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
# 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes])),}
l1 = tf.add(tf.matmul(x,hidden_1_layer['weights']), hidden_1_layer['biases'])
l1 = tf.nn.relu(l1)
l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
l2 = tf.nn.relu(l2)
# l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
# l3 = tf.nn.relu(l3)
output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=ouput, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

hm_epochs = 10
print "give no of epochs"
hm_epoch= raw_input()
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for epoch in range(hm_epochs):
		epoch_loss = 0
		for i in range(int(2000/batch_size)):
			epoch_x = features_train[i*100:(i+1)*100,:]
			epoch_y = labels_train[i*100:(i+1)*100]
			_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y:epoch_y})
		epoch_loss += c
		print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
