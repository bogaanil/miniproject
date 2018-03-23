import numpy as np
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
	features_test[2*i,:] = neg_features[1000+i,:]
	features_test[2*i+1,:] = pos_features[1000+i,:]
	labels_test[2*i]=0
	labels_test[2*i+1]=1

D=3360
K=2
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1
reg = 1e-3 # regularization strength

print "give no of epochs"
epoch= raw_input()
# gradient descent loop
num_examples = 1000*2
for i in xrange(int(epoch)):
	loss = 0
	for batch in range(20):
		X = features_train[batch*10:(batch+1)*10,:]
		num_examples= X.shape[0]
		y = labels_train[batch*10:(batch+1)*10]
		# evaluate class scores, [N x K]
		scores = np.dot(X, W) + b 

		# compute the class probabilities
		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

		# compute the loss: average cross-entropy loss and regularization
		corect_logprobs = -np.log(probs[range(num_examples),y])
		data_loss = np.sum(corect_logprobs)/num_examples
		reg_loss = 0.5*reg*np.sum(W*W)
		loss += data_loss + reg_loss
	
		

		# compute the gradient on scores
		dscores = probs
		dscores[range(num_examples),y] -= 1
		dscores /= num_examples

		# backpropate the gradient to the parameters (W,b)
		dW = np.dot(X.T, dscores)
		db = np.sum(dscores, axis=0, keepdims=True)

		dW += reg*W # regularization gradient

		# perform a parameter update
		W += -step_size * dW
		b += -step_size * db
	print "iteration %d: loss %f" % (i, loss)

scores = np.dot(features_test, W) + b
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == labels_test))
