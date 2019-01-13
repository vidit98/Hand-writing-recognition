from __future__ import division
from __future__ import print_function
import os
import sys
import tensorflow as tf
import random
import cv2
import numpy as np
from random import randint,shuffle
sys.path.append('build')
import bv

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


class Model: 
	"minimalistic TF model for HTR"

	# model constants
	batchSize = 64
	imgSize = (64, 64) # width ht


	def __init__(self, t, v, mustRestore=False):
		"init model: add CNN, RNN and CTC and initialize TF"
		
		
		self.mustRestore = mustRestore
		self.snapID = 0
		self.curr_idx = 0
		self.file1 = open(t)
		
		self.file2 = open(v)
		self.t_list = self.file1.readlines()
		self.v_list = self.file2.readlines()
		self.num_lines = len(self.t_list)
		self.num_lines_val = len(self.v_list)
		self.file1.seek(0)
		self.file2.seek(0)
		# CNN
		self.map = {}
		self.channel = 1
		self.inputImgs = tf.placeholder(tf.float32, shape=(Model.batchSize, Model.imgSize[0], Model.imgSize[1]))
		self.labels = tf.placeholder(tf.float32, shape=(None, 62))
		
		# self.inputImgs, self.labels = self.img_mapping("../data/NIST")
		# self.inputImgs, self.labels = self.minibatch_generator(self.images, self.labels, 64)
		self.logits, self.prob = self.setupCNN(self.inputImgs)
		self.loss = tf.contrib.losses.softmax_cross_entropy(self.prob,self.labels)
		
		# optimizer for NN parameters
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		# self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
		optimizer = tf.train.AdamOptimizer(learning_rate=.001)
		scope_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		print(scope_variable)
		grads_and_vars = optimizer.compute_gradients(self.loss, scope_variable)
        # if self.config.grad_clip:
        #    clipped_grads_and_vars = [(tf.clip_by_norm(item[0],self.config.clip_val),item[1]) for item in grads_and_vars] 
		self.optimizer = optimizer.apply_gradients(grads_and_vars)
		self.grad_norm = tf.global_norm([item[0] for item in grads_and_vars])


		# initialize TF
		(self.sess, self.saver) = self.setupTF()

	def reset(self):
		shuffle(self.t_list)
		shuffle(self.v_list)
		self.curr_idx = 0
		self.file1.seek(0)
		self.file2.seek(0)


			
	def setupCNN(self, cnnIn3d):
		"create CNN layers and return output of these layers"
		# cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

		cnnIn3d = tf.reshape(cnnIn3d, (64,64,64,1))
		covnet = tf.contrib.layers.conv2d(cnnIn3d, 64, 3)
		
		covnet = tf.contrib.layers.conv2d(covnet, 64, 3)

		covnet = tf.contrib.layers.max_pool2d(covnet, 2)

		
		covnet = tf.contrib.layers.conv2d(covnet, 128, 3)
		
		covnet = tf.contrib.layers.conv2d(covnet, 128, 3)

		covnet = tf.contrib.layers.max_pool2d(covnet, 2)

		
		covnet = tf.contrib.layers.conv2d(covnet, 256, 3)
		
		covnet = tf.contrib.layers.conv2d(covnet, 256, 3)
		
		# covnet = tf.contrib.layers.conv2d(covnet, 256, 3)
		
		fc = tf.reshape(covnet, (Model.batchSize, 256*16*16))

		fc = tf.contrib.layers.fully_connected(fc, 62, activation_fn=None)

		print("SHAPE1--------", fc.shape)
		prob = tf.nn.softmax(fc)
		print(prob)
		print("SHAPE2--------", fc.shape)
		char = tf.one_hot(tf.math.argmax(prob,axis = 1), 62)
		print("SHAPE--------", char.shape)
		return char, fc

	def decode(self,one_hot):
		# print("ONE HOT SHAPE", one_hot.shape)
		# print(one_hot)
		index = np.argmax(one_hot, axis=1)

		
		chars = []
		for i in range(Model.batchSize):
			if index[i] < 10:
				chars.append(str(index[i]))
			elif index[i] < 35:
				chars.append(chr(index[i] - 10 + 65))
			else:
				chars.append(chr(index[i] - 36 + 97))

		return chars
			

			 	

	


	def setupTF(self):
		"initialize TF"
		print('Python: '+ sys.version)
		print('Tensorflow: '+tf.__version__)

		sess=tf.Session() # TF session

		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = './model'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)


	def preprocess(self, im):
	    im=cv2.resize(im, (64,64))
	    im =  im.astype('float32')
	        
	    im=(im -128)/128
	    return im


	def minibatch_generator(self, f):
    	
	    line = []
	    images= np.empty((64,32,32), dtype=np.float32) 
	    img_batch =[]    
	    labels =np.zeros((64,62),  dtype=np.float32) 
	    file = []
	    if f == "train":
	    	file = self.t_list
	    else:
	    	file = self.v_list

	    for i in range(self.curr_idx, self.curr_idx + Model.batchSize):
	    	# print(file.readline())
	    	labels[i-self.curr_idx][int(file[i].split()[1])] = 1

	    images= [self.preprocess(cv2.imread(file[i].split()[0], 0)) for i in range(self.curr_idx, self.curr_idx + Model.batchSize)]
	    self.curr_idx += self.batchSize

	    # print(labels.shape, images.shape)
	    return (images,labels)   

	def hasnext(self):
		return self.num_lines- self.batchSize > self.curr_idx
	def hasnext_val(self):
		return self.num_lines_val > self.curr_idx

	def trainBatch(self, images, labels):
		"feed a batch into the NN to train it"
		rate = 0.01 
		
		(_,  lossVal,grad_norm) = self.sess.run([self.optimizer, self.loss, self.grad_norm], { self.inputImgs : images, self.learningRate : rate, self.labels : labels})
		self.batchesTrained += 1
		return grad_norm, lossVal

	def infer_image(self, image, size):
		images= np.empty((64,size,size), dtype=np.float32) 
		images = [image for i in range(64)]
		return images



	def inferBatch(self, images):
		"feed a batch into the NN to recngnize the texts"
		rate = 0.001 
		decoded, prob = self.sess.run([self.logits, self.prob], { self.inputImgs : images, self.learningRate : rate})
		for x in range(0,1):
			print(prob[x])
		return self.decode(decoded)
	

	def save(self):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
		





# model = Model("train.txt", "val.txt")
# while True:
# 	model.reset()
# 	while model.hasnext():
# 		if(model.batchesTrained%500 == 0):
# 			model.save()


# 		mini_imgs, mini_labels = model.minibatch_generator("train")
# 		# print(mini_labels)
# 		grad, loss = model.trainBatch(mini_imgs, mini_labels)
# 		print('Batch:', model.batchesTrained,'/', model.num_lines/64, 'Loss:', loss, "Grads:", grad)
# 	model.curr_idx = 0
# 	model.save()
total  = 0
t = 0

im = cv2.imread("./data/NOFORM - FREE TEXT_3-1.png", 0)
img = bv.preprocess1(im)
img1 = bv.preprocess2(img)
c = bv.preprocess3(img1)

# img2 = bv.preprocess4(b, img1, img)
cv2.imshow("img", img)
cv2.waitKey(0)
b = c.pop()
# for i in range(len(b)):
# 	crop_im = img[b[i][1]:b[i][1] + b[i][3], b[i][0]:b[i][0] + b[i][2]]
# 	cv2.imshow("da", crop_im);
# 	_, crop_im= cv2.threshold(crop_im, 129, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU )
# 	# crop_im = cv2.adaptiveThreshold(crop_im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,1)
# 	cv2.imshow("da", crop_im);
# 	cv2.waitKey(0);
print("Np of lnes", len(c))
for k in range(len(c)):
	b = c[k]
	wd_e = 0
	avg = [b[i][2] for i in range(len(b))]
	avg_w = sum(avg)/len(avg)
	i = 0
	while i < len(b):
		rect = b[i]
		if i != len(b)-1:
			next_rect = b[i+1]

		if i == len(b)-1:
			crop_im = img[rect[1]-3:rect[1] + rect[3] + 3, rect[0] - wd_e:rect[0] + rect[2] + wd_e]

		elif i == 0:
			wd_e = min(2, abs(rect[0]+ rect[2] - next_rect[0]))
			crop_im = img[rect[1]-3:rect[1] + rect[3] + 3, rect[0] - wd_e:rect[0] + rect[2] + wd_e]#y,x
		else:
			wd_n = min(2, abs(rect[0]+ rect[2] - next_rect[0]))
			crop_im = img[rect[1]-3:rect[1] + rect[3] + 3, rect[0] - wd_e:rect[0]+ wd_e+ rect[2] + wd_n]
			wd_e = wd_n

		print("DIMS", rect[2], avg_w)
		if(int(rect[2]/avg_w) > 1):
			div = int(rect[2]/avg_w)
			split_img = [img[rect[1]-3:rect[1] + rect[3] + 3, rect[0] + i:rect[0] +i + int(rect[2]/div)] for i in range(0, rect[2], int(rect[2]/div))]
			for a in range(div):
				cv2.imshow("Aadfs", split_img[a] )
				cv2.waitKey(0)
			crop_im = split_img[0]
		_,im = cv2.threshold(crop_im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		im = cv2.resize(im, (50, int(50*(im.shape[0]/im.shape[1]))))
		ht = min(50, im.shape[1])
		wd = min(50, im.shape[0])

		ht = int((128 - ht)/2)
		wd = int((128 - wd)/2)

		

		im = cv2.copyMakeBorder(im, top=ht, bottom=ht, left=wd, right=wd, borderType= cv2.BORDER_CONSTANT, value=[255]);


		cv2.imshow("as",im)
		# im = model.preprocess(im)
		
		# mini_imgs = model.infer_image(im, 64)
		# m = model.inferBatch(mini_imgs)
		# print(m[1])
		cv2.waitKey(0)
		i = i+ 1
	
