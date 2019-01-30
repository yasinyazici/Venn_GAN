# omniglot will not work if the path is not changed.

import argparse

import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from lib.utils import *
from itertools import combinations
from glob import glob
from random import shuffle
import fid
import pickle
from copy import copy

ds = tf.contrib.distributions

parser = argparse.ArgumentParser('')
parser.add_argument('--batch_size', '-b',type=int, default=64)
parser.add_argument('--resolution',type=int, default=28)
parser.add_argument('--dataset', '-d', type=str, default='cifar10') #'mnist', 'mnist_fashion','cifar10'
parser.add_argument('--model', '-m', type=str, default='dcgan') #dcgan, resnet
parser.add_argument('--max_iter', type=int, default=50000)
parser.add_argument('--decay', type=float, default=0.999)
#parser.add_argument('--sn', type=bool, default=False)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', default='classifier', help='Directory to output the result')
parser.add_argument('--viz_size', type=int, default=128, help='number of images to display')
parser.add_argument('--cal_every', type=int, default=2000, help='Interval of evaluation')
parser.add_argument('--viz_every', type=int, default=1000,help='Interval of display')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

params = dict(
    learning_rate=0.0002,
    beta1=0.5,
    beta2=0.9,
)

sn = False
bn = False
dir_name = os.path.dirname(__file__)
log_dir = os.path.abspath('{}/logs/{}_{}_{}'.format(dir_name,args.out, args.dataset, args.model))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
path = os.path.normpath(dir_name)
x = path.split(os.sep)
if 'users' in x:
    data_dir = '/home/users/***/***/scratch/data/'
elif '***' in x:
    data_dir = '/home/***/Documents/data/'
else:
    raise NotImplementedError("not implemented")

if args.dataset == 'mnist':
    from keras.datasets import mnist
    from models.model_mnist import generator,discriminator
    (trainx, trainy), (testx, testy) = mnist.load_data()
    trainx = np.reshape(trainx.astype('float32'),[-1,28,28,1])/127.5 - 1
    testx = np.reshape(testx.astype('float32'),[-1,28,28,1])/127.5 - 1
    
elif args.dataset == 'mnist_fashion':
    from keras.datasets import fashion_mnist
    from models.model_mnist import generator,discriminator
    (trainx, trainy), (testx, testy) = fashion_mnist.load_data()
    trainx = np.reshape(trainx.astype('float32'),[-1,28,28,1])/127.5 - 1
    testx = np.reshape(testx.astype('float32'),[-1,28,28,1])/127.5 - 1
    
elif args.dataset == 'cifar10':
    from keras.datasets import cifar10
    if args.model == 'dcgan':
        from models.model_32_dcgan import generator,discriminator
    elif args.model == 'resnet':
        from models.model_32_resnet import generator,discriminator
    (trainx, trainy), (testx, testy) = cifar10.load_data()
    trainx = trainx.astype('float32')/127.5 - 1
    testx = testx[:1000].astype('float32')/127.5 - 1
    trainy = np.squeeze(trainy)
    testy = np.squeeze(testy[:1000])
    
elif args.dataset == 'omniglot' and args.resolution==28:
    from lib.utils import read_omniglot
    #from models.model_32_dcgan import generator,discriminator
    from models.model_mnist import generator,discriminator
    if args.resolution == 28:
        convert = 'grayscale'
    else:
        convert = 'RGB'
    trainx, trainy = read_omniglot(data_dir + 'omniglot-master/python/images_background',args.resolution,convert)
    if args.resolution == 28:
        trainx = np.expand_dims(trainx,axis=3)/127.5 - 1
    else:
        trainx = trainx/127.5 - 1
        
elif args.dataset == 'comnist' and args.resolution==28:
    from lib.utils import read_comnist
    #from models.model_32_dcgan import generator,discriminator
    from models.model_mnist import generator,discriminator
    if args.resolution == 28:
        convert = 'grayscale'
    else:
        convert = 'RGB'
    trainx, trainy = read_comnist(data_dir + 'CoMNIST-master/images',args.resolution,convert)
    if args.resolution == 28:
        trainx = np.expand_dims(trainx,axis=3)/127.5 - 1
    else:
        trainx = trainx/127.5 - 1 

trainy = trainy.astype('int32')
testy = testy.astype('int32')

tf.reset_default_graph()
    
dataset = tf.data.Dataset.from_tensor_slices((trainx,trainy)).shuffle(10000).repeat()
iterator = dataset.batch(args.batch_size).prefetch(100).make_one_shot_iterator()
sample, label = iterator.get_next()

c      = discriminator(sample, num_agent=10, sn=sn, name = 'classifier')
c_test = discriminator(testx , num_agent=10, sn=sn, name = 'ema_classifier')

#loss functions
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=c, labels=label))
accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(c, 1, output_type=tf.int32), label), tf.float32))

# Evaluate model
correct_pred = tf.equal(tf.argmax(c_test, 1), testy)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
c_vars = tf.global_variables(scope="classifier")
c_ema_vars = tf.global_variables(scope="ema_classifier")

#optimizers 
if True:#args.optimizer == 'adam':
    train_opt = tf.train.AdamOptimizer(params['learning_rate'],params['beta1'],params['beta2'])
    train_op = train_opt.minimize(loss, var_list=c_vars)
else:
    raise NotImplementedError

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

with tf.control_dependencies([train_op]):
    ema_op = exponential_moving_average(c_ema_vars, c_vars, args.decay)
    
sess.run(copy_params(c_ema_vars,c_vars))
    
for i in tqdm(range(args.max_iter+1)):
    f, acc_tr, _, _ = sess.run([loss, accuracy_train, train_op, ema_op])             
        
    if ((i) % args.viz_every == 0):
        print([f, acc_tr])
    if ((i) % args.cal_every == 0) and (i !=0):
        acc = sess.run(accuracy) 
        print(acc)
   
temp1 = sess.run(c_ema_vars) 
temp2 = sess.run(c_vars) 
path = os.path.normpath(dir_name)
x = path.split(os.sep)
if 'users' in x:
    np.save('/home/users/***/***/project/cooperation-game-master/logs/c_ema_vars_{}_{}'.format(args.model,args.dataset),np.asarray(temp1))
    np.save('/home/users/***/***/project/cooperation-game-master/logs/c_vars_{}_{}'.format(args.model,args.dataset),np.asarray(temp2))
elif '***' in x:
    np.save('/home/***/Documents/python/cooperation-game-master/logs/c_ema_vars_{}_{}'.format(args.model,args.dataset),np.asarray(temp1))
    np.save('/home/***/Documents/python/cooperation-game-master/logs/c_vars_{}_{}'.format(args.model,args.dataset),np.asarray(temp2))
else:
    raise NotImplementedError("not implemented")

