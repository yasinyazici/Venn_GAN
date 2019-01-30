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
import collections 
import scipy.special
from glob import glob
import fid

ds = tf.contrib.distributions

parser = argparse.ArgumentParser('')
parser.add_argument('--batch_size', '-b',type=int, default=16)
parser.add_argument('--resolution',type=int, default=28)
parser.add_argument('--num_agent', '-n',type=int, default=2)
parser.add_argument('--mode_multiplier',type=int, default=1) # number of modes per region
parser.add_argument('--imbalance',type=str, default='balanced') # 'balanced', 'type1', 'type2'
parser.add_argument('--dataset', '-d', type=str, default='omniglot_CL') #'mnist', 'mnist_fashion','cifar10', 'omniglot', 'omniglot_CG', 'omniglot_CL', 'omniglot_GL'
parser.add_argument('--distribution', type=str, default='small') #'small', 'big'
parser.add_argument('--gen_type', type=str, default='conditional') #conditional, independent
parser.add_argument('--objective', '-o', type=str, default='gan') #gan, hinge, wgan-gp
parser.add_argument('--model', '-m', type=str, default='dcgan') #dcgan, resnet
parser.add_argument('--reg', '-r', type=str, default='real') #'None','real', 'fake'
parser.add_argument('--classify','-c', type=bool, default=True)
parser.add_argument('--connection_map', type=str, default='v1') #'v1','v3'
parser.add_argument('--z_dim', '-z', type=int, default=128)
parser.add_argument('--h1', type=float, default=1.0)
parser.add_argument('--h2', type=float, default=1.0)
parser.add_argument('--h3', type=float, default=1.0)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--class_scale', type=float, default=0.1)
parser.add_argument('--n_dis', type=int, default=1, help='number of discriminator update per generator update')
parser.add_argument('--max_iter', type=int, default=20000)
parser.add_argument('--decay', type=float, default=0.999)
#parser.add_argument('--sn', type=bool, default=False)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', default='venn', help='Directory to output the result')
parser.add_argument('--viz_size', type=int, default=36, help='number of images to display')
parser.add_argument('--cal_every', type=int, default=1000, help='Interval of evaluation')
parser.add_argument('--viz_every', type=int, default=1000,help='Interval of display')
parser.add_argument('--out_quality', type=str, default='normal',help='quality of display') #"normal", 'high'
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

params = dict(
    disc_learning_rate=0.0002,
    gen_learning_rate=0.0002,
    beta1=0.0,
    beta2=0.9,
)

# hyperparam search grid of 2 list
# [0.0, 0.125, 0.25, 0.5, 1.0]

# a is propostion for each mode
if args.num_agent == 2:
    alpha = [args.h1, args.h1, args.h2]
    batch_size_sl = args.h1 + args.h2
elif args.num_agent == 3:
    alpha = [args.h1, args.h1, args.h1, args.h2, args.h2, args.h2, args.h3]
    batch_size_sl = args.h1 + args.h2 + args.h2 + args.h3
  
num_region = np.power(2,args.num_agent) -1
num_modes = num_region * args.mode_multiplier
batch_size_z = args.batch_size / args.mode_multiplier
bs_list = [int(a*batch_size_z) for a in alpha]
batch_size_d = int(batch_size_sl * batch_size_z)

sn_d = True
sn_g = True
bn = True
if args.reg in ['real', 'fake']:
    bn = False
    sn_d = False
    sn_g = False

# Paths
#==============================================================================
dir_name = os.path.dirname(__file__)
log_dir = os.path.abspath('{}/logs/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(dir_name,args.out, args.dataset, 
                          args.connection_map, str(args.num_agent), args.distribution, args.gen_type, 
                          args.classify, args.model, args.reg, str(args.imbalance)))
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

# Loading datasets and importing appropriate networks
#============================================================================== 
if args.dataset == 'mnist':
    from keras.datasets import mnist
    from models.model_mnist import generator,discriminator
    (trainx, trainy), (testx, testy) = mnist.load_data()
    trainx = np.reshape(trainx,[-1,28,28,1])/127.5 - 1
    testx = np.reshape(testx,[-1,28,28,1])/127.5 - 1
    
elif args.dataset == 'mnist_fashion':
    from keras.datasets import fashion_mnist
    from models.model_mnist import generator,discriminator
    (trainx, trainy), (testx, testy) = fashion_mnist.load_data()
    trainx = np.reshape(trainx,[-1,28,28,1])/127.5 - 1
    testx = np.reshape(testx,[-1,28,28,1])/127.5 - 1
    
elif args.dataset == 'cifar10':
    from keras.datasets import cifar10
    if args.model == 'dcgan':
        from models.model_32_dcgan import generator,discriminator
    elif args.model == 'resnet':
        from models.model_32_resnet import generator,discriminator
    (trainx, trainy), (testx, testy) = cifar10.load_data()
    trainx = trainx/127.5 - 1
    testx = testx/127.5 - 1
    
elif (args.dataset in ['omniglot','omniglot_CG','omniglot_CL','omniglot_GL']) and args.resolution==28:
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
        
elif args.dataset == 'celeba':# and args.resolution==64:
    if args.model == 'dcgan':
        from models.model_64_dcgan import generator,discriminator
    elif args.model == 'resnet':
        from models.model_64_resnet import generator,discriminator
    basename_data = ['celebA_males_64', 'celebA_females_64']
    
elif args.dataset == 'cifar10and100':
    from keras.datasets import cifar10, cifar100
    if args.model == 'dcgan':
        from models.model_32_dcgan import generator,discriminator
    elif args.model == 'resnet':
        from models.model_32_resnet import generator,discriminator
    (trainx, trainy), (testx, testy) = cifar10.load_data()
    trainx_10 = trainx[:1000]/127.5 - 1
    trainy_10 = np.ones((trainx_10.shape[0],1),'int32')
    (trainx, trainy), (testx, testy) = cifar100.load_data()
    trainx_100 = trainx[:50000]/127.5 - 1
    trainy_100 = np.zeros((trainx_100.shape[0],1),'int32')
    trainx = np.concatenate([trainx_100,trainx_10],axis=0)
    trainy = np.concatenate([trainy_100,trainy_10],axis=0)
    del trainx_100, trainx_10, trainy_100, trainy_10
    basename_stats = ['fid_stats_cifar100_tf_train.npz', 'fid_stats_cifar10_tf_train.npz']
    dist_stats_path = []
    dist_stats_path.append(data_dir + basename_stats[0])
    dist_stats_path.append(data_dir + basename_stats[1])
    
elif args.dataset == 'stanford_kaggle':# and args.resolution==64:
    if args.model == 'dcgan':
        from models.model_64_dcgan import generator,discriminator
    elif args.model == 'resnet':
        from models.model_64_resnet import generator,discriminator
    basename_data = ['stanford_dogs_64', 'dogs_cats_64_kaggle']
    basename_stats = ['fid_stats_stanford_dogs_64_tf.npz', 'fid_stats_dogs_cats_64_kaggle_tf.npz']
    dist_stats_path = []
    dist_stats_path.append(data_dir + basename_stats[0])
    dist_stats_path.append(data_dir + basename_stats[1])
    #inception_path = fid.check_or_download_inception(data_dir + 'imagenet_model')
    
fid_score_calc = False
if args.dataset in ['stanford_kaggle','cifar10and100']:
    fid_score_calc = True
    inception_path = data_dir + 'imagenet_model/classify_image_graph_def.pb'
    #dist_stats_path = data_dir + basename_stats
    
# connetion map
#==============================================================================
if args.connection_map == 'v1' and args.num_agent == 2:
    connection_map = [[0,2],
                      [1,2]]
elif args.connection_map == 'v1' and  args.num_agent == 3:
    connection_map = [[0,3,5,6],
                      [1,4,5,6],
                      [2,3,4,6]]
elif args.connection_map == 'v3' and args.num_agent == 2:
    connection_map = [[0,2],
                      [  2]]
elif args.connection_map == 'v3' and args.num_agent == 3:
    connection_map = [[0,5,6],
                      [  5,6],
                      [    6]]
else:
    raise NotImplementedError
    
relevant_idx = list(set([item for sublist in connection_map for item in sublist]))

# gen custom ditstibutions
#==============================================================================+
if args.dataset =='omniglot':
    dist_list = [[0],
                 [1],
                 [2]]
elif args.dataset == 'omniglot_CG':
    dist_list = [[0],
                 [1]]
elif args.dataset == 'omniglot_CL':
    dist_list = [[0],
                 [2]]
elif args.dataset == 'omniglot_GL':
    dist_list = [[1],
                 [2]]
elif args.dataset in ['comnist','cifar10and100']:
    dist_list = [[0],
                 [1]]
elif args.distribution =='big' and args.num_agent == 3 and args.connection_map == 'v1':
    dist_list = []
elif args.distribution =='small' and args.num_agent == 3 and args.connection_map == 'v1':
    dist_list = [[0,3,5,6],
                 [1,4,5,6],
                 [2,3,4,6]]
    cc = [[0],[1],[2],[3],[4],[5],[6]]
elif args.distribution =='small_no' and args.num_agent == 3 and args.connection_map == 'v1':
    dist_list = [[0,3,5],
                 [1,4,7],
                 [2,8,9]]
elif args.distribution =='small' and args.num_agent == 2 and args.connection_map == 'v1':
    dist_list = [[0,2],
                 [1,2]]
    cc = [[0],[1],[2]]
elif args.distribution =='big' and args.num_agent == 2 and args.connection_map == 'v1':
    dist_list = [[0,1,2,3,4,5,6],
                 [3,4,5,6,7,8,9]]
    cc = [[0,1,2],[7,8,9],[3,4,5,6]]
elif args.distribution =='small' and args.num_agent == 2 and args.connection_map == 'v3': 
    dist_list = [[0,1],
                 [  1]]
elif args.distribution =='big' and args.num_agent == 2 and args.connection_map == 'v3': 
    dist_list = [[0,1,2,3,4,5,6,7,8,9],
                 [          5,6,7,8,9]]
elif args.distribution =='small' and args.num_agent == 3 and args.connection_map == 'v3': 
    dist_list = [[0,1,2],
                 [  1,2],
                 [    2]]
    cc = [[0],[1],[2]]
elif args.distribution =='big' and args.num_agent == 3 and args.connection_map == 'v3': 
    dist_list = [[0,1,2,3,4,5,6,7,8,9],
                 [      3,4,5,6,7,8,9],
                 [            6,7,8,9]] 
    cc = [[0,1,2],[3,4,5],[6,7,8,9]]
else:
    raise NotImplementedError

# not a generic code, aims 3 agents games 
#==============================================================================    
if args.connection_map == 'v1' and args.num_agent == 3 and args.dataset in ['mnist', 'mnist_fashion','cifar10']:
    trainx_list = []
    trainy_list = []  
    if args.imbalance == 'type1':
        b_list = [500,2000,6000] # ==>[500,1000,2000]
    elif args.imbalance == 'type2':
        b_list = [6000,6000,6000] # ==>[6000,3000,2000]
    else:
        b_list = [2000,4000,6000] # ==>[2000,2000,2000]
    
    for digit in list(set([item for sublist in dist_list for item in sublist])):
        if digit in [0,1,2]:
            trainx_list.append(trainx[np.squeeze(trainy==digit)][:b_list[0]])
            trainy_list.append(trainy[np.squeeze(trainy==digit)][:b_list[0]])
        elif digit in [3,4,5]:
            trainx_list.append(trainx[np.squeeze(trainy==digit)][:b_list[1]])
            trainy_list.append(trainy[np.squeeze(trainy==digit)][:b_list[1]])
        else:
            trainx_list.append(trainx[np.squeeze(trainy==digit)][:b_list[2]])
            trainy_list.append(trainy[np.squeeze(trainy==digit)][:b_list[2]])
    trainx = np.concatenate(trainx_list,axis=0)
    trainy = np.concatenate(trainy_list,axis=0)
 
#==============================================================================
count_overlap = collections.defaultdict(int) # hashmap to count overlaps
count_occurence = collections.defaultdict(int) # hasmap to update where to select subset of current digit

# constructing hashmap digit => occurence
for set_ in dist_list: 
    for digit in set_:
        count_overlap[digit] += 1
            
distrib=[]   
if args.dataset in ['celeba','stanford_kaggle']:
    for i in range(args.num_agent):
        distrib.append(glob(data_dir+ basename_data[i] +'/*.jpg'))
        
elif args.dataset in ['mnist', 'mnist_fashion','cifar10']:
    for i in range(args.num_agent):
        l = []        
        for digit in dist_list[i]:
            n_digit = np.sum(trainy==digit) # count elements

            start=count_occurence[digit]*n_digit//count_overlap[digit]
            end =(count_occurence[digit]+1)*n_digit//count_overlap[digit]

            count_occurence[digit] += 1 # 

            l.append(trainx[np.squeeze(trainy==digit)][start:end])
        trainx1=np.vstack(l)
        distrib.append(trainx1[np.random.permutation(trainx1.shape[0])])

else: # for the datasets which are not contrived
    for i in range(args.num_agent):
        l = []
        for digit in dist_list[i]:
            l.append(trainx[np.squeeze(trainy==digit)])
        trainx1=np.vstack(l)
        distrib.append(trainx1[np.random.permutation(trainx1.shape[0])])

# visualize datasets 
#==============================================================================
if args.dataset not in ['celeba','stanford_kaggle']:
    f, axarr = plt.subplots(1, args.num_agent, figsize=(12, 4))
    for i,ax in enumerate(axarr):
        X = grid_x(distrib[i][:args.viz_size])
        ax.imshow(X)
    plt.show()

def disc_loss(real,fake):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real))) +\
           tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
def gen_loss(fake):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
def class_loss(logits,labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

tf.reset_default_graph()

data_samples=[]
list_iter=[]
if args.dataset not in ['celeba','stanford_kaggle']:
    sh = distrib[0].shape
    features_placeholder = tf.placeholder(distrib[0].dtype, (None, sh[1], sh[2], sh[3]),name='features_pl')
else:
    features_placeholder = tf.placeholder(tf.string, name='features_pl')
    
for i in range(args.num_agent):
    dataset = tf.data.Dataset.from_tensor_slices(features_placeholder).shuffle(10000).repeat()
    if args.dataset in ['celeba','stanford_kaggle']: dataset = dataset.map(parse_function)
    iterator = dataset.batch(batch_size_d).prefetch(100).make_initializable_iterator()
    data_samples.append(tf.cast(iterator.get_next(),tf.float32))
    list_iter.append(iterator)

'''
data_samples = []
for i in range(args.num_agent):
    dataset = tf.data.Dataset.from_tensor_slices(distrib[i]).shuffle(10000).repeat()
    if args.dataset in ['celeba']: dataset = dataset.map(parse_function)
    iterator_unl = dataset.batch(batch_size_d).prefetch(100).make_one_shot_iterator()
    sample = tf.cast(iterator_unl.get_next(),tf.float32)
    data_samples.append(sample) 
'''

def sample_z(bs_list,same_input=False):
    if same_input:
        noise = np.random.normal(size=(bs_list[0],args.z_dim)).astype(dtype='float32')
        noise = np.vstack([noise] * len(bs_list))
    else:
        noise = np.random.normal(size=(sum(bs_list),args.z_dim)).astype(dtype='float32')
        
    modes = np.concatenate([idx*np.ones(bs, dtype='int32') for idx, bs in enumerate(bs_list)],axis=0)
    return noise, modes

is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
noise_ph = tf.placeholder(tf.float32, (None, args.z_dim), name='noise_ph')
modes_ph = tf.placeholder(tf.int32, (None,), name='modes_ph')

#generator
#gen_out = generator(noise_ph, mode=modes_ph, num_modes=num_modes, sn=sn_g, bn=bn, is_training=is_training_pl, name="gen")
#gen_list = tf.split(gen_out, bs_list, 0)
#gen_ema = generator(noise_ph, mode=modes_ph, num_modes=num_modes, sn=sn_g, bn=bn, is_training=False, name="ema_gen")
#gen_ema = tf.split(gen_ema, bs_list, 0)

#generator
if args.gen_type == 'conditional':
    gen_out = generator(noise_ph, mode=modes_ph, num_modes=num_modes, sn=sn_g, bn=bn, is_training=is_training_pl, name="gen")
    gen_list = tf.split(gen_out, bs_list, 0)
    gen_ema = generator(noise_ph, mode=modes_ph, num_modes=num_modes, sn=sn_g, bn=bn, is_training=False, name="ema_gen")
elif args.gen_type == 'independent':
    gen_list = []
    noise_list = tf.split(noise_ph,num_modes,axis=0)
    for i in range(num_modes):
        gen_list.append(generator(noise_list[i], sn=sn_g, bn=bn, is_training=is_training_pl, name="gen_{}".format(i)))  
    gen_ema= []
    for i in range(num_modes):
        gen_ema.append(generator(noise_list[i], sn=sn_g, bn=bn, is_training=False, name="ema_gen_{}".format(i))) 
    gen_ema = tf.concat(gen_ema,axis=0)
    

#standard discriminators
dis_list=[[0]*2 for j in range(args.num_agent)]
gen_samples = []
for i in range(args.num_agent):
    dis_list[i][0] = discriminator(data_samples[i], sn=sn_d, name = 'dis_'+str(i))
    gen_samples.append(tf.concat([gen_list[region] for region in connection_map[i]],axis=0))
    dis_list[i][1] = discriminator(gen_samples[i], sn=sn_d, reuse=True, name = 'dis_'+str(i))
    
#loss functions
if args.objective == 'gan':
    d_loss = [disc_loss(*dis) for dis in dis_list]
    g_loss = [gen_loss(dis[-1]) for dis in dis_list]    
else:
    raise NotImplementedError
    
#classifier and its objective
if args.classify:
    input_class = tf.concat([gen_list[indx] for indx in relevant_idx], axis=0)
    c = discriminator(input_class, len(relevant_idx), sn=sn_d, name = 'classifier')
    labels = tf.concat([idx*tf.ones(bs, dtype='int32') for idx, bs in enumerate([bs_list[indx] for indx in relevant_idx])],axis=0)
    d_loss.append(args.class_scale*class_loss(logits=c, labels=labels))
    g_loss.append(args.class_scale*class_loss(logits=c, labels=labels))

if args.reg == 'real':
    for i in range(args.num_agent):
        ddx = tf.gradients(dis_list[i][0], data_samples[i])[0]
        print(ddx.get_shape().as_list())
        d_loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(ddx),axis=[1,2,3])) * args.scale)
elif args.reg == 'fake':
    for i in range(args.num_agent):
        ddx = tf.gradients(dis_list[i][1], gen_samples[i])[0]
        print(ddx.get_shape().as_list())
        d_loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(ddx),axis=[1,2,3])) * args.scale)

g_vars = tf.global_variables(scope="gen")
g_ema_vars = tf.global_variables(scope="ema_gen")
d_vars = tf.global_variables(scope="dis")  
if args.classify:
    d_vars += tf.global_variables(scope="classifier")  
#optimizers 
if True:#args.optimizer == 'adam':
    g_train_opt = tf.train.AdamOptimizer(params['gen_learning_rate'],params['beta1'],params['beta2'])
    d_train_opt = tf.train.AdamOptimizer(params['disc_learning_rate'],params['beta1'],params['beta2'])
    d_train_op = d_train_opt.minimize(np.sum(d_loss), var_list=d_vars)
    g_train_op = g_train_opt.minimize(np.sum(g_loss), var_list=g_vars)
else:
    raise NotImplementedError

with tf.control_dependencies([d_train_op,g_train_op]):
    ema_op = exponential_moving_average(g_ema_vars,g_vars,args.decay)

sess = tf.InteractiveSession()
[sess.run(list_iter[i].initializer, feed_dict={features_placeholder: distrib[i]}) for i in range(args.num_agent)]
sess.run(tf.global_variables_initializer())
sess.run(copy_params(g_ema_vars,g_vars))

eval_score = False
if args.dataset in ['mnist','mnist_fashion','cifar10']:
    eval_score = True
    pt_vars = np.load('/home/***/Documents/python/cooperation-game-master/logs/c_ema_vars_{}_{}.npy'
                  .format(args.model,args.dataset))
    _,h,w,c = np.shape(trainx)
    sample = tf.placeholder(tf.float32, (None, h, w, c), name='sample')
    qs = tf.argmax(discriminator(sample, num_agent=10, sn=False, name = 'q_score'), 1)
    q_vars = tf.global_variables(scope="q_score")
    sess.run(copy_params(q_vars,pt_vars))


if args.num_agent == 2:
    labels = ['S_1 \\backslash S_2','S_2 \\backslash S_1','S_1 \cap S_2']
elif args.num_agent == 3:
    labels = ['S_1 \\backslash (S_2 \cup S_3)','S_2 \\backslash (S_1 \cup S_3)','S_3 \\backslash (S_1 \cup S_2)',
              '(S_1 \cap S_3) \\backslash S_2','(S_2 \cap S_3) \\backslash S_1','(S_1 \cap S_2) \\backslash S_3',
              'S_1 \cap S_2 \cap S_3']
else:
    raise NotImplementedError
    

if fid_score_calc:
    mu_real = []
    sigma_real = [] 
    FID_list = []
    fid.create_inception_graph(inception_path)# load the graph into the current TF graph
    for i in range(len(dist_stats_path)):
        f = np.load(dist_stats_path[i])
        mu_real.append(f['mu'][:])
        sigma_real.append(f['sigma'][:])
        f.close()
  
fs = []  
score_list = [[] for _ in range(len(relevant_idx))] 
n_fix,m_fix = sample_z([args.viz_size]*num_region, same_input=True)
for i in tqdm(range(args.max_iter+1),disable=False):
    for _ in range(args.n_dis):
        #f, _ = sess.run([d_loss, d_train_op,],{is_training_pl:True})    
        n,m = sample_z([args.batch_size]*num_region, same_input=True)
        f, _ = sess.run([d_loss, d_train_op],{is_training_pl:True, noise_ph:n, modes_ph:m})   
    n,m = sample_z([args.batch_size]*num_region, same_input=True)
    g, _, _ = sess.run([g_loss, g_train_op,ema_op],{is_training_pl:True, noise_ph:n, modes_ph:m})
    
    if ((i) % 500 == 0):
        fs.append(f)
        print(f)
        print(g)

    if ((i) % args.viz_every == 0) and (i !=0):
        print('step: ',i)    
        fig, axarr = plt.subplots(1,len(relevant_idx), figsize=(4*len(relevant_idx), 4))
        fig.tight_layout()
        list_images = sess.run(gen_ema,{is_training_pl:False, noise_ph:n_fix, modes_ph:m_fix})
        list_images = np.split(list_images, num_region, axis=0)
        for idx, j in enumerate(relevant_idx):
            x=list_images[j]
            axarr[idx].imshow(grid_x(x))
            axarr[idx].set_xlabel(r'$'+labels[j]+'$',fontsize=20)
            axarr[idx].set_xticks([])
            axarr[idx].set_yticks([])
        if args.out_quality == 'high':
            fig.savefig(log_dir+'/image_{0:06}.png'.format(i), format = 'png',dpi = 300)
        else:
            fig.savefig(log_dir+'/image_{0:06}.jpg'.format(i))
        #fig.savefig(log_dir+'/image_{0:06}'.format(i), format = 'pdf',dpi = 1200)

    if ((i) % args.cal_every == 0) and (i !=0) and eval_score:
        q_score = []
        in_list = [[] for _ in range(len(relevant_idx))] 
        for _ in range(100):
            n,m = sample_z([100]*num_region, same_input=True)
            list_images = sess.run(gen_ema,{is_training_pl:False, noise_ph:n, modes_ph:m})
            list_images = np.split(list_images, num_region, axis=0)
            for idx,j in enumerate(relevant_idx):
                in_list[idx].append(((list_images[j]  + 1.) * 127.5).astype('float32'))
                
        for idx,_ in enumerate(relevant_idx):
            tlist=[]
            for k in range(100):
                tlist.append(sess.run(qs,{sample:in_list[idx][k]}))
            score_list[idx].append(sum([vvv in cc[idx] for vvv in np.concatenate(tlist)])/10000.0)
        print([v[-1]*100 for v in score_list])
            
                
    if ((i) % args.cal_every == 0) and (i !=0) and fid_score_calc:

        fid_score = []
        in_list = [[] for _ in range(len(relevant_idx))] 
        for _ in range(10000/args.batch_size):
            n,m = sample_z([args.batch_size]*num_region, same_input=True)
            list_images = sess.run(gen_ema,{is_training_pl:False, noise_ph:n, modes_ph:m})
            list_images = np.split(list_images, num_region, axis=0)
            for idx,j in enumerate(relevant_idx):
                in_list[idx].append(((list_images[j]  + 1.) * 127.5).astype('float32'))
            
        for idx,_ in enumerate(relevant_idx):
            mu_gen, sigma_gen = fid.calculate_activation_statistics(np.concatenate(in_list[idx]), sess, batch_size=100)
            fid_score.append(fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real[idx], sigma_real[idx]))
            #print('FID score [%.4f]'%(fid_score))
        in_list = []
        FID_list.append([fid_score])          
        np.save(log_dir+'/fid_score.npy',np.asarray(FID_list))
        
np.save(log_dir+'/loss_score.npy',np.vstack(fs)) 
[v[-1]*100 for v in score_list]
       
        
