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
import scipy.special
import fid
from lib.Data_model import Data_model
import cPickle as pickle


ds = tf.contrib.distributions

parser = argparse.ArgumentParser('')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--batch_size', '-b',type=int, default=16)
parser.add_argument('--resolution',type=int, default=28)
parser.add_argument('--num_agent', '-n',type=int, default=3)
parser.add_argument('--connection_type', '-c', type=str, default='v1') #'v1','v3'
parser.add_argument('--distribution', type=str, default='small') #'small', 'big'
parser.add_argument('--dataset', '-d', type=str, default='mnist') #'mnist', 'mnist_fashion','cifar10', 'omniglot', 'omniglot_CG', 'omniglot_CL', 'omniglot_GL'
parser.add_argument('--imbalance',type=str, default='balanced') # 'balanced', 'type1', 'type2'
parser.add_argument('--gen_type', type=str, default='conditional') #conditional, independent
parser.add_argument('--objective', '-o', type=str, default='gan') #gan, hinge, wgan-gp
parser.add_argument('--model', '-m', type=str, default='dcgan') #dcgan, resnet
parser.add_argument('--reg', '-r', type=str, default='None') #'None','real', 'fake'
parser.add_argument('--classify', type=bool, default=True)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--z_dim', '-z', type=int, default=128)
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--class_scale', type=float, default=0.1)
parser.add_argument('--n_dis', type=int, default=1, help='number of discriminator update per generator update')
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--decay', type=float, default=0.999)
#parser.add_argument('--sn', type=bool, default=False)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', default='venn', help='Directory to output the result')
parser.add_argument('--viz_size', type=int, default=36, help='number of images to display')
parser.add_argument('--cal_every', type=int, default=2000, help='Interval of evaluation')
parser.add_argument('--viz_every', type=int, default=1000,help='Interval of display')
parser.add_argument('--out_quality', type=str, default='normal',help='quality of display') #"normal", 'high'
args = parser.parse_args()
print(args)

# fix random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# network settings
g_lr=0.0002
d_lr=0.0002
beta1 = args.beta1
beta2 = args.beta2

sn_d = True
sn_g = False
bn = True
if args.reg in ['real', 'fake']:
    bn = False
    sn_d = False
    sn_g = False

# Paths
#==============================================================================
dir_name = os.path.dirname(__file__)
log_dir = os.path.abspath('{}/logs/{}_{}/{}_{}_{}_{}_{}_{}_{}_{}'.format(dir_name,args.out, args.dataset, 
                          args.connection_type, str(args.num_agent), args.distribution, args.gen_type, 
                          args.classify, args.model, args.reg, str(args.imbalance)))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
path = os.path.normpath(dir_name)
x = path.split(os.sep)
if 'users' in x:
    data_dir = '/home/users/ntu/yasin001/scratch/data/'
    stat_dir = '/home/users/ntu/yasin001/project/cooperation-game-master/stats/'
elif 'yazici' in x:
    data_dir = '/home/yazici/Documents/data/'
    stat_dir = '/home/yazici/Documents/python/cooperation-game-master/stats/'
else:
    raise NotImplementedError("not implemented")

inception_path = data_dir + 'imagenet_model/classify_image_graph_def.pb'

data_model = Data_model(args.dataset, args.model, args.resolution, data_dir, \
                        args.connection_type, args.num_agent, args.distribution, args.imbalance)

_, generator, discriminator = data_model.dataset_and_model()
connection_map, dist_list, cc, relevant_idx = data_model.cmap_and_dlist()
distrib = data_model.process_distribution()
trainx = data_model.trainx

num_region = np.power(2,args.num_agent) -1
bs_list = [args.batch_size for _ in range(num_region)]
batch_size_d = [len(sub_map)*args.batch_size for sub_map in connection_map]

# visualize datasets 
#==============================================================================
if args.dataset not in ['celeba','stanford_kaggle','celeba_ffhq']:
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


data_samples=[]
for i in range(args.num_agent):
    dataset = tf.data.Dataset.from_tensor_slices(distrib[i]).shuffle(10000).repeat()
    if args.dataset in ['celeba','stanford_kaggle','celeba_ffhq']: dataset = dataset.map(parse_function)
    iterator = dataset.batch(batch_size_d[i]).prefetch(100).make_one_shot_iterator()
    data_samples.append(tf.cast(iterator.get_next(),tf.float32))

def sample_z(bs_list,same_input=False):
    if same_input:
        noise = np.random.normal(size=(bs_list[0],args.z_dim)).astype(dtype='float32')
        noise = np.vstack([noise] * len(bs_list))
    else:
        noise = np.random.normal(size=(sum(bs_list),args.z_dim)).astype(dtype='float32')
        
    modes = np.concatenate([idx*np.ones(bs, dtype='int32') for idx, bs in enumerate(bs_list)],axis=0)
    return noise, modes

#placeholders
is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
noise_ph = tf.placeholder(tf.float32, (None, args.z_dim), name='noise_ph')
modes_ph = tf.placeholder(tf.int32, (None,), name='modes_ph')

#generator
if args.gen_type == 'conditional':
    gen_out = generator(noise_ph, mode=modes_ph, num_modes=num_region, sn=sn_g, bn=bn, is_training=is_training_pl, name="gen")
    gen_list = tf.split(gen_out, bs_list, 0)
    gen_ema = generator(noise_ph, mode=modes_ph, num_modes=num_region, sn=sn_g, bn=bn, is_training=False, name="ema_gen")
elif args.gen_type == 'independent':
    gen_list = []
    noise_list = tf.split(noise_ph,num_region,axis=0)
    for i in range(num_region):
        gen_list.append(generator(noise_list[i], sn=sn_g, bn=bn, is_training=is_training_pl, name="gen_{}".format(i)))  
    gen_ema= []
    for i in range(num_region):
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
    c = discriminator(input_class, len(relevant_idx), sn=False, name = 'classifier')
    labels = tf.concat([idx*tf.ones(bs, dtype='int32') for idx, bs in enumerate([bs_list[indx] for indx in relevant_idx])],axis=0)
    d_loss.append(args.class_scale*class_loss(logits=c, labels=labels))
    g_loss.append(args.class_scale*class_loss(logits=c, labels=labels))

if args.reg == 'real':
    for i in range(args.num_agent):
        ddx = tf.gradients(dis_list[i][0], data_samples[i])[0]
        print(ddx.get_shape().as_list())
        d_loss.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(ddx),axis=[1,2,3]))) * args.scale)
elif args.reg == 'fake':
    for i in range(args.num_agent):
        ddx = tf.gradients(dis_list[i][1], gen_samples[i])[0]
        print(ddx.get_shape().as_list())
        d_loss.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(ddx),axis=[1,2,3]))) * args.scale)

g_vars = tf.global_variables(scope="gen")
g_ema_vars = tf.global_variables(scope="ema_gen")
d_vars = tf.trainable_variables(scope="dis")  
if args.classify:
    d_vars += tf.trainable_variables(scope="classifier")  
    
#optimizers 
if True:#args.optimizer == 'adam':
    g_train_opt = tf.train.AdamOptimizer(g_lr, beta1, beta2)
    d_train_opt = tf.train.AdamOptimizer(d_lr, beta1, beta2)
    d_train_op = d_train_opt.minimize(np.sum(d_loss), var_list=d_vars)
    g_train_op = g_train_opt.minimize(np.sum(g_loss), var_list=g_vars)
else:
    raise NotImplementedError

with tf.control_dependencies([d_train_op,g_train_op]):
    ema_op = exponential_moving_average(g_ema_vars,g_vars,args.decay)

# accuracy evaluation
eval_score = False
if args.dataset in ['mnist','mnist_fashion','cifar10']:
    eval_score = True
    acc_list = []
    pt_vars = np.load( stat_dir+'c_ema_vars_{}_{}.npy'.format(args.model,args.dataset))
    _,h,w,c = np.shape(trainx)
    sample = tf.placeholder(tf.float32, (None, h, w, c), name='sample')
    logits, feat = discriminator(sample, num_agent=10, features=True, sn=False, name = 'q_score')
    qs = tf.argmax(logits, 1)
    q_vars = tf.global_variables(scope="q_score")
    copy_op = copy_params(q_vars,pt_vars)
    
# fid stats load
fid_score_calc = False
if args.dataset in ['mnist','mnist_fashion','cifar10']:
    fid_score_calc = True
    FID_list = []
    path2load = stat_dir+'fid_dist_{}_{}_{}_{}_{}.npy'.format(args.dataset, args.model,\
                              args.distribution, str(args.num_agent), args.connection_type)
    with open (path2load, 'rb') as fp:
        (mu_real, sigma_real) = pickle.load(fp)   
'''    
fid_score_calc = False
if args.dataset in ['mnist','mnist_fashion','cifar10']:
    fid_score_calc = True
    FID_list = []
    path2load = stat_dir+'fid_{}_{}_{}_{}_{}.npy'.format(args.dataset, args.model,\
                              args.distribution, str(args.num_agent), args.connection_type)
    with open (path2load, 'rb') as fp:
        (mu_real, sigma_real) = pickle.load(fp)
'''        
        
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(copy_params(g_ema_vars,g_vars))
sess.run(copy_op)
      
# misc settings       
if args.num_agent == 2:
    fig_labels = ['S_1 \\backslash S_2','S_2 \\backslash S_1','S_1 \cap S_2']
elif args.num_agent == 3:
    fig_labels = ['S_1 \\backslash (S_2 \cup S_3)','S_2 \\backslash (S_1 \cup S_3)','S_3 \\backslash (S_1 \cup S_2)',
              '(S_1 \cap S_3) \\backslash S_2','(S_2 \cap S_3) \\backslash S_1','(S_1 \cap S_2) \\backslash S_3',
              'S_1 \cap S_2 \cap S_3']
else:
    raise NotImplementedError
        
if args.dataset in ['celeba','ffhq','celeba_ffhq']:
    args.viz_size = 16
fs = []  
n_fix,m_fix = sample_z([args.viz_size]*num_region, same_input=True)
for i in tqdm(range(args.max_iter+1),disable=False):
    for _ in range(args.n_dis):
        n,m = sample_z([args.batch_size]*num_region, same_input=True)
        f, _ = sess.run([d_loss, d_train_op],{is_training_pl:True, noise_ph:n, modes_ph:m})   
    n,m = sample_z([args.batch_size]*num_region, same_input=True)
    g, _, _ = sess.run([g_loss, g_train_op,ema_op],{is_training_pl:True, noise_ph:n, modes_ph:m})
    
    if ((i) % 500 == 0):
        fs.append(f)
        np.save(log_dir+'/loss_score.npy',np.vstack(fs)) 
        print(f)
        print(g)
        
    if ((i) % args.viz_every == 0):
        print('step: ',i)    
        fig, axarr = plt.subplots(1,len(relevant_idx), figsize=(4*len(relevant_idx), 4))
        fig.tight_layout()
        list_images = sess.run(gen_ema,{is_training_pl:False, noise_ph:n_fix, modes_ph:m_fix})
        list_images = np.split(list_images, num_region, axis=0)
        for idx, j in enumerate(relevant_idx):
            x=list_images[j]
            axarr[idx].imshow(grid_x(x))
            axarr[idx].set_xlabel(r'$'+fig_labels[j]+'$',fontsize=20)
            axarr[idx].set_xticks([])
            axarr[idx].set_yticks([])
        if args.out_quality == 'high':
            fig.savefig(log_dir+'/image_{0:06}.png'.format(i), format = 'png',dpi = 300)
        else:
            fig.savefig(log_dir+'/image_{0:06}.jpg'.format(i))
            
            '''  
    if ((i) % args.cal_every == 0) and (i !=0) and eval_score:
        
        in_list = [[] for _ in range(len(relevant_idx))]
        for _ in range(100):
            n,m = sample_z([100]*num_region, same_input=True)
            list_images = sess.run(gen_ema,{is_training_pl:False, noise_ph:n, modes_ph:m})
            list_images = np.split(list_images, num_region, axis=0)
            for idx,j in enumerate(relevant_idx):
                in_list[idx].append(list_images[j].astype('float32'))
            
        q_score = [] 
        for idx,_ in enumerate(relevant_idx):
            tlist=[]
            for k in range(100):
                tlist.append(sess.run(qs,{sample:in_list[idx][k]}))
            q_score.append(np.mean([np.any(item == cc[idx]) for item in np.concatenate(tlist)]))
        
        acc_list.append([np.round(v*100,2) for v in q_score])
        np.save(log_dir+'/acc_score.npy',np.vstack(acc_list)) 
        print(acc_list[-1])
            
              
    if ((i) % args.cal_every == 0) and (i !=0) and fid_score_calc:

        fid_score = []
        in_list = [[] for _ in range(len(relevant_idx))] 
        for _ in range(100):
            n,m = sample_z([100]*num_region, same_input=True)
            list_images = sess.run(gen_ema,{is_training_pl:False, noise_ph:n, modes_ph:m})
            list_images = np.split(list_images, num_region, axis=0)
            for idx,j in enumerate(relevant_idx):
                in_list[idx].append(list_images[j].astype('float32'))
            
        for idx,_ in enumerate(relevant_idx):
            list_ = []
            for batch in in_list[idx]:
                list_.append(sess.run(feat,{sample:batch}))
            list_ = np.reshape(np.concatenate(list_,0),(10000,-1))
            mu_gen = np.mean(list_, axis=0)
            sigma_gen = np.cov(list_, rowvar=False)            
            fid_score.append(np.round(fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real[idx], sigma_real[idx]),2))
        print(fid_score)
        in_list = []
        FID_list.append(fid_score)          
        np.save(log_dir+'/fid_score.npy',np.vstack(FID_list))
    '''    
    if ((i) % args.cal_every == 0) and (i !=0) and eval_score:
        
        in_list = [[] for _ in range(args.num_agent)]
        for _ in range(100):
            n,m = sample_z([100]*num_region, same_input=True)
            list_images = sess.run(gen_ema,{is_training_pl:False, noise_ph:n, modes_ph:m})
            list_images = np.split(list_images, num_region, axis=0)
            for idx, sub_list in enumerate(connection_map):
                for item in sub_list:
                    in_list[idx].append(list_images[item].astype('float32'))
            
        q_score = [] 
        for idx in range(args.num_agent):
            tlist=[]
            for k in range(100):
                tlist.append(sess.run(qs,{sample:in_list[idx][k]}))
            q_score.append(np.mean([np.any(item == dist_list[idx]) for item in np.concatenate(tlist)]))
        
        acc_list.append([np.round(v*100,2) for v in q_score])
        np.save(log_dir+'/acc_score.npy',np.vstack(acc_list)) 
        print(acc_list[-1])
       
                
    if ((i) % args.cal_every == 0) and (i !=0) and fid_score_calc:

        fid_score = []
        for idx in range(args.num_agent):
            list_ = []
            for k in range(100):
            #for batch in in_list[idx]:
                list_.append(sess.run(feat,{sample:in_list[idx][k]}))
            list_ = np.reshape(np.concatenate(list_,0),(10000,-1))
            mu_gen = np.mean(list_, axis=0)
            sigma_gen = np.cov(list_, rowvar=False)            
            fid_score.append(np.round(fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real[idx], sigma_real[idx]),2))
        print(fid_score)
        in_list = []
        FID_list.append(fid_score)          
        np.save(log_dir+'/fid_score.npy',np.vstack(FID_list))


        