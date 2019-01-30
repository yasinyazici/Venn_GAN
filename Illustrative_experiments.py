import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from lib.utils import *
from lib.ops import conditioner
from itertools import combinations
ds = tf.contrib.distributions
from models.model_toy import generator,discriminator

parser = argparse.ArgumentParser('')
parser.add_argument('--batch_size', '-b',type=int, default=64)
parser.add_argument('--data_size',type=int, default=64)
parser.add_argument('--num_agent',type=int, default=3)
parser.add_argument('--dataset', '-d', type=str, default='cifar10') #'mnist', 'cifar10'
parser.add_argument('--distribution', type=str, default='small') #'small', 'big'
parser.add_argument('--objective', '-o', type=str, default='gan') #gan, hinge, wgan-gp
parser.add_argument('--model', '-m', type=str, default='dcgan') #dcgan, resnet
parser.add_argument('--gen_type', '-c', type=str, default='independent') #conditional, independent
parser.add_argument('--reg', '-r', type=str, default='d_reg') #d_reg, g_reg
parser.add_argument('--z_dim', '-z', type=int, default=128)
parser.add_argument('--scale', type=float, default=10.0)
parser.add_argument('--n_dis', type=int, default=1, help='number of discriminator update per generator update')
parser.add_argument('--max_iter', type=int, default=5000)
parser.add_argument('--decay', type=float, default=0.999)
#parser.add_argument('--sn', type=bool, default=False)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
#parser.add_argument('--out', default='logs_dogs_cats_mt_64', help='Directory to output the result')
parser.add_argument('--viz_size', type=int, default=25, help='number of images to display')
parser.add_argument('--cal_every', type=int, default=10000, help='Interval of evaluation')
parser.add_argument('--viz_every', type=int, default=200,help='Interval of display')
# args = parser.parse_args()
args = parser.parse_args("--gpu 0 ".split())

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print(args)

bn_g = False
sn_g = False
sn_d = False
params = dict(disc_learning_rate=0.0002,gen_learning_rate=0.0002,beta1=0.0,beta2=0.9)

num_modes = np.power(2,args.num_agent) -1
batch_size = np.power(2,args.num_agent-1) * args.batch_size
batch_size_z = args.batch_size
#params['batch_size_z'] = num_modes * params['batch_size_each_mode']

slim = tf.contrib.slim

connection_map = [[0,3,5,6],
                  [1,4,5,6],
                  [2,3,4,6]]

list_loc= [[-1 ,-1], [0.0, 1], [1  ,-1], [0  ,-1], [1.0, 0], [-1.0,0], [0,0]]
def sample_mog(batch_size, num_mixt=4,num_agent=3, std=0.01):
    data_sample=[]
    cat = ds.Categorical(tf.zeros(num_mixt))
    for i in range(num_agent):
        loc = [list_loc[r] for r in connection_map[i]]
        comps = [ds.MultivariateNormalDiag([float(p[0]),float(p[1])], [std, std]) for p in loc]
        data_sample.append(ds.Mixture(cat, comps).sample(batch_size))
    return data_sample

def disc_loss(real,fake):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real))
                          +tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
def gen_loss(fake):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))

tf.reset_default_graph()

data_samples=sample_mog(batch_size,std=0.1)

noise = ds.Normal(tf.zeros(args.z_dim), tf.ones(args.z_dim)).sample(num_modes*batch_size_z)
#noise = ds.Normal(tf.zeros(args.z_dim), tf.ones(args.z_dim)).sample(batch_size_z)
#noise = tf.concat([noise]*num_modes,axis=0)
modes = tf.concat([i*tf.ones(batch_size_z, dtype='int32') for i in range(num_modes)],axis=0)
is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')

#generator
if args.gen_type == 'conditional':
    gen_out = generator(noise, is_training=is_training_pl ,mode=modes, num_modes=num_modes, sn=sn_g, bn= bn_g, name="gen")
    gen_list = tf.split(gen_out,num_modes,axis=0)
elif args.gen_type == 'independent':
    gen_list = []
    noise_list = tf.split(noise,num_modes,axis=0)
    for i in range(num_modes):
        gen_list.append(generator(noise_list[i], is_training=is_training_pl , sn=sn_g, bn= bn_g, name="gen_{}".format(i)))
        
#standard discriminators
dis_list=[[0]*2 for j in range(args.num_agent)]
for i in range(args.num_agent):
    dis_list[i][0] = discriminator(data_samples[i], sn=sn_d, name = 'dis_'+str(i))
    gen_samples=tf.concat([gen_list[x] for x in connection_map[i]],axis=0)
    dis_list[i][1] = discriminator(gen_samples, sn=sn_d, reuse=True, name = 'dis_'+str(i))

#loss functions
if args.objective == 'gan':
    d_loss = sum([disc_loss(*dis) for dis in dis_list])
    g_loss = sum([gen_loss(dis[-1]) for dis in dis_list])
    g_vars = tf.global_variables(scope="gen")
    d_vars = tf.global_variables(scope="dis")
    
else:
    raise NotImplementedError

#optimizers 
if True:#args.optimizer == 'adam':
    g_train_opt = tf.train.AdamOptimizer(params['gen_learning_rate'],params['beta1'],params['beta2'])
    d_train_opt = tf.train.AdamOptimizer(params['disc_learning_rate'],params['beta1'],params['beta2'])
    d_train_op = d_train_opt.minimize(d_loss, var_list=d_vars)
    g_train_op = g_train_opt.minimize(g_loss, var_list=g_vars)
else:
    raise NotImplementedError

sess = tf.InteractiveSession()
#init input pipelines first
sess.run(tf.global_variables_initializer())

err_list = []
for i in tqdm(range(args.max_iter+1),disable=True):
    for _ in range(args.n_dis):
        f, _ = sess.run([d_loss, d_train_op],{is_training_pl:True})         
    sess.run([g_train_op],{is_training_pl:True})
    
    if ((i) % args.viz_every == 0):
        print('step: ',i)    
        x=[sess.run(g,{is_training_pl:False}) for g in gen_list]
        
        print(x[0].shape)
        #fig, axarr = plt.subplots(1,2, sharex='col', sharey='row', figsize=(12,4))

        fig = plt.figure(1, figsize=(4, 4))
        [plt.plot(v[:,0],v[:,1],'.',alpha=0.5) for v in x]
        plt.axis([-1.5,1.5,-1.5,1.5])
        plt.show()
        
        mean_region = [np.mean(region,axis=0) for region in x]
        err_list.append(np.mean([np.mean(np.abs(xx-yy)) for xx,yy in zip(list_loc,mean_region)]))
        
        #raise ValueError('bad things!')

fig.savefig("/home/***/Pictures/multi_agent_gan/gaussian_toy.pdf", bbox_inches='tight')

'''

print('step: ',i)    
x=[sess.run(g,{is_training_pl:False}) for g in gen_list]
y=[sess.run(d) for d in data_samples]
labels = ['p_{data_1}','p_{data_2}','p_{data_3}']

print(x[0].shape)
fig, axarr = plt.subplots(1,3, sharex='col', sharey='row', figsize=(14,4))

for i,v in enumerate(y):
    axarr[i].plot(v[:,0],v[:,1],'.',alpha=0.5)
    axarr[i].set_xlim([-1.5,1.5])
    axarr[i].set_ylim([-1.5,1.5])
    axarr[i].set_xlabel(r'$'+labels[i]+'$',fontsize=20)

        
axarr.show()
        
        rect_o = patches.Rectangle((-1.42,-0.5),2.80,1.9,linewidth=2,edgecolor='orange',facecolor='none')
        rect_g = patches.Rectangle((-0.5,-1.38),1.92,1.92,linewidth=2,edgecolor='g',facecolor='none')
        rect_b = patches.Rectangle((-1.38,-1.42),1.92,1.88,linewidth=2,edgecolor='b',facecolor='none')
        axarr[0].add_patch(rect_o)
        axarr[0].add_patch(rect_g)
        axarr[0].add_patch(rect_b)
        
    if ((i) % args.viz_every == 0):
        print('step: ',i)    
        x=[sess.run(g,{is_training_pl:False}) for g in gen_list]
        y=[sess.run(d) for d in data_samples]

        print(x[0].shape)
        fig, axarr = plt.subplots(1,2, sharex='col', sharey='row', figsize=(12,4))

        [axarr[1].plot(v[:,0],v[:,1],'.',alpha=0.5) for v in x]
        axarr[1].set_xlim([-1.5,1.5]);axarr[0].set_ylim([-1.5,1.5])
        [axarr[0].plot(v[:,0],v[:,1],'.',alpha=0.5) for v in y]
        axarr[0].set_xlim([-1.5,1.5]);axarr[1].set_ylim([-1.5,1.5])
        
        axarr.show()

        mean_region = [np.mean(region,axis=0) for region in x]
        err_list.append(np.mean([np.mean(np.abs(xx-yy)) for xx,yy in zip(list_loc,mean_region)]))
        
'''
