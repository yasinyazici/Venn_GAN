import tensorflow as tf
from lib.ops import * 

g_dim = 256
d_dim = 256

def generator(z_inp, mode=0, num_modes = 1, sn=False, bn=True, is_training=False, reuse=False, name='gen'):
    
    with tf.variable_scope(name, reuse=reuse):

        z = z_inp#unit_vector(z_inp)
        z = tf.expand_dims(tf.expand_dims(z, 1), 1)
        x = conv(z, channels=g_dim, kernel=4, stride=1, pad=3, sn=sn, use_bias=False, scope ='conv_0')
        if bn: x = batch_norm(x, is_training, center=False, scale=False, scope='bn0')
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim, name = 'cond_0')
        x = lrelu(x, alpha=0.2)
        
        x = conv(x, channels=g_dim/2, kernel=4, stride=1, pad=3, sn=sn, use_bias=False, scope ='conv_1')
        if bn: x = batch_norm(x, is_training, center=False, scale=False, scope='bn1')
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim/2, name = 'cond_1')
        x = lrelu(x, alpha=0.2)
            
        x = up_sample(x, scale_factor=2)
        x = conv(x, channels=g_dim/4, kernel=3, stride=1, pad=1, sn=sn, use_bias=False, scope ='conv_2')
        if bn: x = batch_norm(x, is_training, center=False, scale=False, scope='bn2')
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim/4, name = 'cond_2')
        x = lrelu(x, alpha=0.2)
            
        x = up_sample(x, scale_factor=2)
        x = conv(x, channels=g_dim/8, kernel=3, stride=1, pad=1, sn=sn, use_bias=False, scope ='conv_3')
        if bn: x = batch_norm(x, is_training, center=False, scale=False, scope='bn3')
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim/8, name = 'cond_3')
        x = lrelu(x, alpha=0.2)
        
        x = conv(x, channels=1, kernel=3, stride=1, pad=1, sn=sn, use_bias=True, scope ='conv_4')
        x = tf.tanh(x, name='tanh')
        
        return x
'''    
def generator(z_inp, mode=0, num_modes = 1, sn=False, bn=True, is_training=False, reuse=False, name='gen'):
    
    with tf.variable_scope(name, reuse=reuse):

        z = z_inp#unit_vector(z_inp)
        ohot = tf.one_hot(mode,num_modes)
        z = tf.concat([z,ohot],axis=1)
        z = tf.expand_dims(tf.expand_dims(z, 1), 1)
        x = conv(z, channels=g_dim, kernel=4, stride=1, pad=3, sn=sn, use_bias=False, scope ='conv_0')
        x = lrelu(x, alpha=0.2)
        
        x = conv(x, channels=g_dim/2, kernel=4, stride=1, pad=3, sn=sn, use_bias=False, scope ='conv_1')
        x = lrelu(x, alpha=0.2)
            
        x = up_sample(x, scale_factor=2)
        x = conv(x, channels=g_dim/4, kernel=3, stride=1, pad=1, sn=sn, use_bias=False, scope ='conv_2')
        x = lrelu(x, alpha=0.2)
            
        x = up_sample(x, scale_factor=2)
        x = conv(x, channels=g_dim/8, kernel=3, stride=1, pad=1, sn=sn, use_bias=False, scope ='conv_3')
        x = lrelu(x, alpha=0.2)
        
        x = conv(x, channels=1, kernel=3, stride=1, pad=1, sn=sn, use_bias=True, scope ='conv_4')
        x = tf.tanh(x, name='tanh')
        
        return x
'''
def discriminator(x_inp, num_agent=1, multi_head=False, sn=False, bn=False, reuse=False, is_training=False, name='disc'):
    
    with tf.variable_scope(name, reuse=reuse):
        x = x_inp
        x = conv(x, channels=d_dim/4, kernel=4, stride=2, pad=1, sn=sn, use_bias=True, scope ='conv_0')
        if bn: x = batch_norm(x, is_training, center=True, scale=True, scope='bn0')
        x = lrelu(x, alpha=0.2)

        x = conv(x, channels=d_dim/2, kernel=4, stride=2, pad=1, sn=sn, use_bias=True, scope ='conv_1')
        if bn: x = batch_norm(x, is_training, center=True, scale=True, scope='bn1')
        x = lrelu(x, alpha=0.2)

        x = conv(x, channels=d_dim, kernel=4, stride=2, pad=1, sn=sn, use_bias=True, scope ='conv_2')
        if bn: x = batch_norm(x, is_training, center=True, scale=True, scope='bn2')
        x = lrelu(x, alpha=0.2)

        out = conv(x, channels=num_agent, kernel=3, stride=1, pad=0, sn=sn, use_bias=True, scope ='conv_3')
        out = tf.squeeze(out)
        
        if multi_head:
            out2 = conv(x, channels=1, kernel=3, stride=1, pad=0, sn=sn, use_bias=True, scope ='conv_3_m')
            out2 = tf.squeeze(out2)
            return out, out2
        else:
            return out
        
def sanity_generator(batch_size=100, name='gen'):
    with tf.variable_scope(name):
        #split_num = 100/batch_size
        x = tf.get_variable("kernel", shape=[batch_size, 28, 28, 1], initializer=tf.initializers.truncated_normal(0.0,0.5))
        x = tf.tanh(x, name='tanh')
        #x = tf.split(x,split_num,axis=0)
                    
        return x
        
def encoder(x_inp, sn=False, reuse=False,name='enc'):
    
    with tf.variable_scope(name, reuse=reuse):
        x = x_inp
        x = conv(x, channels=d_dim/4, kernel=4, stride=2, pad=1, sn=sn, use_bias=True, scope ='conv_0')
        x = lrelu(x, alpha=0.2)

        x = conv(x, channels=d_dim/2, kernel=4, stride=2, pad=1, sn=sn, use_bias=True, scope ='conv_1')
        x = lrelu(x, alpha=0.2)

        x = conv(x, channels=d_dim, kernel=4, stride=2, pad=1, sn=sn, use_bias=True, scope ='conv_2')
        x = lrelu(x, alpha=0.2)

        return x
    
def classifier(x_inp, num_agent=1, multi_head=False, multi_layer=False, sn=False, reuse=False,name='classifier'):
    
    with tf.variable_scope(name, reuse=reuse):
        x = x_inp
        if multi_layer:
            x = conv(x, channels=num_agent, kernel=3, stride=1, pad=1, sn=sn, use_bias=True, scope ='ml0')
            
        out = conv(x, channels=num_agent, kernel=3, stride=1, pad=0, sn=sn, use_bias=True, scope ='conv_3')
        out = tf.squeeze(out)
        
        if multi_head:
            out2 = conv(x, channels=1, kernel=3, stride=1, pad=0, sn=sn, use_bias=True, scope ='conv_3_m')
            out2 = tf.squeeze(out2)
            return out, out2
        else:
            return out