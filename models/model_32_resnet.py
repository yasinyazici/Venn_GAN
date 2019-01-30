import tensorflow as tf
from lib.ops import * 

g_dim = 64
d_dim = 64
g_max = 512
d_max = 512

def generator(z_inp, mode=0, num_modes = 1, sn=False, bn=True, is_training=False, reuse=False,name='gen'):
    with tf.variable_scope(name, reuse=reuse):

        #z = unit_vector(z_inp)
        z = z_inp
        z = tf.expand_dims(tf.expand_dims(z, 1), 1)
            
        x = conv(z, channels=g_dim*8, kernel=4, stride=1, pad=3, use_bias=False, sn=sn, scope='fc')
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim*8, name = 'cond_0')
        
        x = resblock_g(x, g_dim*8, g_dim*8, sn, 'res_1') 
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim*8, name = 'cond_2')
        x = up_sample(x, scale_factor=2)
        x = resblock_g(x, g_dim*4, g_dim*4, sn, 'res_2')  
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim*4, name = 'cond_3')
        x = up_sample(x, scale_factor=2)
        x = resblock_g(x, g_dim*2, g_dim*2, sn, 'res_3')
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim*2, name = 'cond_4')
        x = up_sample(x, scale_factor=2)
        x = resblock_g(x, g_dim*1, g_dim*1, sn, 'res_4')

        x = lrelu(x, 0.2)
        x = conditioner(x, mode=mode, num_modes=num_modes, dim=g_dim*1, name = 'cond_5')
        x = conv(x, channels=3, kernel=3, stride=1, pad=1, use_bias=True, sn=sn, scope='conv_o')
        x = tf.tanh(x, name='tanh')
                    
        return x

def discriminator(x_inp, num_agent=1, multi_head=False, sn = False, reuse=False, name='disc'):
    
    with tf.variable_scope(name, reuse=reuse):
        x = x_inp
        x = conv(x, channels=d_dim*1, kernel=3, stride=1, pad=1, sn=sn, use_bias=True, scope='conv_i')
        
        x = resblock_d(x, d_dim*1, d_dim*1, sn, 'res_0')
        x = down_sample(x)
        x = resblock_d(x, d_dim*1, d_dim*2, sn, 'res_1')
        x = down_sample(x)
        x = resblock_d(x, d_dim*2, d_dim*4, sn, 'res_2')
        x = down_sample(x)
        x = resblock_d(x, d_dim*4, d_dim*8, sn, 'res_3')

        x = lrelu(x, 0.2)
        o_gan = conv(x, channels=num_agent, kernel=4, stride=1, pad=0, sn=sn, use_bias=True, scope='conv_o')
        o_gan = tf.squeeze(o_gan)
        
        if multi_head: 
            o_class = conv(x, channels=1, kernel=4, stride=1, pad=0, sn=sn, use_bias=True, scope='conv_c')
            o_class = tf.squeeze(o_class)
            return o_gan, o_class
        else:
            return o_gan

def resblock_g(x, hid_dim, out_dim, sn, name):
    with tf.variable_scope(name):
        s = x
        if x.get_shape().as_list()[-1] != out_dim:
            s = conv(x, channels=out_dim, kernel=1, stride=1, pad=0, sn=sn, use_bias=False, scope='conv_s')
        x = lrelu(x, 0.2)
        x = conv(x, channels=hid_dim, kernel=3, stride=1, pad=1, sn=sn, use_bias=True, scope='conv_0')
        x = lrelu(x, 0.2)
        x = conv(x, channels=out_dim, kernel=3, stride=1, pad=1, sn=sn, use_bias=False, scope='conv_1')
        return s + x
    
def resblock_d(x, hid_dim, out_dim, sn, name):
    with tf.variable_scope(name):
        s = x
        if x.get_shape().as_list()[-1] != out_dim:
            s = conv(x, channels=out_dim, kernel=1, stride=1, pad=0, sn=sn, use_bias=False, scope='conv_s')
        x = lrelu(x, 0.2)
        x = conv(x, channels=hid_dim, kernel=3, stride=1, pad=1, sn=sn, use_bias=True, scope='conv_0')
        x = lrelu(x, 0.2)
        x = conv(x, channels=out_dim, kernel=3, stride=1, pad=1, sn=sn, use_bias=True, scope='conv_1')
        return s + x