import tensorflow as tf
from lib.ops import * 

g_dim = 256
d_dim = 256

def generator(z_inp, mode=0, num_modes = 1, output_dim=2, sn=False, bn=True, is_training=False, reuse=False, name='gen'):
    
    with tf.variable_scope(name, reuse=reuse):

        x = z_inp#unit_vector(z_inp)
        x = fully_connected(x, g_dim, sn=sn, scope='fully_0')
        if bn: x = batch_norm(x, is_training, center=False, scale=False, scope='bn0')
        x = conditioner_fc(x, mode=mode, num_modes=num_modes, dim=g_dim, name = 'cond_0')
        x = lrelu(x, alpha=0.2)
        
        x = fully_connected(x, g_dim, sn=sn, scope='fully_1')
        if bn: x = batch_norm(x, is_training, center=False, scale=False, scope='bn1')
        x = conditioner_fc(x, mode=mode, num_modes=num_modes, dim=g_dim, name = 'cond_1')
        x = lrelu(x, alpha=0.2)
            
        x = fully_connected(x, g_dim, sn=sn, scope='fully_2')
        if bn: x = batch_norm(x, is_training, center=False, scale=False, scope='bn2')
        x = conditioner_fc(x, mode=mode, num_modes=num_modes, dim=g_dim, name = 'cond_2')
        x = lrelu(x, alpha=0.2)
        
        x = fully_connected(x, output_dim, sn=sn, scope='fully_3')
        #x = tf.linear(x, name='tanh')
        
        return x

def discriminator(x_inp, num_agent=1, multi_head=False, sn=False, reuse=False,name='disc'):
    
    with tf.variable_scope(name, reuse=reuse):
        x = x_inp
        x = fully_connected(x, d_dim, sn=sn, scope='fully_0')
        x = lrelu(x, alpha=0.2)
        
        x = fully_connected(x, d_dim, sn=sn, scope='fully_1')
        x = lrelu(x, alpha=0.2)

        x = fully_connected(x, d_dim, sn=sn, scope='fully_2')
        x = lrelu(x, alpha=0.2)

        out = fully_connected(x, num_agent, sn=sn, scope='fully_3')
        
        if multi_head:
            out2 = fully_connected(x, 1, sn=sn, scope='fully_3_mh')
            return out, out2
        else:
            return out

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