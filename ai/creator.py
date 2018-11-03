import tensorflow as tf
import data.preprocess as preprocess

HL_sizes = [40]
in_size, out_size = \
    preprocess.getIOdims()


def create_layers():

    hm_layers = len(HL_sizes) + 1
    layers = []

    for i in range(hm_layers-1):

        prev_size = HL_sizes[i - 1] if i != 0 else in_size
        curr_size = HL_sizes[i]

        layer = {'size': curr_size,
                 'weight': tf.Variable(tf.random_normal([prev_size, curr_size]), name='weight_hidden'+str(i)),
                 'bias': tf.Variable(tf.random_normal([curr_size]), name='bias_hidden'+str(i))
                }

        layers.append(layer)

    prev_size = HL_sizes[-1]
    curr_size = out_size

    out_layer = {'size': out_size,
                 'weight': tf.Variable(tf.random_normal([prev_size, curr_size]), name='weight_outer'),
                 'bias': tf.Variable(tf.random_normal([curr_size]), name='bias_outer')
                }

    layers.append(out_layer)

    print('Neural layers created.')
    return layers
