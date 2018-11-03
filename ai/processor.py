import data.preprocess as preprocessor
import ai.creator as creator

import os
import numpy as np
import tensorflow as tf


def isInit():
    parent_path = os.path.abspath(os.getcwd())
    load_dir = os.path.join(parent_path, '../ai/model_saves/')
    meta_name = 'my_model.meta'
    load_path = os.path.join(load_dir,meta_name)
    is_init = os.path.exists(load_path)
    return is_init


def neural_fn(data, layers):

    layer_outs = np.empty(len(layers), object)

    for i, layer in enumerate(layers):

        weight = layer['weight']
        bias = layer['bias']

        if i < len(layers)-1:

            prev_out = layer_outs[i-1] if i != 0 else data
            this_out = tf.nn.relu(tf.add(tf.matmul(prev_out, weight), bias))

            layer_outs[i] = this_out

        else:

            prev_out = layer_outs[i-1]
            final_out = tf.add(tf.matmul(prev_out, weight), bias)

            return final_out


def get_output_to(data):

    HL_sizes = creator.HL_sizes
    hm_layers = len(creator.HL_sizes) + 1
    layers = []

    parent_path = os.path.abspath(os.getcwd())
    load_dir = os.path.join(parent_path, '../ai/model_saves/')
    load_path = os.path.join(load_dir, 'my_model.meta')

    with tf.Session() as session:

        # LOAD MODEL

        loader = tf.train.import_meta_graph(load_path)
        loader.restore(session,tf.train.latest_checkpoint(load_dir))
        graph = tf.get_default_graph()

        for i in range(0, hm_layers-1):

            layer_size = HL_sizes[i]

            # weight = tf.get_variable('weight_hidden'+str(i)+':0')
            # bias = tf.get_variable('bias_hidden'+str(i)+':0')
            weight = graph.get_tensor_by_name('weight_hidden'+str(i)+':0')
            bias = graph.get_tensor_by_name('bias_hidden'+str(i)+':0')

            layer = {'f_fum': layer_size,
                     'weight': weight,
                     'bias': bias
                    }

            layers.append(layer)

        # weight = tf.get_variable('weight_outer:0')
        # bias = tf.get_variable('bias_outer:0')
        weight = graph.get_tensor_by_name('weight_outer:0')
        bias = graph.get_tensor_by_name('bias_outer:0')

        out_layer = {'f_fum': creator.out_size,
                     'weight': weight,
                     'bias': bias
                    }

        layers.append(out_layer)

        # RUN SESSION

        inp = tf.placeholder(tf.float32)
        neural_output = neural_fn(inp, layers)
        soft_output = tf.nn.softmax(neural_output)
        # guarantee_initialized_variables(session)

        output = session.run(soft_output, {inp: [preprocessor.human2aiConverter(data)]})

    return output[0]


def get_stats():

    parent_path = os.path.abspath(os.getcwd())
    load_dir = os.path.join(parent_path, '../ai/model_saves/')
    meta_name = 'my_model.meta'
    load_path = os.path.join(load_dir, meta_name)

    with tf.Session() as session:

        loader = tf.train.import_meta_graph(load_path)
        loader.restore(session, tf.train.latest_checkpoint(load_dir))
        graph = tf.get_default_graph()

        operations = []
        for op in graph.get_operations():
            operations.append(op.name)

        return operations


def raw_to_soft_output(raw):

    print(raw)      # todo: remove when done.

    with tf.Session() as sess:

        _ = tf.constant(raw)
        max_prob = tf.reduce_max(_)

        sess.run(tf.global_variables_initializer())
        result = sess.run(max_prob)

        for class_nr, class_prob in enumerate(raw):
            if str("%.5f" % class_prob)[:4] == str("%.5f" % result)[:4]:
                return class_nr


def soft_to_speech_output(soft):
    return responseArray[soft]


responseArray = [ # read this from a txt.
    'You feel happy.',
    'You seem sad.',
    'You look excited.',
    'You seem disturbed.'
]


"""
def guarantee_initialized_variables(session, list_of_variables = None):
    if list_of_variables is None:
        list_of_variables = tf.all_variables()
    uninitialized_variables = list(tf.get_variable(name) for name in
                                   session.run(tf.report_uninitialized_variables(list_of_variables)))
    session.run(tf.initialize_variables(uninitialized_variables))
    return uninitialized_variables
"""
