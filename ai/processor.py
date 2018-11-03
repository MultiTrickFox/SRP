import data.preprocess as preprocessor
import ai.creator as creator

import os
import numpy as np
import tensorflow as tf


def isInit():
    parentPath = os.path.abspath(os.getcwd())
    load_dir = os.path.join(parentPath,'../ai/model_saves/')
    meta_name = 'my_model.meta'
    load_path = os.path.join(load_dir,meta_name)
    return os.path.exists(load_path)


def neuralFunction(data,layers):
    layerOutputs = np.empty(len(layers),object)

    for i in range (0,len(layers)):
        currentWeights = layers[i]['weight']
        currentBiass = layers[i]['bias']

        if i < len(layers)-1:
            if i == 0:
                prevOutput = data
            else:
                prevOutput = layerOutputs[i-1]

            currentOutput = tf.nn.relu(tf.add(tf.matmul(prevOutput,currentWeights),currentBiass))
            layerOutputs[i] = currentOutput

        else:
            prevOutput = layerOutputs[i-1]
            finalOutput = tf.add(tf.matmul(prevOutput,currentWeights),currentBiass)
            return finalOutput


def getOutput(data):
    layers = []

    numNodesHLs = creator.numNodesHLs
    amountLayers = len(creator.numNodesHLs)+1

    parentPath = os.path.abspath(os.getcwd())
    load_dir = os.path.join(parentPath,'../ai/model_saves/')
    load_path = os.path.join(load_dir,'my_model.meta')
    with tf.Session() as session:

        loader = tf.train.import_meta_graph(load_path)
        loader.restore(session,tf.train.latest_checkpoint(load_dir))
        graph = tf.get_default_graph()

        for i in range(0, amountLayers-1):
            nrNodeCurrLayer = numNodesHLs[i]
            #currentWeight = tf.get_variable('weight_hidden'+str(i)+':0')
            #currentBias = tf.get_variable('bias_hidden'+str(i)+':0')
            currentWeight = graph.get_tensor_by_name('weight_hidden'+str(i)+':0')
            currentBias = graph.get_tensor_by_name('bias_hidden'+str(i)+':0')

            currentHL = {'f_fum':nrNodeCurrLayer,
                        'weight':currentWeight,
                        'bias':currentBias
                        }
            layers.append(currentHL)

        #currentWeight = tf.get_variable('weight_outer:0')
        #currentBias = tf.get_variable('bias_outer:0')
        currentWeight = graph.get_tensor_by_name('weight_outer:0')
        currentBias = graph.get_tensor_by_name('bias_outer:0')

        outputLayer = {'f_fum':None,
                    'weight':currentWeight,
                    'bias':currentBias
                    }
        layers.append(outputLayer)
        input = tf.placeholder(tf.float32)

        neuralOutput = neuralFunction(input,layers)
        softOutput = tf.nn.softmax(neuralOutput)

        #guarantee_initialized_variables(session)
        inputData = preprocessor.human2aiConverter(data)
        output = session.run(softOutput,{input:[inputData]})
        output = output[0]

    return output


def getStats():
    parentPath = os.path.abspath(os.getcwd())
    load_dir = os.path.join(parentPath,'../ai/model_saves/')
    meta_name = 'my_model.meta'
    load_path = os.path.join(load_dir,meta_name)

    with tf.Session() as session:

        loader = tf.train.import_meta_graph(load_path)
        loader.restore(session,tf.train.latest_checkpoint(load_dir))
        graph = tf.get_default_graph()

        operations = []
        for op in graph.get_operations():
            operations.append(op.name)

        return operations

def raw_to_soft_output(raw):
    print(raw) #here 4 debug purp
    with tf.Session() as sess:
        _ = tf.constant(raw)
        max_prob = tf.reduce_max(_)
        sess.run(tf.global_variables_initializer())
        softResult = sess.run(max_prob)
        for classNr in range(0, len(raw)):
            currentProb = raw[classNr]
            if str("%.5f" % currentProb)[:4] == str("%.5f" % softResult)[:4]:
                return classNr


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
