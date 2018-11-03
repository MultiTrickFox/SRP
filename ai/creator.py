import tensorflow as tf
import data.preprocess as preprocess

numNodesHLs = [40]


def createLayers():
    numInputSize,numClasses = preprocess.getIOdims()
    layers = []
    amountLayers = len(numNodesHLs)+1

    for i in range(0, amountLayers-1):
        if i == 0:
            nrNodePrevLayer = numInputSize
        else:
            nrNodePrevLayer = numNodesHLs[i-1]
        nrNodeCurrLayer = numNodesHLs[i]

        currentHL = {'f_fum':nrNodeCurrLayer,
                    'weight':tf.Variable(tf.random_normal([nrNodePrevLayer, nrNodeCurrLayer]),name='weight_hidden'+str(i)),
                    'bias':tf.Variable(tf.random_normal([nrNodeCurrLayer]),name='bias_hidden'+str(i))
                    }
        layers.append(currentHL)

    nrNodePrevLayer = numNodesHLs[(len(numNodesHLs)-1)]
    nrNodeCurrLayer = numClasses
    outputLayer = {'f_fum':None,
                 'weight':tf.Variable(tf.random_normal([nrNodePrevLayer, nrNodeCurrLayer]),name='weight_outer'),
                 'bias':tf.Variable(tf.random_normal([nrNodeCurrLayer]),name='bias_outer')
                 }
    layers.append(outputLayer)
    print('Neural layers created.')
    return layers

