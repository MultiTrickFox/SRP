import data.preprocess as preprocessor
import ai.processor as processor

import res.custom_optimizers as optimZ

import os
import numpy as np
import tensorflow as tf
path = os.path.abspath(os.getcwd())

learningRate = 0.00001
numEpochs = 10
batchingRatio = 0.1

acc_min = 0.7


def startTraining(layers):
    trainX,trainY,devX,devY,testX,testY = loadPickles()

    trainSize = len(trainX)
    batchSize = trainSize*batchingRatio

    with tf.Session() as session:

        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        prediction = processor.neuralFunction(x,layers)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
        optimizer = tf.train.AdamOptimizer(learningRate)
        #optimizer = tf.train.AdagradOptimizer(learningRate)
        #optimizer = optimZ.AMSGrad(learningRate)
        #optimizer = optimZ.Adam(learningRate)
        trainer = optimizer.minimize(cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)),tf.float32))

        dev_cost_sum = tf.summary.scalar('1_dev_cost', cost)
        train_cost_sum = tf.summary.scalar('2_train_cost', cost)
        dev_acc_sum = tf.summary.scalar('3_dev_acc', accuracy)
        train_acc_sum = tf.summary.scalar('4_train_acc', accuracy)

        train_summary = tf.summary.merge([train_cost_sum,train_acc_sum])
        dev_summary = tf.summary.merge([dev_cost_sum,dev_acc_sum])

        session.run(tf.global_variables_initializer())

        summary_file = tf.summary.FileWriter(os.path.join(path, "../ai/logs/"), graph=tf.get_default_graph())

        loss = None
        previousLoss = None
        #isDone = False

        print('Model training in progress..\n')
        #while not isDone:
        while True:

            for currentEpoch in range(numEpochs):

                previousLoss = loss if loss is not None else 0
                batchCounter = 0
                loss = 0

                while batchCounter*batchSize < trainSize:
                    start_ptr = int(batchCounter*batchSize)
                    end_ptr = int((batchCounter+1)*batchSize)
                    batchX_train = trainX[start_ptr:end_ptr]
                    batchY_train = trainY[start_ptr:end_ptr]

                    _, cTrain = session.run([trainer,cost],{x:batchX_train,y:batchY_train})
                    loss += cTrain

                    batchCounter += 1

                train_sum = train_summary.eval({x:trainX,y:trainY})
                dev_sum = dev_summary.eval({x:devX,y:devY})

                summary_file.add_summary(train_sum)
                summary_file.add_summary(dev_sum)

                print('Epoch',currentEpoch,'completed. Loss:',loss)

            if loss < previousLoss: #todo: save when?
                print('\nLoss Decrease Successful.')
                test_acc = accuracy.eval({x:testX,y:testY})
                print('Test Accuracy:',test_acc,'\n')
                if test_acc > acc_min: #add more cond?
                    saveSesh(session)
            #else:
                 #isDone = True

        #return layers


def loadPickles():
    try:
        return gofetchpickles()
    except:
        print('Training Pickles not found.')
        preprocessor.Samples2Pickles()
        return gofetchpickles()


def gofetchpickles():
    trainPickle = os.path.join(path, "../data/trainData.pkl")
    devPickle = os.path.join(path, "../data/devData.pkl")
    testPickle = os.path.join(path, "../data/testData.pkl")
    '''
    trainX, trainY, = pickle.load(open(trainPickle,"rb"))
    devX, devY = pickle.load(open(devPickle,"rb"))
    testX, testY = pickle.load(open(testPickle,"rb"))
    '''
    # Mac OS X pickle fixed version

    trainX, trainY, = preprocessor.pickle_load(trainPickle)
    devX, devY = preprocessor.pickle_load(devPickle)
    testX, testY = preprocessor.pickle_load(testPickle)

    print('Pickles loaded.')
    return np.array(trainX),\
           np.array(trainY),\
           np.array(devX),\
           np.array(devY),\
           np.array(testX),\
           np.array(testY)


def saveSesh(session,my_model_name=None):
    parentPath = os.path.abspath(os.getcwd())
    save_dir = os.path.join(parentPath,'../ai/model_saves/')
    model_name = 'my_model' if my_model_name is None else my_model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,model_name)
    saver = tf.train.Saver()
    saver.save(session,save_path)
    print("Trained Model saved in path:",save_path,'\n')

#someTensor.eval = #tf.get_default_session().run(someTensor)
