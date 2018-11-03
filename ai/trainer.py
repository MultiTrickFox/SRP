import data.preprocess as preprocessor
import ai.processor as processor

# import res.custom_optimizers as optimZ

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.client \
    import device_lib
path = os.path.abspath(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


lr = 0.01
hm_epochs = 10
batch_ratio = 0.1


def train(layers):

    train_x, train_y, dev_x, dev_y, test_x, test_y = load_data_pkl()

    hm_samples = len(train_x)
    batch_size = hm_samples * batch_ratio

    with tf.Session() as session:

        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        prediction = processor.neural_fn(x, layers)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(lr)

        # optimizer = tf.train.AdagradOptimizer(learningRate)
        # optimizer = optimZ.AMSGrad(learningRate)
        # optimizer = optimZ.Adam(learningRate)

        trainer = optimizer.minimize(cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)), tf.float32))

        dev_cost_sum = tf.summary.scalar('1_dev_cost', cost)
        train_cost_sum = tf.summary.scalar('2_train_cost', cost)
        dev_acc_sum = tf.summary.scalar('3_dev_acc', accuracy)
        train_acc_sum = tf.summary.scalar('4_train_acc', accuracy)

        train_summary = tf.summary.merge([train_cost_sum,train_acc_sum])
        dev_summary = tf.summary.merge([dev_cost_sum,dev_acc_sum])

        session.run(tf.global_variables_initializer())

        summary_file = tf.summary.FileWriter(os.path.join(path, "../ai/logs/"), graph=tf.get_default_graph())

        print('Model training in progress..\n')

        while True:

            for epoch in range(hm_epochs):

                batch_counter = 0
                loss = 0

                while batch_counter * batch_size < hm_samples:

                    start_ptr = int(batch_counter*batch_size)
                    end_ptr = int((batch_counter+1)*batch_size)
                    train_x_batch = train_x[start_ptr:end_ptr]
                    train_y_batch = train_y[start_ptr:end_ptr]

                    with tf.device(get_available_gpus()[0]):
                        _, loss_train = session.run([trainer, cost], {x: train_x_batch, y: train_y_batch})

                    loss += loss_train

                    batch_counter += 1

                train_sum = train_summary.eval({x: train_x, y: train_y})
                dev_sum = dev_summary.eval({x: dev_x, y: dev_y})

                summary_file.add_summary(train_sum)
                summary_file.add_summary(dev_sum)

                print(f'Epoch {epoch} completed. Loss: {loss}')

    # return layers


def load_data_pkl():
    try:
        return fetch_pickles()
    except Exception as exception:
        print('Training Pickles not found.', exception)
        preprocessor.samples_2_pickles()
        return fetch_pickles()


def fetch_pickles():
    train_pkl = os.path.join(path, "../data/trainData.pkl")
    dev_pkl = os.path.join(path, "../data/devData.pkl")
    test_pkl = os.path.join(path, "../data/testData.pkl")
    '''
    train_x, train_y, = pickle.load(open(train_pkl,"rb"))
    dev_x, dev_y = pickle.load(open(dev_pkl,"rb"))
    test_x, test_y = pickle.load(open(test_pkl,"rb"))
    '''
    # Mac OS X pickle fixed version

    train_x, train_y, = preprocessor.pickle_load(train_pkl)
    dev_x, dev_y = preprocessor.pickle_load(dev_pkl)
    test_x, test_y = preprocessor.pickle_load(test_pkl)

    print('Pickles loaded.')
    return np.array(train_x), \
           np.array(train_y), \
           np.array(dev_x),   \
           np.array(dev_y),   \
           np.array(test_x),  \
           np.array(test_y)


def save_session(session, my_model_name=None):

    parent_path = os.path.abspath(os.getcwd())
    save_dir = os.path.join(parent_path, '../ai/model_saves/')
    model_name = 'my_model' if my_model_name is None else my_model_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir,model_name)
    saver = tf.train.Saver()
    saver.save(session, save_path)

    print("Trained Model saved in path:",save_path,'\n')

# someTensor.eval = #tf.get_default_session().run(someTensor)


# CPU / GPU Detection


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']