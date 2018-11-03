'''
import sys
sys.path.append('../')
'''

import os
import controller.aiController as ai
parentPath = os.path.abspath(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # here4debugpurposes
# import tensorflow as tf
# import ai.creator as create
# import ai.processor as process
# import ai.trainer as train
import data.preprocess as preprocess

#preprocess.Samples2Pickles()
#ai.init()


while True: print('>',ai.ask(input('\nAsk me..\n')))
'''

#UNUSUED -- demo purposes

sentence = 'Im sad to see you leave'

networks = []

untrained_net = create.createLayers()
networks.append(untrained_net)

#trained_net = ai.init()
#networks.append(trained_net)

with tf.Session() as sess:
    x = tf.placeholder(tf.float32)
    realInput = preprocess.human2aiConverter(sentence)
    output = process.neuralFunction(x,untrained_net)
    output = tf.nn.softmax(output)

    sess.run(tf.global_variables_initializer())

    print('Here is the result of UNTRAINED NET:')
    print(sess.run(output,{x:[realInput]}))
    print(sess.run(output,{x:[realInput]}))
    print(sess.run(output,{x:[realInput]}))
    print(sess.run(output,{x:[realInput]}))
    print(sess.run(output,{x:[realInput]}))

print('Here is the result of TRAINED NET:')
print(ai.ask(sentence))
print(ai.ask(sentence))
print(ai.ask(sentence))
print(ai.ask(sentence))
print('')
while True:


'''












'''







'''
import time
import speech_recognition as sr


os.system('say Hello')
#activity.init() #tod: unlock --- did.

# bu kalanlar full activityRecommend'in icinde olucak
import controller.activityController
load_dir = os.path.join(parentPath, '../res/audio/')
listening_sound = 'listening.wav'
listening_path = os.path.join(load_dir, listening_sound)
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False
time.sleep(0.07)
os.system('say as humans would say, tell me about your self..')
with sr.Microphone() as source:
    while True:
        # todo: play the sound actually
        activityController.playSound(listening_path)
        audio = recognizer.listen(source,timeout=120)
        try: # using Sphinx
            user_words = recognizer.recognize_google(audio)
            print(user_words)
            os.system('say Sphinx understood you.')
            response = ai.ask(user_words)
            print('>',response)
            speech = ai.soft_to_speech(response)
            os.system('say'+' '+speech)
            #return response #activate at the recommendActivity
        except sr.UnknownValueError:
            os.system('say Sphinx could not understand you,'+
                      'Trying Google Speech Recognition.')
            try: # using Google Speech Recognition
                user_words = recognizer.recognize_sphinx(audio)
                print(user_words)
                os.system('say Sphinx understood you.')
                response = ai.ask(user_words)
                print('>',response)
                speech = ai.soft_to_speech(response)
                os.system('say'+' '+speech)
                #return response #activate at the recommendActivity
            except sr.UnknownValueError:
                os.system('say Google Speech could not understand you either or not found.')

        wait = input('')
        os.system('say I am listening.')
