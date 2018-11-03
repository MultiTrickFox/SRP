import ai.creator as create
import ai.trainer as train
import ai.processor as process

import data.preprocess as preprocess


def init():
    print('AI Module Initializing...')
    babyNetwork = create.createLayers()
    thisNetwork = train.startTraining(babyNetwork)
    return thisNetwork #optional return

def ask(input,output_raw=False):
    if not process.isInit():
        print('AI module is not present.')
        init()
    else:
         print('AI module is present.')
    output = process.getOutput(input)
    if not output_raw:
        output = process.raw_to_soft_output(output)
    return output

#additional functions

# def init_further_training(data,label):
#     train.furtherTrain(data,label)

def init_repreprocess():
    preprocess.Samples2Pickles()

def getStats():
    stats = process.getStats()
    return stats

def raw_to_soft(raw_output):
    softOutput = process.raw_to_soft_output(raw_output)
    return softOutput

def soft_to_response(soft_output):
    speechOutput = process.soft_to_speech_output(soft_output)
    return speechOutput


# USAGE

# -> soft_output = ai.ask('A sentence')
# -> raw_output = ai.ask('B sentence',raw_output=True)
# -> soft_output = ai.raw_to_soft(raw_output)
# -> speech_output = ai.soft_to_speech(soft_output)
