import activity.start as startActivity
import activity.labelYourMusic as labelActivity
import activity.recommendMe as recommendActivity

import controller.aiController as ai
import data.preprocess as preprocess

import res.utils as utils
import os

parentPath = os.path.abspath(os.getcwd())

def init():
    startActivity.init()

def initStartActivity():
    startActivity.init()

def initLabelActivity():
    labelActivity.init()

def initRecommendActivity():
    recommendActivity.init()

def askAI(input):
    return ai.ask(input)

def askAI_raw(input):
    return ai.ask(input,output_raw=True)

def askAI_verbal(input):
    return ai.soft_to_response(ai.ask(input))

def getStatsAI():
    return ai.getStats()

def raw2soft(raw):
    return ai.raw_to_soft(raw)

def soft2verb(soft):
    return ai.soft_to_response(soft)

def re_eval_pickles():
    preprocess.samples_2_pickles()

def re_init_ai():
    ai.init()

def center_gui(guiObj):
    utils.center_gui(guiObj)

def say(string):
    utils.speak(string)

def playSound(sound_path):
    utils.playSound(sound_path)

def getRealTimeString():
    return utils.getRealTimeString()
