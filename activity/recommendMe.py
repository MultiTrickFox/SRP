import random
import time
import os
from glob import glob

actControl = None

parentPath = os.path.abspath(os.getcwd())


def init(): #extension: meta data olarak sarkinin neresinde markladigini kaydet, o kismi analiz et?
    connectToActivityController()
    play_continuous = False

    actControl.say('Would you like to loop')
    user_words = actControl.getRealTimeString()
    if 'yes' in user_words: play_continuous = True
    time.sleep(1)

    actControl.say('As humans would say, tell me about your self.')

    while True:
        actControl.say('Please be descriptive.')
        time.sleep(0.7)
        user_words = actControl.getRealTimeString()
        if 'label' in user_words:
            actControl.initLabelActivity()
        else:
            output = getResponse(user_words)
            song_path = fetchSong(output)
            if song_path is not None:
                actControl.say('I have prepared your song.')
                time.sleep(0.4)
                actControl.say('Good bye')
                actControl.playSound(song_path)
            else:
                actControl.say('I did not find any saved music, for this emotion.')

            _ = input('')
            if play_continuous:
                while True:
                    next_song = fetchSong(output)
                    actControl.playSound(next_song)
                    _ = input('')
            else:
                actControl.say('I am listening.')


def getResponse(input):
    raw = actControl.askAI_raw(input)
    soft = actControl.raw2soft(raw)
    actControl.say(actControl.soft2verb(soft))
    return soft


def fetchSong(output):
    # expansion: o birine de baska ai karar vericek => this ai'dan 2 olasilik al.
    path_to_songs = os.path.join(parentPath,"../data/user/mymusic/class"+str(output))
    path_to_songs = os.path.abspath(path_to_songs)
    print(path_to_songs) #todo: debug purp
    songs = glob(path_to_songs+'/*.mp3')
    #allfiles = os.listdir(path_to_songs)
    #songs = [ fname for fname in allfiles if fname.endswith('.mp3')]
    print(songs) #todo: debug purp
    if songs == []:
        return None
    else:
        selection = random.choice(songs)
        path_to_song = os.path.join(path_to_songs,selection)
        print(path_to_song) #todo: debug purp
        return path_to_song


def connectToActivityController():
    global actControl
    import controller.activityController
    actControl = controller.activityController
