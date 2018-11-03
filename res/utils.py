import speech_recognition as sr
import pygame
import os

# a place for shared stuff

parentPath = os.path.abspath(os.getcwd())
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False
load_dir = os.path.join(parentPath, '../res/audio/')
listening_sound = 'listening.mp3'
listening_path = os.path.join(load_dir, listening_sound)

def center_gui(guiObj):
    guiObj.update_idletasks()
    width = guiObj.winfo_width()
    height = guiObj.winfo_height()
    x = (guiObj.winfo_screenwidth() // 2) - (width // 2)
    y = (guiObj.winfo_screenheight() // 2) - (height // 2)
    guiObj.geometry('{}x{}+{}+{}'.format(width, height, x, y))

def playSound(sound_path):
    sound_path = os.path.abspath(sound_path)
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()

def getRealTimeString():
    with sr.Microphone() as source:
        while True:
            playSound(listening_path)
            audio = recognizer.listen(source,timeout=170)
            try: # using Sphinx
                user_words = recognizer.recognize_google(audio)
                print(user_words)
                os.system('say Sphinx understood you.')
                return user_words
            except sr.UnknownValueError:
                os.system('say Sphinx could not understand you,')
                """
                          +
                    'Trying Google Speech Recognition.')
                try: # using Google Speech Recognition
                    user_words = recognizer.recognize_sphinx(audio)
                    print(user_words)
                    os.system('say Google understood you.')
                    return user_words
                except sr.UnknownValueError:
                    os.system('say Google Speech could not understand you either or not found.')
                """

def speak(speech):
        os.system('say '+speech)

