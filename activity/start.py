import os
import time
from multiprocessing import Process
from threading import Thread
import tkinter as tk
from PIL import ImageTk, Image

start_w_hitenter = False

actControl = None


def init():
    connectToActivityController()

    showStartGUI()
    actControl.say('Hello')

    if start_w_hitenter:
            actControl.say('Hit any key to continue.')
            _ = input('Hit any key to continue..\n')

    while True:
        actControl.say('What would you like to do next?')
        cmd = actControl.getRealTimeString()
        if 'label' in cmd:
            actControl.initLabelActivity()
        elif 'listen' in cmd:
            actControl.initRecommendActivity()
        elif 'develop' in cmd:
            actControl.say('Hello.')
            time.sleep(0.02)
            actControl.say('Master')
            time.sleep(0.07)
            while True:
                actControl.say('Please choose: Evaluate, Initialize or Stats')
                cmd = actControl.getRealTimeString()
                if 'eval' in cmd:
                    actControl.re_eval_pickles()
                elif 'in' in cmd:
                    actControl.re_init_ai()
                elif 'stat' in cmd:
                    print('>','I am:')
                    result = actControl.getStats()
                    print(result)
                elif 'quit' in cmd:
                    break
        elif ('goodbye' or 'quit') in cmd:
            actControl.say('cya')
            break
        else:
            actControl.say('Unrecognized.')


def showStartGUI():
    root_view = tk.Tk()
    root_view.after(3000,lambda: root_view.quit())
    root_view.title("Welcome")
    root_view.geometry("560x560")
    root_view.wait_visibility(root_view)
    root_view.attributes('-alpha', 0.96)
    parent = os.path.abspath(os.getcwd())
    path = os.path.join(parent,'../res/img/logo.jpg')
    img = ImageTk.PhotoImage(Image.open(path))
    panel = tk.Label(root_view, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    actControl.center_gui(root_view)
    root_view.mainloop()
    root_view.destroy()


def connectToActivityController():
    global actControl
    import controller.activityController
    actControl = controller.activityController

"""
    wait = input('')
        controller.say('I am listening.')
"""
