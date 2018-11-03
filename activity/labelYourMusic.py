from tkinter import *
from tkinter import filedialog
import os
import shutil
import pickle
import random as rand

actControl = None

descList = []
currDescList = []
descButtons = []
optButtons = []
pickedDescs = ''

amount_desc_display = 25


def init():
    connectToActivityController()

    global descList
    global currDescList
    global descButtons
    global optButtons
    global pickedDescs

    descList = []
    currDescList = []
    descButtons = []
    optButtons = []
    pickedDescs = ''

    descList = loadDescList()
    currDescList = loadCurrDescList()

    pickedSongPath = pickSong()
    desc_view = pickDescs()

    desc_view.destroy()
    updateUserPrefs(pickedSongPath,pickedDescs)

    return None


def loadDescList():
    path = os.path.abspath(os.getcwd())
    inputPickle = os.path.join(path, "../data/inputArray.pkl")
    with open(inputPickle,'rb') as pklRead:
        return pickle.load(pklRead)


def pickSong():
    root_view = Tk()
    root_view.withdraw()
    root_view.update()
    try: pickedSong = filedialog.askopenfilename()
    except: print('',end='')
    print('Obtained:',pickedSong)

    return pickedSong


def pickDescs():
    global desc_view_open
    global currDescList
    global descButtons

    root_view = Tk()
    root_view.title("Selection Pane")
    root_view.geometry("600x260")
    root_view.wait_visibility(root_view)
    root_view.attributes('-alpha', 0.94)
    actControl.center_gui(root_view)

    labels_frame = Frame(root_view)
    options_frame = Frame(root_view)

    print(currDescList) #todo: debug purposes here.

    initDescriptorGUI(labels_frame)
    initOptionsGUI(options_frame,root_view)

    labels_frame.grid(column=0,columnspan=1)
    options_frame.grid(column=1,columnspan=2)
    labels_frame.grid_rowconfigure(0, weight=1)
    labels_frame.grid_columnconfigure(0, weight=1)
    desc_view_open = True

    optButtons[0].focus() #voice inp bt
    root_view.mainloop()

    return root_view


def updateUserPrefs(song_path,descriptor):
    actControl.say('Updating.')
    song_type = actControl.askAI(descriptor)
    path = os.path.abspath(os.getcwd())
    save_path = os.path.join(path, "../data/user/mymusic/class"+str(song_type))
    #os.rename(path,save_path)
    shutil.move(song_path,save_path)
    actControl.say('Your preferences are saved.')


def buttonRefreshClicked():
    global descList
    global currDescList

    for _,button in enumerate(descButtons):
        changeTo = rand.randint(0,len(descList)-1)
        currDescList[_] = descList[changeTo]
        button['text'] = currDescList[_]


def buttonDescClicked(buttonNr):
    global pickedDescs
    global descButtons
    global descList
    global currDescList

    print(buttonNr) #todo:debug purposes here
    thisButton = descButtons[buttonNr]
    thisDesc = thisButton['text']
    pickedDescs += (thisDesc + ' ')
    changeTo = rand.randint(0,len(descList))
    print(changeTo) #todo: debug purposes here
    thisButton['text'] = descList[changeTo]
    currDescList[buttonNr] = descList[changeTo]
    print(thisDesc) #todo: debug purposes here
    print(pickedDescs) #todo: debug purposes here
    print(thisButton['text'])


def loadCurrDescList():
    global descList
    global amount_desc_display

    copy_descs = descList
    rand.shuffle(copy_descs)
    return copy_descs[:amount_desc_display]


def takeVoiceInput():
    global pickedDescs
    user_words = actControl.getRealTimeString()
    pickedDescs += (user_words+' ')
    global voice_input_label
    voice_input_label['text'] = user_words[:60]+'...'
    actControl.say('Your words are marked.')


def connectToActivityController():
    global actControl
    import controller.activityController
    actControl = controller.activityController


def initOptionsGUI(options_frame,root):
    buttonVoiceInput = Button(options_frame,text='Voice Input',command= lambda : takeVoiceInput())
    optButtons.append(buttonVoiceInput)
    buttonVoiceInput.grid()
    buttonRefresh = Button(options_frame,text='Refresh',command= lambda : buttonRefreshClicked())
    optButtons.append(buttonRefresh)
    buttonRefresh.grid()
    buttonContinue = Button(options_frame,text='Continue',command = lambda : root.quit())
    optButtons.append(buttonContinue)
    buttonContinue.grid()


def initDescriptorGUI(labels_frame): #wtf is this shit bro
    ''' ty for not working.
    for i in range(5):
        for j in range(5):
            id = i+j
            this_desc = str(currDescList[id])
            thisButton = Button(labels_frame,text=this_desc,width=8)
            thisButton.bind(lambda : buttonDescClicked(id))
            descButtons.append(thisButton)
            thisButton.grid(row=i,column=j,sticky='nsew')
    :param labels_frame:
    :return: descButtons.append(  ---> problem is here: how to send button related arguments to another function param??
                Button(labels_frame, text=this_desc, command=lambda : buttonDescClicked(i+j)).grid(row=i, column=j))
    '''

    button0 = Button(labels_frame,text=currDescList[0],command=lambda : buttonDescClicked(0),width=8)
    descButtons.append(button0)
    button0.grid(row=0,column=0,sticky='nsew')
    button1 = Button(labels_frame,text=currDescList[1],command=lambda : buttonDescClicked(1),width=8)
    descButtons.append(button1)
    button1.grid(row=1,column=0,sticky='nsew')
    button2 = Button(labels_frame,text=currDescList[2],command=lambda : buttonDescClicked(2),width=8)
    descButtons.append(button2)
    button2.grid(row=2,column=0,sticky='nsew')
    button3 = Button(labels_frame,text=currDescList[3],command=lambda : buttonDescClicked(3),width=8)
    descButtons.append(button3)
    button3.grid(row=3,column=0,sticky='nsew')
    button4 = Button(labels_frame,text=currDescList[4],command=lambda : buttonDescClicked(4),width=8)
    descButtons.append(button4)
    button4.grid(row=4,column=0,sticky='nsew')

    button5 = Button(labels_frame,text=currDescList[5],command=lambda : buttonDescClicked(5),width=8)
    descButtons.append(button5)
    button5.grid(row=0,column=1,sticky='nsew')
    button6 = Button(labels_frame,text=currDescList[6],command=lambda : buttonDescClicked(6),width=8)
    descButtons.append(button6)
    button6.grid(row=1,column=1,sticky='nsew')
    button7 = Button(labels_frame,text=currDescList[7],command=lambda : buttonDescClicked(7),width=8)
    descButtons.append(button7)
    button7.grid(row=2,column=1,sticky='nsew')
    button8 = Button(labels_frame,text=currDescList[8],command=lambda : buttonDescClicked(8),width=8)
    descButtons.append(button8)
    button8.grid(row=3,column=1,sticky='nsew')
    button9 = Button(labels_frame,text=currDescList[9],command=lambda : buttonDescClicked(9),width=8)
    descButtons.append(button9)
    button9.grid(row=4,column=1,sticky='nsew')

    button10 = Button(labels_frame,text=currDescList[10],command=lambda : buttonDescClicked(10),width=8)
    descButtons.append(button10)
    button10.grid(row=0,column=2,sticky=N+S+E+W)
    button11 = Button(labels_frame,text=currDescList[11],command=lambda : buttonDescClicked(11),width=8)
    descButtons.append(button11)
    button11.grid(row=1,column=2,sticky=N+S+E+W)
    button12 = Button(labels_frame,text=currDescList[12],command=lambda : buttonDescClicked(12),width=8)
    descButtons.append(button12)
    button12.grid(row=2,column=2,sticky=N+S+E+W)
    button13 = Button(labels_frame,text=currDescList[13],command=lambda : buttonDescClicked(13),width=8)
    descButtons.append(button13)
    button13.grid(row=3,column=2,sticky=N+S+E+W)
    button14 = Button(labels_frame,text=currDescList[14],command=lambda : buttonDescClicked(14),width=8)
    descButtons.append(button14)
    button14.grid(row=4,column=2,sticky=N+S+E+W)

    button15 = Button(labels_frame,text=currDescList[15],command=lambda : buttonDescClicked(15),width=8)
    descButtons.append(button15)
    button15.grid(row=0,column=3,sticky=N+S+E+W)
    button16 = Button(labels_frame,text=currDescList[16],command=lambda : buttonDescClicked(16),width=8)
    descButtons.append(button16)
    button16.grid(row=1,column=3,sticky=N+S+E+W)
    button17 = Button(labels_frame,text=currDescList[17],command=lambda : buttonDescClicked(17),width=8)
    descButtons.append(button17)
    button17.grid(row=2,column=3,sticky=N+S+E+W)
    button18 = Button(labels_frame,text=currDescList[18],command=lambda : buttonDescClicked(18),width=8)
    descButtons.append(button18)
    button18.grid(row=3,column=3,sticky=N+S+E+W)
    button19 = Button(labels_frame,text=currDescList[19],command=lambda : buttonDescClicked(19),width=8)
    descButtons.append(button19)
    button19.grid(row=4,column=3,sticky=N+S+E+W)

    button20 = Button(labels_frame,text=currDescList[20],command=lambda : buttonDescClicked(20),width=8)
    descButtons.append(button20)
    button20.grid(row=0,column=4,sticky=N+S+E+W)
    button21 = Button(labels_frame,text=currDescList[21],command=lambda : buttonDescClicked(21),width=8)
    descButtons.append(button21)
    button21.grid(row=1,column=4,sticky=N+S+E+W)
    button22 = Button(labels_frame,text=currDescList[22],command=lambda : buttonDescClicked(22),width=8)
    descButtons.append(button22)
    button22.grid(row=2,column=4,sticky=N+S+E+W)
    button23 = Button(labels_frame,text=currDescList[23],command=lambda : buttonDescClicked(23),width=8)
    descButtons.append(button23)
    button23.grid(row=3,column=4,sticky=N+S+E+W)
    button24 = Button(labels_frame,text=currDescList[24],command=lambda : buttonDescClicked(24),width=8)
    descButtons.append(button24)
    button24.grid(row=4,column=4,sticky=N+S+E+W)

    global voice_input_label
    voice_input_label = Label(labels_frame)
    voice_input_label.grid(row=5,columnspan=4)
