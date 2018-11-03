import pickle
import random
import os
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

tokenizer = word_tokenize
lemmatizer = WordNetLemmatizer().lemmatize
path = os.path.abspath(os.getcwd())


numClasses = 4
ratioTrain = 0.8
ratioDev = 0.1
ratioTest = 0.1

# max_lines_to_read = 7500

min_freq = 0.397  #todos
max_freq = 0.4026 #todos


filesDefaultIdentifs = []
filesClassSamples = [] #,numClasses = getClassSamples()
for i in range(1,numClasses+1):
    currentDefaultFile = os.path.join(path, '../data/defaultIdentifs/class' + str(i) + '.txt')
    filesDefaultIdentifs.append(currentDefaultFile)
    currentClassFile = os.path.join(path, '../data/classSamples/class' + str(i) + '.txt')
    filesClassSamples.append(currentClassFile)


def samples_2_pickles():
    print('Preprocessing samples..')
    
    inputVector = createInputVector()
    dataArray = createDataArray(inputVector)

    random.shuffle(dataArray)
    dataArray = np.array(dataArray)

    dumpInputVector(inputVector)
    dumpData(dataArray)
    print('Preprocess completed.\n')


def createInputVector():
    inputArray = calcPossibleInputs()
    inputArrayUnique = calcUniqueInputs(inputArray)
    inputVector = appendPredefinedInputs(inputArrayUnique)
    return inputVector


def calcPossibleInputs():
    inputArray = []

    for n, file in enumerate(filesClassSamples):
        with open(file,'r') as fileReader:
            lines = fileReader.readlines()
            for line in lines:
                sentence = line.lower()
                words = tokenizer(sentence)
                for word in words:
                    entry = lemmatizer(word)
                    inputArray.append(entry)
    return inputArray


def calcUniqueInputs(inputArray):
    mapWordCount = Counter(inputArray)
    #print(mapWordCount) #4 debug
    sortedCount = sorted(mapWordCount, key = mapWordCount.get, reverse = False)
    print(sortedCount)
    size = len(sortedCount)
    min = int(min_freq * size)
    max = int(max_freq * size)
    #print('Total entry size:',size) # 4 debug
    #print(min,max) # 4 debug
    inputArrayUnique = sortedCount[min:max]
    #print('all_entries:',len(sortedCount),'\n',sortedCount) #4 debug
    print('uniq_entries:',len(inputArrayUnique),'\n',inputArrayUnique) #4 debug
    print('uniq_entries_rev:',len(inputArrayUnique),'\n',list(reversed(inputArrayUnique))) #4 debug
    #print('Unique entries calculated: ', len(inputArrayUnique))
    return inputArrayUnique                                         #logger([array,of,variables],lwl=4)


def appendPredefinedInputs(inputArrayUnique):
    for file in filesDefaultIdentifs:
        with open(file,'r') as fileReader:
            lines = fileReader.readlines()
            for line in lines:
                sentence = line.lower()
                words = tokenizer(sentence)
                for word in words:
                    entry = lemmatizer(word)
                    if entry not in inputArrayUnique:
                        inputArrayUnique.append(entry)
    print('Input Vector Loaded. Size:',len(inputArrayUnique))
    return inputArrayUnique


def createDataArray(inputVector):
    """
    featureSet (dataArray) format:
    [
     [feature, label]
     [[0,1,1,1], [0,1]],
     [[0,1,1,0], -> sizeLexiconShortest [1,0]] -> sizeClass,
     ...
    ]
    """
    dataArray = []
    labelColdArray = np.zeros(numClasses)
    for i in range(0, numClasses):
        currentLabelArray = labelColdArray
        currentLabelArray[i] += 1
        dataArray += singleClassToArray(filesClassSamples[i], currentLabelArray, inputVector)
        print('Class',i+1,'processed.')
    return dataArray


def singleClassToArray(samplesFile, samplesClassification, inputVector):
    currentFeatureSet = []
    # samples = random_sampler(samplesFile, max_lines_to_read)
    for line in samplesFile:
        # print(line,'\n') #log dis. to lvl 5
        currentFeatures = np.zeros(len(inputVector))
        line = line.lower()
        inputs = tokenizer(line)
        for input in inputs:
            possibleInput = lemmatizer(input)
            if possibleInput in inputVector:
                inputIndex = inputVector.index(possibleInput)
                currentFeatures[inputIndex] += 1
        currentFeatureSet.append([currentFeatures, samplesClassification])
    return currentFeatureSet


def dumpInputVector(inputArrayUnique):
    inputPickle = os.path.join(path, "../data/inputArray.pkl")
    with(open(inputPickle,'wb')) as file:
        pickle.dump(inputArrayUnique,file)

    inputText = os.path.join(path, "../data/inputList.txt") # for frequency fixing purposes.
    with open(inputText,'w') as fileWriter:
        for entry in inputArrayUnique:
            fileWriter.write(entry+'\n')


def dumpData(dataArray):
    totalSize = len(dataArray)
    trainSize = int(ratioTrain*totalSize)
    devSize = int(ratioTest*totalSize)
    testSize = int(ratioDev*totalSize)

    trainX = list(dataArray[:, 0][:trainSize])
    trainY = list(dataArray[:, 1][:trainSize])
    devX = list(dataArray[:, 0][trainSize:trainSize+devSize])
    devY = list(dataArray[:, 1][trainSize:trainSize+devSize])
    testX = list(dataArray[:, 0][-testSize:])
    testY = list(dataArray[:, 1][-testSize:])

    trainPicklePath = os.path.join(path, "../data/trainData.pkl")
    devPicklePath = os.path.join(path, "../data/devData.pkl")
    testPicklePath = os.path.join(path, "../data/testData.pkl")

    '''
    with open(trainPicklePath, 'wb') as file:
        pickle.dump([trainX, trainY], file)
        print('training pickle dumped.')
    with open(devPicklePath, 'wb') as file:
        pickle.dump([devX, devY], file)
        print('development pickle dumped.')
    with open(testPicklePath, 'wb') as file:
        pickle.dump([testX, testY], file)
        print('testing pickle dumped.')
    '''

    # Mac OS X pickle fixed version

    pickle_dump([trainX, trainY],trainPicklePath)
    print('training pickle dumped.')
    pickle_dump([devX, devY],devPicklePath)
    print('development pickle dumped.')
    pickle_dump([testX, testY],testPicklePath)
    print('testing pickle dumped.')


def random_sampler(filename, k):
    with open(filename,'r') as f:
        lines = f.readlines()
        samples = random.sample(lines,k)
        for i,line in enumerate(samples):
            samples[i] = line.replace('\n',' ')

    return samples


def human2aiConverter(input):
    inputArrayPickle = os.path.join(path, "../data/inputArray.pkl")
    inputArray = pickle.load(open(inputArrayPickle,'rb'))
    #print(input) #log dis.
    sentence = input.lower()
    words = tokenizer(sentence)
    actualInput = np.zeros(len(inputArray))
    for word in words:
        entry = lemmatizer(word)
        if entry in inputArray:
            index = inputArray.index(entry)
            actualInput[index] = 1
    return actualInput


def getIOdims():
    try:
        inputArrayPickle = os.path.join(path, "../data/inputArray.pkl")
        inputArray = pickle.load(open(inputArrayPickle,'rb'))
    except:
        print('No Input Vector found.')
        samples_2_pickles()
        inputArrayPickle = os.path.join(path, "../data/inputArray.pkl")
        inputArray = pickle.load(open(inputArrayPickle,'rb'))
    inputSize = len(inputArray)
    return inputSize,numClasses

# rest is for Python Pickle OSError: [Errno 22] on Mac OS X - cannot dump pickles larger than 4GB, so we buffer them.

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        # print("writing total_bytes=%s..." % n, flush=True) #log dis.
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            # print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            # print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


if __name__ == '__main__':
    samples_2_pickles()
    print('done.')