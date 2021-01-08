# importing the necessary packages
from PIL import Image, ImageSequence
import cv2
import numpy
import os
import sys
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np
import wave
from sphfile import SPHFile
from pathlib import Path
from glob import glob

current_path = os.getcwd()
ai_path = str(Path(current_path).parent) + '\\ai_engine\\core\\emotion\\'

#define an enum to managing emoji path
class emojis():
    happy = ai_path+'animoji-lion-emojipedia.gif'
    surprise = ai_path+'animoji-alien-emojipedia.gif'
    angry = ai_path+''
    disgust = ai_path+''
    other = ai_path+'giphy.gif'

# Function for implementing the loading animation
def get_emotion(emoji, debug, duration):
    pic_name = emoji
    im = Image.open(pic_name)
    if debug==True:
        for frame in ImageSequence.Iterator(im):
            frame = frame.convert('RGB')
            cv2_frame = numpy.array(frame)
            show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)    
            cv2.imshow(pic_name, show_frame)
            #pauses for depend on duration parameter before fetching next image
            key = cv2.waitKey(duration)

    return pic_name

#Group 3 emotion validation checking
def emotion_validation(voice_path):
    #emotion = "happy"
    #write your logic here
    #record()
    model = Model()
    print('model: {0}'.format('Initial model'))
    my_mfcc = My_Mfcc(voice_path)
    print('read: {0}'.format('Read model'))
    flag = 0
    min_dis = DTW(my_mfcc, model[0])
    for i in range(1, len(model)):
        dis = DTW(my_mfcc, model[i])
        if min_dis > dis:
            min_dis = dis
            flag = i

    emotion_status = ''
    if flag == 0:
        emotion_status = 'happy'
    elif flag == 1:
        emotion_status = 'surprise'
    elif flag == 2:
        emotion_status = 'happy'
    else:
        emotion_status = 'other'
   
    print('Emotion status: {0}'.format(emotion_status))
    #return result have to be in string where the emotion that you setup for front-end must be align. 
    return emotion_status

def Model():
    model = []
    for i in range(1, 5):
        fs, audio = wav.read(ai_path+"\\train_model\\"+str(i)+".wav")
        feature_mfcc = mfcc(audio, samplerate=fs)
        model.append(feature_mfcc)
    return model

def My_Mfcc(voice_path):
    print('voice path: {0}'.format(voice_path))
    fs, audio = wav.read(voice_path)
    feature_mfcc = mfcc(audio, samplerate=fs)
    return feature_mfcc

def DTW(mfcc_1,mfcc_2):
    l1 = len(mfcc_1)
    l2 = len(mfcc_2)

    dis = []
    for i in range(l1):
        d = []
        for j in range(l2):
            d.append(Distance(mfcc_1[i], mfcc_2[j]))
        dis.append(d)

    cost = np.zeros((l1,l2))
    cost[0][0] = dis[0][0]
    for i in range(1, l1):
        cost[i][0] = cost[i-1][0] + dis[i][0]
    for i in range(1, l2):
        cost[0][i] = cost[0][i-1] + dis[0][i]

    for i in range(1, l1):
        for j in range(1, l2):
            cost[i][j] = min(cost[i][j-1] + dis[i][j],
                             cost[i-1][j-1]+ dis[i][j]*2,
                             cost[i-1][j] + dis[i][j])
    return cost[l1-1][l2-1]

def Distance(x, y):
    dis = 0
    for i in range(len(x)):
        dis += (x[i]-y[i])**2
    dis = np.sqrt(dis)
    return dis





