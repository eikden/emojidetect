from core.emotion.emotion3 import emojis, get_emotion, emotion_validation
from core.emotion.emotion2 import *
from core.emotion.emotion1 import *
import numpy
import cv2
import PIL
import os

def detect_emotion(emotion, debug=False):
    emoji_path=''
    print('parameter: {0}'.format(emotion))
    emotion = emotion.lower()
    if(emotion=="happy"):
        emoji_path = emojis.happy
        pic_name = get_emotion(emoji_path, debug, 10)
    elif(emotion=="surprise"):
        emoji_path = emojis.surprise
        pic_name = get_emotion(emoji_path, debug, 100)
    elif(emotion=="angry"):
        #do something
        emotion = "angry"
    elif(emotion=="disgust"):
        #do something
        emotion="disgust"
    else:
        emotion="other"
        emoji_path = emojis.other
        pic_name = get_emotion(emoji_path, debug, 100)
        
    return emotion, emoji_path

def process_speech(voice_path):
    emotion_result = []

    #Add group 1 method here to get a return result with proper array
    emotion_result.append("Other")

    #Add group 2 method here to get a return result with proper array
    emotion_result.append("Other")

    #group 3 result
    emotion_result.append(emotion_validation(voice_path))
    #emotion_result.append("happy")
    return emotion_result
