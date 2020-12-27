from core.emotion.emotion import emojis, get_emotion
import numpy
import cv2
import PIL
import os

def detect_emotion(emotion):
    emoji_path=''
    print('parameter: {0}'.format(emotion))
    if(emotion=="happy"):
        emoji_path = emojis.happy
        pic_name = get_emotion(emoji_path)
    elif(emotion=="surprise"):
        emoji_path = emojis.surprise
        pic_name = get_emotion(emoji_path)
    else:
        emoji_path = emojis.other
        pic_name = get_emotion(emoji_path)
        
    return emoji_path