from core.emotion.emotion import emojis, get_emotion, group_3_emotion_validation
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
    emotion_result.append(group_3_emotion_validation(voice_path))
    
    return emotion_result
