# importing the necessary packages
from PIL import Image, ImageSequence
import cv2
import numpy
import os
import sys
from pathlib import Path
current_path = os.getcwd()
ai_path = str(Path(current_path).parent) + '\\ai_engine\\core\\emotion\\'

#define an enum to managing emoji path
class emojis():
    happy = ai_path+'beaming_face_with_smiling_eyes_128.gif'
    surprise = ai_path+'money_mouth_face_128.gif'
    other = ai_path+'thinking_face_128.gif'

# Function for implementing the loading animation
def get_emotion(emoji):
    pic_name = emoji
    im = Image.open(pic_name)
    for frame in ImageSequence.Iterator(im):
        frame = frame.convert('RGB')
        cv2_frame = numpy.array(frame)
    return pic_name, cv2_frame
