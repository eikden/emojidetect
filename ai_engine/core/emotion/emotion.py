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
    happy = ai_path+'animoji-lion-emojipedia.gif'
    surprise = ai_path+'animoji-alien-emojipedia.gif'
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
