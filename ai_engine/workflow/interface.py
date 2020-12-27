from core.emotion.emotion import emojis
import numpy
import cv2
import PIL
import os

def detect_emotion(emotion):
    if(emotion=="happy"):
        happy = emojis.happy
        pic_name, cv2_frame = emojis.get_emotion(happy)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Window', show_frame)
    elif(emotion=="surprise"):
        surprise = emojis.surprise
        pic_name, cv2_frame = emojis.get_emotion(surprise)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Window', show_frame)
    else:
        other = emojis.other
        pic_name, cv2_frame = emojis.get_emotion(other)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Window', show_frame)

    key = cv2.waitKey(3000)#pauses for 3 seconds before fetching next image
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()