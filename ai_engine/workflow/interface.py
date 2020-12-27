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
        pic_name, cv2_frame = get_emotion(emoji_path)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)    
    elif(emotion=="surprise"):
        emoji_path = emojis.surprise
        pic_name, cv2_frame = get_emotion(emoji_path)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
    else:
        emoji_path = emojis.other
        pic_name, cv2_frame = get_emotion(emoji_path)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Frame', show_frame)
    #pauses for 3 seconds before fetching next image
    key = cv2.waitKey(3000)
    
    #if key == 27:#if ESC is pressed, exit loop
    #    cv2.destroyAllWindows()
    return emoji_path