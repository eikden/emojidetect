# importing the necessary packages
from PIL import Image, ImageSequence
import cv2
import numpy

# Function for implementing the loading animation
def happy():
    pic_name = 'D:\pythonProject4\happy.gif'
    im = Image.open(pic_name)
    for frame in ImageSequence.Iterator(im):
        frame = frame.convert('RGB')
        cv2_frame = numpy.array(frame)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(pic_name, show_frame)
        cv2.waitKey(100)

def surprise():
    pic_name = 'D:\pythonProject4\surprise.gif'
    im = Image.open(pic_name)
    for frame in ImageSequence.Iterator(im):
        frame = frame.convert('RGB')
        cv2_frame = numpy.array(frame)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(pic_name, show_frame)
        cv2.waitKey(100)



if __name__ == '__main__':
    s=input('happy or surprise')
    if(s=="happy"):
        happy()
    if(s=="surprise"):
        surprise()