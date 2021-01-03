from workflow.interface import *
import json
from flask import current_app as app
from flask import request
from ai_api.decorators.login_required import login_required
from ai_api.response.response import Success, Failure
import sys
import os
from pathlib import Path

current_path = os.path.dirname(os.path.abspath(__file__))
dti_path = str(Path(current_path).parent.parent.parent)


@app.route('/detect_emoji', methods = ['POST'])
#@login_required
def detect_emoji():
  try:
    emoji_input = request.form['emoji_input']
    print('Emoji Input: {0}'.format(emoji_input))
    emoji_image = detect_emotion(emoji_input)
    print('Emoji path: {0}'.format(emoji_image))

    return Success({'message':'Completed.', 'data': emoji_image})
  except Exception as error:
    return Failure({ 'message': str(error) }, debug = True )


@app.route("/upload_speech", methods=['POST'])
def upload_speech():
    try:
        if request.method == "POST":
            print(dti_path)
            f = open(dti_path+'\\ai_engine\\data\\file.wav', 'wb')
            f.write(request.get_data("audio_data"))
            f.close()
            #if os.path.isfile('./file.wav'):
            #    print("./file.wav exists")

            #status = process_speech(dti_path++'\\ai_engine\\data\\file.wav')
        return Success({'message':'Completed.', 'data': 'success'})
    except Exception as error:
        return Failure({ 'message': str(error) }, debug = True )
