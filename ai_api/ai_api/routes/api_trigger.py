from workflow.interface import *
import json
from flask import current_app as app
from flask import request
from ai_api.decorators.login_required import login_required
from ai_api.response.response import Success, Failure

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


