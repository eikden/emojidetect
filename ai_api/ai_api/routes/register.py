from flask import current_app as app
from flask import request
import json
from ai_api.models.User import create_user, user_change_password
from ai_api.response.response import Success, Failure

# register account
@app.route('/register', methods=['POST'])
def register():
  first_name = request.form['first_name']
  last_name = request.form['last_name']
  email = request.form['email']
  password = request.form['password']
  user, error, token = create_user(first_name, last_name, email, password)
  if not error:
    message = Success({
      'status': 'sucess',
      'message': 'Successfully registered.',
      'auth_token': token,
    })
    return message
  return Failure(error)

# change password for api account
@app.route('/change_password', methods=['PUT'])
def change_password():
    email=request.form['email']
    old_password = request.form['old_password']
    new_password = request.form['new_password']
    status, error = user_change_password(email, old_password, new_password)
    if not error:
        message = Success({'status':'success',
                           'message':'Successfully password changed.'})
        return message
    return Failure(error)
