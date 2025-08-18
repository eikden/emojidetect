from initial import initial_base
initial_base()

from workflow.interface import *

#test happy emotion
print('Pass happy parameter to test the function.')
detect_emotion("happy", debug=True)
#test surprise emotion
print('Pass surprise parameter to test the function.')
detect_emotion("surprise", debug=True)
#test other emotion
print('Pass other parameter to test the function.')
detect_emotion("other", debug=True)

