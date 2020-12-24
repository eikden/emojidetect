#default test environment setup
import os
import sys
from pathlib import Path
current_path = os.getcwd()
dti_path = str(Path(current_path).parent) + '\\ai_engine'

class initial_base:
#This path will use for server environment setup
    def __init__(self):
        sys.path.append(dti_path)

