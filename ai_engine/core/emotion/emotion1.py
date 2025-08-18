# importing the necessary packages
from PIL import Image, ImageSequence
import cv2
import numpy
import os
import sys
import scipy.io.wavfile as wav
from python_speech_features import mfcc

import wave
from sphfile import SPHFile
from pathlib import Path
from glob import glob
import soundfile # to read audio file
import librosa # to extract speech features
import glob
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
#from IPython.display import clear_output, display, Image
import numpy as np

int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

    # we allow only these emotions ( feel free to tune this on your need )
AVAILABLE_EMOTIONS = {
        "angry",
        "disgust",
        "neutral",
}

current_path = os.getcwd()
ai_path = str(Path(current_path).parent) + '\\ai_engine\\core\\emotion\\'

# Function to extract feature from audio file
def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file 'file_name'
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        'features = extract_feature(path, mel=True, mfcc=True)'
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

def load_data(test_size=0.2):
    X, y = [], []

    current_path = os.getcwd()
    mlp_classifier_model_path = current_path+"\\train_model_1\\"

    #change directory
    for file in glob.glob(current_path+"\\train_model_1\\" + "Actor_*\\*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

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

#Group 1 emotion validation checking
def emotion_validation_g1(voice_path):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    
    z = []
    for file in glob.glob(voice_path):
        # extract speech features
        features_z = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        z.append(features_z)

        #correct this to past your train and test data.
        myModel = MLP_Classifier_Model()
        z_pred = myModel.predict(z)

    emotion_status = str(z_pred[0])
    print(emotion_status)
    return emotion_status


def MLP_Classifier_Model():
      
  # load RAVDESS dataset, 99% training 01% testing
    #X_train, X_test, y_train, y_test = load_data(test_size=0.05)

    #model_params = {
    #                'alpha': 0.01,
    #                'batch_size': 256,
    #                'epsilon': 1e-08, 
    #                'hidden_layer_sizes': (500,), 
    #                'learning_rate': 'adaptive', 
    #                'max_iter': 500, 
    #                }

    ## initialize Multi Layer Perceptron classifier
    ## with best parameters ( so far )
    #model = MLPClassifier(**model_params)

    ## train the model
    #print("[*] Training the model...")
    #model.fit(X_train, y_train)

    ## predict 25% of data to measure how good we are
    #y_pred = model.predict(X_test)

    ## calculate the accuracy
    #accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    #print("Accuracy: {:.2f}%".format(accuracy*100))
    
    # now we save the model
    # make result directory if doesn't exist yet
    mlp_classifier_model_path = ai_path+"\\train_model_1\\"

    #pickle.dump(model, open(mlp_classifier_model_path+"mlp_classifier.model", "wb"))
    file = open(mlp_classifier_model_path+"mlp_classifier.model", 'rb')
    model = pickle.load(file)
    file.close()

    return model