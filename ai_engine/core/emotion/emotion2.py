import os, os.path
import sys
import numpy
import wavio
from scipy.fftpack import dct
from pathlib import Path

current_path = os.getcwd()
ai_path = str(Path(current_path).parent) + '\\ai_engine\\core\\emotion\\'

def get_mfcc(audio_file):
    print(audio_file)
    wav=wavio.read(audio_file)
    sample_rate=wav.rate
    signal=wav.data.T[0]
    # Read the first 3.5s speech data
    signal=signal[0:int(3.5*sample_rate)]
    # plt.plot(signal)
    # plt.show()


    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    # plt.plot(emphasized_signal)
    # plt.show()


    # Framing
    frame_size=0.025
    frame_stride=0.1
    frame_length,frame_step=frame_size*sample_rate,frame_stride*sample_rate
    signal_length=len(emphasized_signal)
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    num_frames=int(numpy.ceil(float(numpy.abs(signal_length-frame_length))/frame_step))

    pad_signal_length=num_frames*frame_step+frame_length
    z=numpy.zeros((pad_signal_length-signal_length))
    pad_signal=numpy.append(emphasized_signal,z)

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[numpy.mat(indices).astype(numpy.int32, copy=False)]
    # plt.plot(pad_signal)
    # plt.show()


    # Windowing
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation

    # FFT and Power Spectrum
    NFFT = 512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum


    # Mel-Frequency Analysis
    low_freq_mel = 0
    nfilt = 40
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB


    # Mel-Frequency Cepstral Coefficients
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape

    n = numpy.arange(ncoeff)
    cep_lifter =22
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift


    # Normalization
    # filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    # plt.plot(filter_banks)
    # plt.show()
    # plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.2, extent=[0,filter_banks.shape[1],0,filter_banks.shape[0]])
    # plt.show()
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    # plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2, extent=[0,mfcc.shape[0],0,mfcc.shape[1]])
    # plt.show()
    return mfcc


def dtw(M1, M2):
    # Initialize array as the size of M1*M2
    M1_len = len(M1)
    M2_len = len(M2)
    cost = [[0 for i in range(M2_len)] for i in range(M1_len)]

    # Initialize dis array
    dis = []
    for i in range(M1_len):
        dis_row = []
        for j in range(M2_len):
            dis_row.append(distance(M1[i], M2[j]))
        dis.append(dis_row)

    # Initialize cost
    cost[0][0] = dis[0][0]
    for i in range(1, M1_len):
        cost[i][0] = cost[i - 1][0] + dis[i][0]
    for j in range(1, M2_len):
        cost[0][j] = cost[0][j - 1] + dis[0][j]

    # Start dynamic programming
    for i in range(1, M1_len):
        for j in range(1, M2_len):
            cost[i][j] = min(cost[i - 1][j] + dis[i][j] * 1,
                             cost[i - 1][j - 1] + dis[i][j] * 2,
                             cost[i][j - 1] + dis[i][j] * 1)
    return cost[M1_len - 1][M2_len - 1]


# Compute the distance between two vectors
def distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum = sum + abs(x1[i] - x2[i])
    return sum


sad_models = []
fear_models = []
sad_mfcc = []
fear_mfcc = []
sad_emot = []
fear_emot = []


#ref_dir = os.listdir(ai_path + '\\train_model_2\\sad_ref\\')
#for file in ref_dir:
#    sad_models.append(file)
#    sad_mfcc.append(get_mfcc(ref_dir + file))

#    actor = file.split('-')[-1].split('.')[0]
#    if actor == '10' or actor == '15':
#        sad_emot.append('weak sad')
#    elif actor == '07' or actor == '24':
#        sad_emot.append('mid sad')
#    elif actor == '03' or actor == '06':
#        sad_emot.append('very sad')


def model():
    ref_dir = os.listdir(ai_path + '\\train_model_2\\fear_ref\\')
    print('fear path looping: {0}'.format(ref_dir))
    for file in ref_dir:
        fear_models.append(file)
        print('filename: {0}'.format(file))
        fullfilepath = ai_path + '\\train_model_2\\fear_ref\\' + file
        print('fullpath: {0}'.format(fullfilepath))
        fear_mfcc.append(get_mfcc(fullfilepath))

    ref_dir = os.listdir(ai_path + '\\train_model_2\\sad_ref\\')
    print('sad path looping: {0}'.format(ref_dir))
    for file in ref_dir:
        sad_models.append(file)
        print('filename: {0}'.format(file))
        fullfilepath = ai_path + '\\train_model_2\\sad_ref\\' + file
        sad_mfcc.append(get_mfcc(fullfilepath))

    return fear_mfcc, sad_mfcc

#    actor = file.split('-')[-1].split('.')[0]
#    if actor == '08':
#        fear_emot.append('weak fear')
#    elif actor == '03':
#        fear_emot.append('mid fear')
#    elif actor == '20' or actor == '06':
#        fear_emot.append('very fear')


def match(tar_path):
    flag = 0
    print('Executing match process for group 2')
    print('tar_path: {0}'.format(tar_path))
    #if tar_path.split('-')[2] == '04':
    tar_mfcc = get_mfcc(tar_path)
    print('debug 1')
    fear_mfcc, sad_mfcc = model()
    print('model load')
    mfcc = fear_mfcc.append(sad_mfcc)
    print('combine both fear and sad model')
    min_dis = dtw(tar_mfcc, mfcc[0])
    print('find the dtw value: {0}'.format(min_dis))
    for i in range(0, len(mfcc)):
        dis = dtw(tar_mfcc, mfcc[i])
        if dis < min_dis:
            min_dis = dis
            flag = i

    emotion_status = ''
    if flag =="08":
        emotion_status = 'weak fear'
    elif flag=="03": 
        emotion_status = 'mid fear'
    elif flag=="20" or flag=="06":
        emotion_status = 'very fear'
    elif flag=="07" or flag=="24":
        emotion_status = 'weak sad'
    elif flag=="03" or flag=="06":
        emotion_status = 'mid sad'
    elif flag=="07" or flag=="24":
        emotion_status = 'very sad'
    else:
        emotion_status = 'other'

    return emotion_status


