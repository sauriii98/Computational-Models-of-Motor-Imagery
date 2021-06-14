import numpy as np
import scipy.io
import pandas as pd
from scipy import fft
from scipy import signal
import matplotlib.pyplot as plt
import os



files = os.listdir('Datasets')
file_names = []
for f in files:
    
    file_name = 'Datasets/'+f.split('.')[0]
    if file_name not in file_names:
        file_names.append(file_name)


def take_input(filepath):
  mat = scipy.io.loadmat(filepath)
  return mat

def get_markers(mat):
  markers = mat['o'][0][0][4]
  markers = np.reshape(markers, (markers.shape[0],))
  return markers

def get_data(mat):
  data = mat['o'][0][0][6]
  data = np.delete(data, -1, axis=1)
  return data

def get_trial_frames(data, markers, class_label):
  frame = [data[i:i+170] for i in range(1, len(markers)) if markers[i] == class_label and markers[i-1] == 0]
  frame = np.array(frame)
  frame = np.transpose(frame,(0,2,1))
  return frame

def design_filter(order, fs, cutoff_freq):
  low_pass_filter = signal.butter(order, cutoff_freq, fs=fs, output='sos')
  return low_pass_filter

def apply_filter(filter, frame, axis=-1):
  return signal.sosfilt(filter, frame, axis=axis)

def fourier_trans(signal, axis=-1):
  return fft.rfft(signal, axis=axis)


def remove_phase_shift(frames):
  ref_phase = np.angle(frames[0], deg=True)
  n_trials = frames.shape[0]
  for i in range(1, n_trials):
    p = np.angle(frames[i], deg=True)
    phase_diff = np.abs(p - ref_phase)

    phase_diff_exp = np.exp(-1 * phase_diff)
    frames[i] *= phase_diff_exp

  return frames


def extract_features(frames):
  yf = frames
  frame_len = 170
  sampling_freq = 200
  freq = fft.rfftfreq(frame_len, 1.0 / sampling_freq)
  trials_features = []
  for trial in yf:
    features_per_channel = []
    for channel in trial:
      features_per_channel.extend(channel[freq < 5])

    trials_features.append(features_per_channel)

  trials_features = np.array(trials_features)

  all_trials = []
  for trial in trials_features:
    features = []
    for channel in trial:
      r = channel.real
      im = channel.imag
      features.append(r)
      if im:
        features.append(im)

    all_trials.append(features)
  all_trials = np.array(all_trials)
  return all_trials


def create_dataset(list_of_all_trials_per_class, list_of_class_labels):
  list_of_datasets = []
  for trials_set, class_label in zip(list_of_all_trials_per_class, list_of_class_labels):
    class_set = pd.DataFrame(trials_set)
    class_set['label'] = class_label
    list_of_datasets.append(class_set)

  dataset = pd.concat(list_of_datasets, ignore_index=True)
  return dataset
