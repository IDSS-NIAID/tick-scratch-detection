import os
import glob
import argparse
import numpy as np
import pickle

import time

from skimage.io import imread
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label

import cv2

from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomBrightness, RandomZoom
from tensorflow.keras.layers import Dropout, SeparableConv2D, Cropping2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L


NUM_CORES = len(os.sched_getaffinity(0)) - 1
NUM_CORES = 1 if NUM_CORES < 1 else NUM_CORES

print(f"using {NUM_CORES} cpu cores for OpenCV")
cv2.setNumThreads(NUM_CORES - 1)

"""
1. load all motion frames from the input folder, return a list of extracted guinea pig ROIs
2. use the pretrained model to batch predict guinea pig ROIs
3. save results to the parent folder of the input folder
"""

IMG_SIZE = 256
NUM_CLASSES = 1
NUM_GPUS = 1
LRATE = 1e-5*NUM_GPUS
EPOCHS = 150
BATCH_SIZE = 64*NUM_GPUS

def findGuineaPig(name):
  # find guinea pig by finding the largest bright object
  image = imread(name)
  
  minThres = 120 # may need to adjust later
  imageCopy = image[:,:,0]
  thresh = threshold_otsu(imageCopy)
  # for unknown reason otsu may produce unreasonably low threshold
  thresh = minThres if thresh < minThres else thresh
  mask = imageCopy > thresh
  
  labelImg = label(mask)
  objs = regionprops(labelImg)

  if not objs:
    print("no animal detected on frame", name)
    return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)# EfficientNet takes in RGB inputs

  # get the largest bright object
  objs = sorted(objs, key=lambda x: x['area'], reverse=True)
  bbox = objs[0].bbox # y0, x0, y1, x1

  pad = 30
  padding = [-pad, -pad, pad, pad]
  bbox = [sum(x) for x in zip(bbox, padding)]
  
  height = bbox[2]-bbox[0]
  width  = bbox[3]-bbox[1]
  
  # make it square
  if height > width:
    delta = (height - width) // 2
    bbox[1] -= delta
    bbox[3] += delta
  elif width > height:
    delta = (width - height) // 2
    bbox[0] -= delta
    bbox[2] += delta
  
  img_height, img_width = image.shape[0], image.shape[1]
  
  # shift if bbox goes beyond image boundaries
  if bbox[0] < 0:
    shift = -bbox[0]
    bbox[0] += shift
    bbox[2] += shift
  if bbox[1] < 0:
    shift = -bbox[1]
    bbox[1] += shift
    bbox[3] += shift
    
  if bbox[2] > img_height:
    shift = bbox[2] - img_height
    bbox[0] -= shift
    bbox[2] -= shift
  if bbox[3] > img_width:
    shift = bbox[3] - img_width
    bbox[1] -= shift
    bbox[3] -= shift
  
  roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
  roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

  return roi
  
def loadGuineaPigROIs(root):
  names = sorted(glob.glob(os.path.join(root, '*.jpg')))
  
  print("found", len(names), "frames in", root)
  
  with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
    results = list(executor.map(findGuineaPig, names))
    
  names = [os.path.basename(v) for v in names]
  return results, names
  
def buildModel(weightFile):
  inputs = Input(shape=[IMG_SIZE, IMG_SIZE, 3])
  base_model = EfficientNetB4(input_tensor=inputs, weights = 'imagenet', include_top=False)

  x = base_model.layers[-3].output
  x = GlobalAveragePooling2D()(x)
  x = Dense(NUM_CLASSES, activation='sigmoid')(x)

  model = Model(inputs=base_model.input, outputs=x)
  model.compile(optimizer=RMSprop(learning_rate=LRATE),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  #model.summary()
  model.load_weights(weightFile)
  
  return model
  
def predictROIs(frames, names, root, weights):
  eb4 = buildModel(weights)
  
  # predict the test set
  batches = [frames[i:i+BATCH_SIZE] for i in range(0, len(frames), BATCH_SIZE)]
  predicts = np.zeros((len(frames), NUM_CLASSES))

  for i, b in enumerate(batches):
    pp = eb4.predict_on_batch( np.array(b) )
    if NUM_CLASSES > 1:
      pp = np.reshape(pp, (len(b), NUM_CLASSES))
    base = i*BATCH_SIZE
    predicts[base:base+len(b)] += pp
    print(i, "out of", len(batches), "done", flush=True)

  if NUM_CLASSES > 1:
    predicts = np.argmax(predicts, axis=1)
    
  results = {}
  for n, v in zip(names, predicts):
    results[n] = v
    
  resultFile = os.path.join(os.path.dirname(root), os.path.basename(root)+'.pkl')
  with open(resultFile, 'wb') as rFile:
    pickle.dump(results, rFile)
  
  return
  
def getWeightFileName(folder):
  # there is one and only one weight file
  weights = glob.glob(os.path.join(folder, '*.h5'))
  return weights[0]
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, help="path to the input folder containing motion frames")
  args = parser.parse_args()
  
  weights = getWeightFileName('./weights')
  
  root = args.input.strip('"')
  tStart = time.time()
  rois, names = loadGuineaPigROIs(root)
  tEnd = time.time()
  print('roi extraction done in', tEnd-tStart, 'seconds', flush=True)
  
  predictROIs(rois, names, root, weights)
  
  print('done.')
