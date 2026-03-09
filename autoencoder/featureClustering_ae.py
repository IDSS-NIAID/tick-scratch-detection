import os, sys, shutil, random, glob
import numpy as np
import umap
import hdbscan

import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.io import imsave
from sklearn.decomposition import PCA
from skimage.color import rgb2gray

BATCH_SIZE = 1024
PATCH_SIZE = 256
RANDOM_SEED = 2024
TARGET_COMPONENTS = 32

DATA_ROOT = '../data'
BASE = 'clips'
DATASET = 'dataset'

ROOT   = os.path.join(DATA_ROOT, BASE+'_'+DATASET            )
OUTPUT = os.path.join(DATA_ROOT, BASE+'_'+DATASET+'_ae_clusters')

def getAllFrames(all_trials, all_trials_names):
  """
  combine frames and frame names loaded using loadAllTrialClips* functions to two order-mached lists
  """
  all_frames = []
  all_frames_names = []

  for trial in all_trials:
    clips = all_trials[trial]
    clipsNames = all_trials_names[trial]
    for clip in clips:
      frames = clips[clip]
      framesNames = clipsNames[clip]
      framesNames = [trial + '_' + clip + '_' + v for v in framesNames]

      all_frames.extend(frames)
      all_frames_names.extend(framesNames)

  return all_frames, all_frames_names
  
def loadAllTrialClips(root, ext_string, resize_frame = True, frame_size = [128, 128], padding = 30):
  """
  load all clips organized by Trial and ClipID
  :param root: root path containing trials and clip subfolders
  :param ext_string: file extention for frame images, for example: *.jpg, *.tiff, etc.
  :param resize_frame: True if the frames need to be resized, False if loading frames as is
  :param frame_size: target frame image size
  :return: a dictionary of trials, each trial is a dictionary of clips, each clip is a list of grayscale image
           a dictionary of frame names
  """
  allTrials = {}
  allTrialsNames = {}

  trials = sorted(glob.glob(os.path.join(root, '*')))
  trials = [os.path.basename(v) for v in trials if os.path.isdir(v)]

  for trial in trials:
    clips = sorted(glob.glob(os.path.join(root, trial, '*')))
    clips = [os.path.basename(v) for v in clips if os.path.isdir(v)]

    allClips = {}
    allClipsNames = {}

    for clip in clips:
      frames = sorted(glob.glob(os.path.join(root, trial, clip, ext_string)))

      frameList = []
      frameNameList = []
      for frame in frames:
        frameNpy = imread(frame)
        frameNpy = findGuineaPig(frameNpy, padding)
        if resize_frame:
          frameDtype = frameNpy.dtype
          frameNpy = resize(frameNpy, frame_size, preserve_range=True, anti_aliasing=True).astype(frameDtype)

        frameList.append(frameNpy)
        frameNameList.append(os.path.basename(frame))

      allClips[clip] = frameList
      allClipsNames[clip] = frameNameList

    allTrials[trial] = allClips
    allTrialsNames[trial] = allClipsNames

  return allTrials, allTrialsNames
  
def buildModel(weight_file):
  base_model = buildAE(lrate=1e-5, patch_size=PATCH_SIZE)
  base_model.load_weights(weight_file)

  model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense_1').output)
  model.summary()

  return model

def getFeatures(frames, model):
  features = []

  batches = [frames[i:i + BATCH_SIZE] for i in range(0, len(frames), BATCH_SIZE)]

  for b in batches:
    predicts = model.predict_on_batch(np.array(b))

    for pp in predicts:
      features.append(pp.flatten())

  return np.array(features)

def preprocessFrames(frames):
  all_frames = []
  for frame in frames:
    if len(frame.shape) > 2:
      all_frames.append( rgb2gray(frame) / 255. )
    else:
      all_frames.append( frame / 255. )

  return all_frames

def doPCA(features):
  return PCA(n_components=TARGET_COMPONENTS).fit_transform(features)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

allTrials, allTrialsNames = loadAllTrialClips(ROOT, '*.jpg', True, [PATCH_SIZE, PATCH_SIZE])
allFrames, allFramesNames = getAllFrames(allTrials, allTrialsNames)

print('Frames:', len(allFrames))

allFramesProcessed = preprocessFrames(allFrames)
allFeatures = getFeatures(allFramesProcessed, buildModel('./autoencoder.h5'))

allFeaturesPCA = doPCA(allFeatures)
print('allFeatures:', allFeatures.shape)
print('allFeaturesPCA:', allFeaturesPCA.shape)

standard_embedding = umap.UMAP(random_state=RANDOM_SEED).fit_transform(allFeaturesPCA)

labels = hdbscan.HDBSCAN(min_cluster_size=100, approx_min_span_tree=False).fit_predict(standard_embedding)
uu, cc = np.unique(labels, return_counts=True)

print('Cluster\tPoints')
for uuu, ccc in zip(uu, cc):
  print(uuu, ccc)

if os.path.exists(OUTPUT):
  shutil.rmtree(OUTPUT)

for u in uu:
  os.makedirs(os.path.join(OUTPUT, str(u)), exist_ok=True)

for ll, ii, nn in zip(labels, allFrames, allFramesNames):
  imsave(os.path.join(OUTPUT, str(ll), nn), ii, check_contrast=False)

clustered = (labels >= 0)

plt.figure(figsize=(10, 10))

plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral')

plt.savefig(os.path.join(OUTPUT, 'clusters.png'))

print('done.')
