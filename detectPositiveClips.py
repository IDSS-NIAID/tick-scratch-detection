"""
detect positive events from frame predictions
"""

import os
import sys
import glob
import pickle
import shutil
import re
import numpy as np
import cv2

import collections
import multiprocessing
import argparse

def getVideoFPS(video_name):
  videoStream = cv2.VideoCapture(video_name)
  return videoStream.get(cv2.CAP_PROP_FPS)

def getPredictedVideoNames(predict_root):
  names = glob.glob(os.path.join(predict_root, '*.pkl'))
  names = [os.path.basename(v) for v in names]
  return names

def getVideoName(base_name, video_root):
  key = os.path.splitext(base_name)[0]
  t = key.split('_')[0]

  names = glob.glob(os.path.join(video_root, '*'+t+'*.mp4'))
  if len(names) == 0:
    print("No video found for:", t)
    sys.exit()

  return names[0], t

def makeOneClip(item):
  
  clip = item.clipNo
  frames = item.frames
  video_name = item.vName
  fps = item.FPS
  video_output = item.outPath
  key = item.key
  
  clipStr = '%06d' % clip

  # get the start and end time for the current clip
  tStart = str(int(frames[0]  / fps) - 1)
  tEnd   = str(int(frames[-1] / fps) + 1)

  # ffmpeg cut the clip
  clipName = "_".join([clipStr,
                       tStart,
                       tEnd])
  ffmpeg_cmd = " ".join(["ffmpeg",
                         "-i", re.escape(video_name),
                         "-y",
                         "-ss", tStart,
                         "-to", tEnd,
                         "-c copy",
                         re.escape(os.path.join(video_output, key, clipName+'.mp4'))])
  print(ffmpeg_cmd)
  os.system(ffmpeg_cmd)

parser = argparse.ArgumentParser()
parser.add_argument("--predict_root", required=True)
parser.add_argument("--clip_output", required=True)
parser.add_argument("--video_root", required=True)
args = parser.parse_args()

PREDICT_ROOT  = args.predict_root
CLIP_OUTPUT   = args.clip_output
VIDEO_ROOT    = args.video_root

POSITIVE_THRESHOLD = 0.75
MOTION_MAX_GAP = 50
CLIP_MIN_LENGTH = 15

os.makedirs(CLIP_OUTPUT, exist_ok=True)

# prepare for parallel processing
ParaPairs = collections.namedtuple('ParaPairs', ['clipNo', 'frames', 'vName', 'FPS', 'outPath', 'key'])
pairs = []

trials = getPredictedVideoNames(PREDICT_ROOT)

for trial in trials:
  key = os.path.splitext(trial)[0]
  
  videoName = os.path.join(VIDEO_ROOT, key+'.mp4')
  
  fps = getVideoFPS(videoName)
  os.makedirs(os.path.join(CLIP_OUTPUT, key), exist_ok=True)

  print(key, fps, videoName)

  # 1. load trial_000.pkl predictions
  with open(os.path.join(PREDICT_ROOT, trial), 'rb') as f:
    framePredict = pickle.load(f)

  # 2. reconstruct frame ids
  frameIDs = list(framePredict.keys())
  frameIDs = [int(os.path.splitext(v)[0].split('-')[1]) for v in frameIDs]

  # 3. positive frame ids with prediction scores > threshold
  positiveFrameIDs = []
  for ii, (kk, vv) in enumerate(framePredict.items()):
    if vv[0] > POSITIVE_THRESHOLD:
      positiveFrameIDs.append(frameIDs[ii])

  # 4. split positive frame ids based on frame number continuity
  positiveClips = np.split(positiveFrameIDs, 
                           np.where(np.diff(positiveFrameIDs) > MOTION_MAX_GAP)[0] + 1)
  for ii, cc in enumerate(positiveClips):
    if cc[-1]-cc[0] >= CLIP_MIN_LENGTH:
      pairs.append(ParaPairs(clipNo=ii, frames=cc, vName=videoName, FPS=fps, outPath=CLIP_OUTPUT, key=key))

pairsTuple = tuple(pairs)
pool = multiprocessing.Pool(processes = len(os.sched_getaffinity(0)))
pool.map(makeOneClip, pairsTuple)

print('done.')
