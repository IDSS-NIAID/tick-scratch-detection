import os
import cv2
import sys
import glob

import pickle

import numpy as np

def frameDifference(frame1, frame2, kernel=np.array((9,9), dtype=np.uint8)):
  f1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
  f2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

  f_diff = cv2.subtract(f1_gray, f2_gray)
  f_diff = cv2.medianBlur(f_diff, 3)

  msk = cv2.adaptiveThreshold(f_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 11, 3)
  #msk = cv2.medianBlur(msk, 3)
  #msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel, iterations=1)

  numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(msk)

  msk_filtered = np.zeros_like(labels)

  min_size = 9*9
  for i in range(1, numLabels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > min_size:
      msk_filtered[labels == i] = 255

  return msk_filtered

ROOT = '../../data/duration_repaired/20240430_no_tick'
OUTPUT = '../../data/r2/frames_with_motion_no_tick'

ROOT = '../../data/duration_repaired/20240429_tick'
OUTPUT = '../../data/r2/frames_with_motion'

SAVE_WITH_OVERLAY = False

KEY = 'P382R'
CLIP = '000'

os.makedirs(OUTPUT, exist_ok=True)
movies = sorted(glob.glob(os.path.join(ROOT, '*'+KEY+'*.mp4')))
print(movies)

os.makedirs(os.path.join(OUTPUT, KEY, CLIP), exist_ok=True)

videoStream = cv2.VideoCapture(movies[0])
print('Frames', videoStream.get(cv2.CAP_PROP_FRAME_COUNT))
print('FPS', videoStream.get(cv2.CAP_PROP_FPS) )

framePrev = None
frameCurrent = None

limit = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))

for ii in range(limit):
  ret, frame = videoStream.read()

  if ii == 0:
    framePrev = frame
  elif ii == 1:
    frameCurrent = frame
  else:
    framePrev = frameCurrent
    frameCurrent = frame

  if frameCurrent is not None and framePrev is not None:
    f_diff = frameDifference(framePrev, frameCurrent)

    if np.amax(f_diff) > 0:
      pixelCount = np.sum(f_diff)//255
      #print('  ', ii, pixelCount)

      if SAVE_WITH_OVERLAY:
        f_overlay = frameCurrent.copy()
        f_overlay[:,:,2][f_diff > 0] = 255
        cv2.imwrite(os.path.join(OUTPUT, KEY, CLIP, 'frame-%08d.jpg' % ii ), f_overlay)
      else:
        cv2.imwrite(os.path.join(OUTPUT, KEY, CLIP, 'frame-%08d.jpg' % ii ), frameCurrent)

print('done.')
