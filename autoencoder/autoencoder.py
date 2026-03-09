import os
import glob
import random

import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Sequence

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

import PIL
from PIL import Image, ImageEnhance

def buildAE(lrate, patch_size):
  input = layers.Input(shape=(patch_size, patch_size, 1))

  # Encoder
  x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
  x = layers.MaxPooling2D((2, 2), padding="same")(x)

  x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2), padding="same")(x)

  x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2), padding="same")(x)

  x = layers.Conv2D(4, (3, 3), activation="relu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2), padding="same")(x)

  x = layers.Flatten()(x)
  x = layers.Dense(512, activation='relu')(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dense(512, activation='relu')(x)
  x = layers.Dense(1024, activation='relu')(x)
  x = layers.Reshape( (16, 16, 4) )(x)

  # Decoder
  x = layers.Conv2DTranspose(4, (2, 2), strides=2, activation="relu", padding="same")(x)
  x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)

  x = layers.Conv2DTranspose(8, (2, 2), strides=2, activation="relu", padding="same")(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

  x = layers.Conv2DTranspose(16, (2, 2), strides=2, activation="relu", padding="same")(x)
  x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

  x = layers.Conv2DTranspose(32, (2, 2), strides=2, activation="relu", padding="same")(x)
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

  x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

  # Autoencoder
  autoencoder = Model(input, x)
  autoencoder.compile(optimizer=Adam(learning_rate=lrate),
                      loss="mse")
  autoencoder.summary()

  return autoencoder
  
def loadData(root, patch_size):
  subfolders = glob.glob(os.path.join(root, '*'))
  subfolders = [os.path.basename(v) for v in subfolders if os.path.isdir(v)]

  allFrames = []
  for sf in subfolders:
    frames = glob.glob(os.path.join(root, sf, '*.jpg'))

    for frame in frames:
      ff = imread(frame)
      if len(ff.shape) > 2:
        allFrames.append(rgb2gray(ff))
      else:
        allFrames.append(ff)

  return np.array(allFrames)

class DataGen(Sequence):
  def __init__(self, data, batchSize, aug=False):
    self.data = data
    self.batchSize = batchSize
    self.aug = aug

  def __len__(self):
    return len(self.data) // self.batchSize

  def augment(self, img):
    newImg = Image.fromarray(img, mode="L")

    enhancer = ImageEnhance.Brightness(newImg)
    newImg = enhancer.enhance(random.uniform(0.75, 1.25))

    enhancer = ImageEnhance.Contrast(newImg)
    newImg = enhancer.enhance(random.uniform(0.75, 1.25))

    t = random.randint(0, 4)
    if t == 0:
      newImg = newImg.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    elif t == 1:
      newImg = newImg.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    elif t == 2:
      rot = random.randint(0,360)
      tran = random.randint(-25, 25)
      newImg = newImg.rotate(rot, resample=PIL.Image.Resampling.BICUBIC, translate=(tran, tran), fillcolor=0)
    elif t == 3:
      newImg = newImg.transpose(PIL.Image.TRANSPOSE)

    return np.array(newImg)

  def __getitem__(self, index):
    nn = self.data[index * self.batchSize : (index+1) * self.batchSize]

    imgs = []
    for n in nn:
      if self.aug == True:
        imgs.append(self.augment(n) / 255.)
      else:
        imgs.append(n / 255.)

    return np.array(imgs), np.array(imgs)

ROOT = '../data/clips/'
LRATE = 1e-4
VERBOSE = 1
PATCH_SIZE = 256

np.random.seed(2024)

allData = loadData(ROOT, PATCH_SIZE)
ae = buildAE(LRATE, PATCH_SIZE)

np.random.shuffle(allData)
mark = int(allData.shape[0]*0.85)
trainData = allData[:mark]
testData  = allData[mark:]

print(trainData.shape, testData.shape)

trnDataGen = DataGen(trainData,
                     256,
                     aug=True)
valDataGen = DataGen(testData,
                     256,
                     aug=True)

f5name = 'autoencoder'

cbs = []
cbs.append(ModelCheckpoint(f5name+'.weights.h5',
                           monitor='val_loss',
                           verbose=VERBOSE,
                           save_best_only=True,
                           save_weights_only=True,
                           mode='auto'))
cbs.append(EarlyStopping(monitor="val_loss",
                         min_delta=0,
                         patience=7,
                         verbose=VERBOSE,
                         mode="auto",
                         baseline=None,
                         restore_best_weights=False))
cbs.append(ReduceLROnPlateau(monitor="val_loss",
                             factor=0.1,
                             patience=3,
                             min_lr=LRATE*0.001))

hist = ae.fit(
  trnDataGen,
  epochs=100,
  validation_data=valDataGen,
  verbose=VERBOSE,
  callbacks=cbs,
  workers = len(os.sched_getaffinity(0))
)

print('done.')
