import os, sys, shutil, random, glob
import numpy as np

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

from keras_cv.layers import RandomHue, RandomSaturation

from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L

#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

import PIL
from PIL import Image, ImageEnhance

Image.MAX_IMAGE_PIXELS = None

from skimage.io import imread
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#print("keras        {}".format(tf.keras.__version__))
print("tensorflow   {}".format(tf.__version__))
print("number of cpu cores {}".format(len(os.sched_getaffinity(0))))

IMG_SIZE = 256
NUM_CLASSES = 1
NUM_GPUS = 1
LRATE = 1e-5*NUM_GPUS
EPOCHS = 150
BATCH_SIZE = 16*NUM_GPUS
LOAD_EXTRA = True
LAST_LAYER = 50

def gray2rgb(g):
  return np.concatenate((g, g, g))
  
def loadData(root):
  classes = ['neg', 'pos']
  labels = {'neg': 0,
            'pos': 1
           }

  random.seed(2024)
  np.random.seed(2024)

  TEST_SIZE = 1024

  names_train = {}
  names_val = {}
  names_test = {}

  num_train = []
  num_val = []
  for cc in classes:
    names = sorted(glob.glob(os.path.join(root, cc, '*.jpg')))
    names_test[cc] = random.sample(names, TEST_SIZE)
    names = [v for v in names if v not in names_test[cc]]
    names_train[cc] = random.sample(names, int(0.8*len(names)) )
    names_val[cc] = [v for v in names if v not in names_train[cc]]
    num_train.append(len(names_train[cc]))
    num_val.append(len(names_val[cc]))

  print(num_train, num_val)

  max_num_train = np.amax(num_train)
  max_num_val = np.amax(num_val)

  # oversampling for equal sized classes
  for cc in classes:
    if max_num_train > len(names_train[cc]):
      names_train[cc].extend(random.choices(names_train[cc], k=max_num_train - len(names_train[cc])))
    if max_num_val > len(names_val[cc]):
      names_val[cc].extend(random.choices(names_val[cc], k=max_num_val - len(names_val[cc])))

  imgs_train = []
  imgs_val   = []
  imgs_test  = []

  lbls_train = []
  lbls_val   = []
  lbls_test  = []

  for cc in classes:
    for nn in names_train[cc]:
      img = imread(nn)
      if len(img.shape) == 2:
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
      imgs_train.append(img)
      lbls_train.append(labels[cc])
    for nn in names_val[cc]:
      img = imread(nn)
      if len(img.shape) == 2:
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
      imgs_val.append(img)
      lbls_val.append(labels[cc])
    for nn in names_test[cc]:
      img = imread(nn)
      if len(img.shape) == 2:
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
      imgs_test.append(img)
      lbls_test.append(labels[cc])

  return np.array(imgs_train), np.array(lbls_train), np.array(imgs_val), np.array(lbls_val), np.array(imgs_test), np.array(lbls_test)

if len(sys.argv) != 5:
  print('please provide slurm_job_id fold keyword unfrozen_num_layers')
  sys.exit()
else:
  JOBID      = sys.argv[1]
  FOLD       = sys.argv[2]
  KEYWORD    = sys.argv[3]
  LAST_LAYER = int(sys.argv[4])

data_augmentation = tf.keras.Sequential(
  [
    RandomBrightness(0.1),
    RandomContrast(0.1),
    RandomFlip(),  
    RandomRotation(0.5)
  ]
)

inputs = Input(shape=[IMG_SIZE, IMG_SIZE, 3])
inputs = data_augmentation(inputs)

base_model = EfficientNetB4(input_tensor=inputs, weights = 'imagenet', include_top=False)

x = base_model.layers[-3].output
x = GlobalAveragePooling2D()(x)
x = Dense(NUM_CLASSES, activation='sigmoid')(x)

for layer in base_model.layers[:-LAST_LAYER]:
  layer.trainable = False

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=RMSprop(learning_rate=LRATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
#tf.keras.utils.plot_model(model, show_shapes=True)

# load data
ROOT = '../data/classification'
trn_x, trn_y, val_x, val_y, tst_x, tst_y = loadData(ROOT)

print(trn_x.shape, trn_y.shape)
print(val_x.shape, val_y.shape)
print(tst_x.shape, tst_y.shape)

verbose = 1

f5name = KEYWORD + '_'  + JOBID + '_f'+FOLD + '_i' + str(EPOCHS)
cbs = []
cbs.append(ModelCheckpoint(f5name+'.h5',
                           monitor='val_loss',
                           verbose=verbose,
                           save_best_only=True,
                           save_weights_only=True,
                           mode='auto'))
cbs.append(EarlyStopping(monitor="val_loss",
                         min_delta=0,
                         patience=7,
                         verbose=verbose,
                         mode="auto",
                         baseline=None,
                         restore_best_weights=False))
cbs.append(ReduceLROnPlateau(monitor="val_loss",
                             factor=0.1,
                             patience=3,
                             min_lr=LRATE*0.001))

if NUM_CLASSES > 1:
  trn_y = to_categorical(trn_y, NUM_CLASSES)
  val_y = to_categorical(val_y, NUM_CLASSES)
  
hist = model.fit(
    trn_x, trn_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=verbose,
    validation_data=(val_x, val_y)
)

vloss = 1
vacc  = 0

METRIC = 'val_accuracy'
for l, d in zip(hist.history['val_loss'], hist.history[METRIC]):
  if l < vloss:
    vacc = d
    vloss = l

print('lowest loss:', vloss, 'highest '+METRIC+':', vacc)

f5name_final = f5name + '_' + str(int(vacc*1000))
shutil.move(f5name + '.h5', f5name_final+'.h5')

# predict the test set
batches = [tst_x[i:i+BATCH_SIZE] for i in range(0, len(tst_x), BATCH_SIZE)]
predicts = np.zeros((len(tst_x), NUM_CLASSES))

model.load_weights(f5name_final+'.h5')

counter = 0
for b in batches:
  pp = model.predict_on_batch( np.array(b) )
  if NUM_CLASSES > 1:
    pp = np.reshape(pp, (len(b), NUM_CLASSES))
  base = counter*BATCH_SIZE
  predicts[base:base+len(b)] += pp
  counter += 1

if NUM_CLASSES > 1:
  predicts = np.argmax(predicts, axis=1)

print(f5name_final)
fpr, tpr, threshold = metrics.roc_curve(tst_y, predicts)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.savefig(os.path.join(ROOT, 'auc.png'))

print('done.')
