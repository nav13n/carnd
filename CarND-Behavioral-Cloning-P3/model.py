
# coding: utf-8

# In[48]:

import sys
sys.executable
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import threading
np.random.seed(7) 

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.utils import np_utils
import math
import random

# Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
SAMPLES_PER_EPOCH=((20000//BATCH_SIZE)*BATCH_SIZE)
NO_OF_EPOCHS = 6


# In[49]:

def get_data():
    """
    Returns list of image frames and steering angles 
    """
    
    # Load dataset
    colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv('./data/driving_log.csv', skiprows=[0], names=colnames)

    
    ## Data Exploration ##
    
    # Get a summary of the dataset
    print(data.describe(), "\n")

    image_frames = data[['center', 'left', 'right']].values
    steering_angles = data['steering'].values

    print("Image Frames Shape:", image_frames.shape, "\n")
    print("Steering Angles Shape:", steering_angles.shape, "\n")
    print("Data loaded...")

    return image_frames, steering_angles


# In[50]:

#Data Exploration
#Histogram visualisation to see which steering angle ranges are most overrepresented
def visualise_steering_angles(steering_angles):
    binwidth = 0.025
    plt.hist(steering_angles, bins=np.arange(min(steering_angles), max(steering_angles) + binwidth, binwidth))
    plt.title('Number of augmented images per steering angle')
    plt.xlabel('Steering Angle')
    plt.ylabel('# Augmented Images')
    plt.show()
#images_frames, steering_angles = get_data()
#visualise_steering_angles(steering_angles)


# In[51]:

def load_data(image_frame, steering_angle):
    """
    Randomly picks one of the [center, left, right] images in image frame and returns it along with adjusted steering angle
    """
    index = np.random.randint(3)
    if (index == 1):
        steering_angle += 0.25    # Steering angle adjustment for left camera
    elif (index == 2):
        steering_angle -= 0.25    # Steering angle adjustment for right camera
    
    image_path = image_frame[index].split('/')[-1]
    current_image_path = './data/IMG/'+image_path
    image = mpimg.imread(current_image_path)

    return image, steering_angle


# In[52]:

### Data Augmentation Methods ###
### Credits: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def augment_brightness(image):
    """
    Returns an augmented image with random darkening
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    # Randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()
    
    # Apply the brightness reduction to the V channel
    hsv[:,:,2] = hsv[:,:,2]*random_bright
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return new_img 


def random_shadow(image):
    """
    Randomly distorts an image by adding a random shadow
    """
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def horizontal_vertical_shift(img, steering_angle):
    """
    Returns a randomly shifted image in horizontal/ vertical direction and adjusted steering angle according to the shift
    """
     # Translation
    rows, cols, channels = img.shape
    # Translation
    tx = 100*np.random.uniform() - 100/2
    steering_angle = steering_angle + tx/100*2*.2
    
    ty = 40*np.random.uniform() - 40/2
    
    transform_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])
    
    translated_image = cv2.warpAffine(img, transform_matrix, (cols, rows))
    return translated_image, steering_angle
        

def flip(image, measurement):
    """
    Returns a horizontly flipped image and adjusted steering angle
    """
    new_image = cv2.flip(image,1)
    new_measurement = measurement*(-1)
    return new_image, new_measurement


def crop(image):
    """
    Returns a cropped image cropped on top and bottom to remove sky and car hood position and resized to model input size 
    """
    cropped = cv2.resize(image[60:140,:], (64,64))

    return cropped

def augment_training_data(image_frame, steering_angle):
    """
    Augment the training data with random augmentation methods
    """
    image, adjusted_steering_angle = load_data(image_frame, steering_angle)
    augmented_image = augment_brightness(image)
    augmented_image = random_shadow(augmented_image)
    augmented_image, adjusted_steering_angle = horizontal_vertical_shift(augmented_image, adjusted_steering_angle)
    flip_image = np.random.randint(2)
    if flip_image==0:
        augmented_image, adjusted_steering_angle = flip(augmented_image, adjusted_steering_angle)
    return augmented_image, adjusted_steering_angle


# In[53]:

def visualise_image_augmentation(image, steering_angle):
    """
    Utility method to visualise augmentation operation on random images
    """
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4, 2, figsize=(14,14))
    fig.subplots_adjust(hspace = .2, wspace=.008)

    
    ax1.axis('off')
    ax1.set_title('Original | Angle : ' + str(steering_angle))
    ax1.imshow(image)
    
    ax2.axis('off')
    ax2.set_title('Brightness Aug.| Angle : '+str(steering_angle))
    ax2.imshow(augment_brightness(image))
    
    ax3.axis('off')
    ax3.set_title('Original | Angle : ' + str(steering_angle))
    ax3.imshow(image)
    
    ax4.axis('off')
    ax4.set_title('Shadow Aug.| Angle : '+str(steering_angle))
    ax4.imshow(random_shadow(image))
    
    ax5.axis('off')
    ax5.set_title('Original | Angle : ' + str(steering_angle))
    ax5.imshow(image)
    
    hv_x,hv_y = horizontal_vertical_shift(image, steering_angle)
    ax6.axis('off')
    ax6.set_title('HV Shift Aug.| Angle : '+str(hv_y))
    ax6.imshow(hv_x)
    
    ax7.axis('off')
    ax7.set_title('Original | Angle : ' + str(steering_angle))
    ax7.imshow(image)
    
    flip_x, flip_y = flip(image,steering_angle)
    ax8.axis('off')
    ax8.set_title('Flip Aug.| Angle : '+str(flip_y))
    ax8.imshow(flip_x)
    plt.show()

#image_frames, steering_angles = get_data()
#choice = np.random.randint(len(image_frames))
#x, y = load_data(image_frames[choice],steering_angles[choice])
#visualise_image_augmentation(x,y)


# In[54]:

def visualise_model(model):
    plot(model, to_file='model.png', show_shapes=True)
    img = read_image('model.png')
    # original image
    plt.subplots(figsize=(5,10))
    plt.subplot(111)
    plt.axis('off')
    plt.imshow(array_to_img(img))
    plt.show()
    
#model= get_model()


# In[55]:

### Data Generator###

# Adding to avoid threading issues with generators as mentioned here
#https://github.com/fchollet/keras/issues/1638
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self
    
    def __next__(self):
        with self.lock:
            return self.it.__next__()
        
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


generated_steering_angles = []
@threadsafe_generator
def generator(image_frames, steering_angles, batch_size=BATCH_SIZE):
    """
    Returns a batch of generated data by randomly applying brightness augmentation, shadow augmentation, vertical flipping, horizontal and vertical shift methods on original data. 
    """
    X_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    y_train = np.zeros((batch_size,), dtype = np.float32)

    while True:
        
        image_frames, steering_angles = shuffle(image_frames, steering_angles)
        for i in range(batch_size):
            choice = np.random.randint(len(image_frames))
            x, y = augment_training_data(image_frames[choice],steering_angles[choice])
            X_train[i] = crop(x)
            y_train[i] = y
            generated_steering_angles.append(y)    
        yield X_train, y_train


# In[56]:

###  Model Structure ####
def get_model():
    #Based on Comma.ai model [https://github.com/commaai/research/blob/master/train_steering_model.py]
    model = Sequential()
    
    #Input Normalisation Layer
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(64,64,3)))
    
    #Convolutional Layer
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu', name='Conv1'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv2'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv3'))
    
    #Fl
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512, activation='elu', name='FC1'))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1, name='output'))

    adam = Adam(lr = 0.0001)
    model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
    
    return model


# In[57]:

if __name__ == "__main__":
       
    image_frames, steering_angles = get_data()
    
    # Create train and test split with 10% of the data as test data 
    image_frames, steering_angles = shuffle(np.array(image_frames), np.array(steering_angles))
    X_train, X_valid, y_train, y_valid = train_test_split(image_frames, steering_angles, test_size=0.1)
    
    # Initialise training and validation generators
    train_generator = generator(X_train, y_train, batch_size=32)
    validation_generator = generator(X_valid, y_valid, batch_size=32)

    model = get_model()
    model.summary()
    model.fit_generator(train_generator, samples_per_epoch = SAMPLES_PER_EPOCH, nb_epoch=NO_OF_EPOCHS, validation_data = validation_generator, nb_val_samples = len(X_valid),verbose=1)
    model.save('model.h5')
    
    #Visualise generated data spread
    visualise_steering_angles(generated_steering_angles) 
    
    print('Successfully completed training!')

