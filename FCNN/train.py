import os
import tensorflow as tf
import numpy as np
import keras
import scipy.ndimage as ndimage
from keras import backend as K
import cv2
import CNN

def act(x):
    return (K.sigmoid(x) * x)
keras.utils.generic_utils.get_custom_objects().update({'act': keras.layers.Activation(act)})
    

model = CNN.Density()

# Convenience function to normalise
def normalize(array):
    return (array-np.mean(array))/np.std(array)

# Read Data
def read_data(path):
    data1=[]
    anno1=[]
    
    for base_path in path:
        imList = os.listdir(base_path)
    
        for i in range(1,(len(imList)//2)+1):
            
            img1 = cv2.imread(os.path.join(base_path, str(i).zfill(3)+"cell.tif"), cv2.IMREAD_GRAYSCALE)
            img1 = np.expand_dims(img1, axis=-1)
            data1.append(normalize(img1))
        
            img2 = cv2.imread(os.path.join(base_path, str(i).zfill(3)+"dots.png"), cv2.IMREAD_GRAYSCALE)
            img2 = 100.0*(img2[:, :] > 0)
            img2 = np.asarray(img2, dtype='float64')
            img2 = ndimage.gaussian_filter(img2, sigma=2, order=0, mode="constant")
            img2 = np.expand_dims(img2, axis=-1)
            anno1.append(img2)

    return (np.asarray(data1, dtype='float64'), np.asarray(anno1, dtype='float64'))


def train_(base_path, weight):

    try:
        K.get_session().run(tf.global_variables_initializer())
        model.load_weights("weights/"+weight+".h5")
    except:
        print("No saved checkpoint")


    data, anno = read_data([base_path])

    # Ensures size is divisible by 4 as the network downsamples and upsamples
    while data.shape[1] %4 !=0:
        data = data[:,:-1,:]
        anno = anno[:,:-1,:]
        
    while data.shape[2] %4 != 0:
        data = data[:,:,:-1]
        anno = anno[:,:,:-1]

    # Change to train on some, validate on rest
    trainnum = round(data.__len__()*(10/10))
    
    data = data[:trainnum]
    anno = anno[:trainnum]

    val = (normalize(data[trainnum:]), anno[trainnum:])
    
    data2=np.rot90(data, axes=(1,2))
    anno2=np.rot90(anno, axes=(1,2))
    
    data = np.append(data, data+np.random.randn(*data.shape), axis=0)
    anno = np.append(anno, anno, axis=0)
    
    if data.shape[1]==data.shape[2]:
        data = np.append(data, data2, axis=0)
        anno = np.append(anno, anno2, axis=0)
    
    
    print(data.shape, anno.shape)
    model.fit(normalize(data), anno, epochs=10, batch_size=1, validation_data=val)

    model.save_weights("weights/"+weight+".h5")


if __name__ == '__main__':

    #Uncomment to train. First parameter is path of training folder, second is name of saved weights
    #train_("train","synthetic")