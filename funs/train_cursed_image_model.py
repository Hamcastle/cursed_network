import numpy as np
import glob
import os
import copy
import keras
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.engine.input_layer import Input
from keras.callbacks import ModelCheckpoint

def setup_datagens(train_data_dir,validation_data_dir,img_width,img_height,batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode="nearest",
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen     = ImageDataGenerator(
        rescale=1./255,
        fill_mode="nearest",
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True)

    val_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True)

    return train_generator,val_generator

def setup_model(image_height=224,image_width=224):
    try:

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(image_height,image_width,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='sigmoid'))
        return model
    except:
        raise

def save_trained_model(trained_model,output_path):
    '''
    Saves the model after training
    Inputs: a trained keras model
    '''
    try:
        model_json   = trained_model.to_json()
        output_path  = os.path.expanduser(output_path)
        with open(output_path+'cursed_image_model.json','w') as json_file:
            json_file.write(model_json)
        print('finished writing out the model as json')
    except:
        raise

def main():

    img_width, img_height = 224, 224

    output_path = os.path.expanduser('out/')
    train_data_dir = os.path.expanduser('data/train/')
    validation_data_dir = os.path.expanduser('data/validation/')
    nb_train_samples = len(glob.glob('data/train/cursed/'))
    nb_validation_samples = len(glob.glob('data/validation/cursed/'))

    epochs = 75
    batch_size = 40

    train_generator,val_generator=setup_datagens(train_data_dir,validation_data_dir,
        img_width,img_height,batch_size)
    model = setup_model()
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit_generator(train_generator,steps_per_epoch=2000//batch_size,epochs=epochs,validation_data=val_generator,validation_steps=800//batch_size)
    model.save(output_path+'cursed_image_model.h5')
    save_trained_model(model,output_path)

if __name__ == '__main__':
    main()