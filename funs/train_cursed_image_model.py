import numpy as np
import glob
import os
import copy
import keras
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.applications import MobileNet
from keras.callbacks import ModelCheckpoint
import argparse


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
        base_model=MobileNet(weights='imagenet',include_top=False,input_shape=(image_height,image_width,3)) 
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x=Dense(1024,activation='relu')(x)
        x=Dropout(0.5)(x)
        x=Dense(1024,activation='relu')(x) #dense layer 2
        x=Dropout(0.5)(x)
        x=Dense(512,activation='relu')(x) #dense layer 3
        preds=Dense(1,activation='sigmoid')(x)
        model=Model(inputs=base_model.input,outputs=preds)
        return model
    except:
        raise

def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-imw","--img_width",default=224,help="Size to set image widths to")
        ap.add_argument("-imh","--img_height",default=224,help="Size to set image height to")
        ap.add_argument("-btch","--batch_size",default=40,help="Data augmentation bath size")
        ap.add_argument("-e","--epochs",default=30,help="Training epochs")
        ap.add_argument("-o","--output_path",default="out/",help="Destination for the trained model files")
        ap.add_argument("-t","--train_data_dir",default="data/train/",help="Path to the training data")
        ap.add_argument("-v","--validation_data_dir",default="data/validation/",help="Path to the validation data")
        args = vars(ap.parse_args())


        nb_train_samples = len(glob.glob('data/train/cursed/'))
        nb_validation_samples = len(glob.glob('data/validation/cursed/'))


        train_generator,val_generator=setup_datagens(args['train_data_dir'],args['validation_data_dir'],
            args['img_width'],args['img_height'],args['batch_size'])
        model = setup_model()
        model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
        model.fit_generator(train_generator,steps_per_epoch=2000//args['batch_size'],epochs=args['epochs'],validation_data=val_generator,validation_steps=800//args['batch_size'])
        model.save(args['output_path']+'cursed_image_model.h5')
    except:
        raise

if __name__ == '__main__':
    main()