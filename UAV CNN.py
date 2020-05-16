'''
Abstract: Python program that uses the Keras module to create, train and test a Convolutional Neural Network (CNN) based off of database of .bmp images
          This CNN will be used to detect defects on cargo airplanes

Created for Group 4, Senior Design, North Carolina A&T State University

Author: Adam Benedict

Language: Python 3

Date Completed: 2020-04-24
'''

import keras
from keras.models import *
import os

directory = 'C:/Users/Adam/Pictures/NEU surface defect database' # Location of NEU Database

model_location = 'C:/Users/Adam/UAV.h5'

class supporting_Functions():
    def retrieve_Images_by_Label(d, label):
        for value in list(d.values()):
            if label in value:
                yield value 

    def image_prep(directory, filename):
        from PIL import Image
        import numpy as np
        from keras import backend as K

        num_classes = 10


        image_location = os.path.join(directory, filename).replace("\\", "/")           
        im = Image.open(image_location)     
        im_nparray = np.asarray(im) # Convert Image into numpy array

        image_rows, image_columns = im.size
        image = im_nparray[:,:image_columns] 

        if "Cr" in image_location:     #   Crazing
            image_Label = (1,)
        elif "In" in image_location:   #   Inclusion
            image_Label = (2,)
        elif "Pa" in image_location:   #   Patches
            image_Label = (3,)
        elif "PS" in image_location:   #   Pitted Surface
            image_Label = (4,)
        elif "RS" in image_location:   #   Rolled-In Scale
            image_Label = (5,)
        elif "Sc" in image_location:   #   Scratch
            image_Label = (6,)

        image_rows, image_columns = im.size        
        im_nparray = np.asarray(im) # Convert Image into numpy array
        image = im_nparray[:,:image_columns]


        if K.image_data_format() == 'channels_first':
            image = image.reshape(1, 1, image_rows, image_columns)
            input_shape = (1, image_rows, image_columns)
        else:
            image = image.reshape(1, image_rows, image_columns, 1)
            input_shape = (image_rows, image_columns, 1)

        image = image.astype('float32')
        image /= 255

        # convert class vectors to binary class matrices
        image_Label = keras.utils.to_categorical(image_Label, num_classes)

        return image, image_Label

class main():
    def create_cnn():
        print("Creating Convolutional Neural Network...")

        from keras.datasets import mnist
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D

        image_rows, image_columns = 200, 200
        input_shape = (image_rows, image_columns, 1)

        num_classes = 10

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

        model.save(model_location)

    def train():
        print("Training Convolutional Neural Network...")


        import gc

        batch_size = 128
        epochs = 5
        
        labels = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
        d = {loopNumber:filename for loopNumber, filename in enumerate( os.listdir(directory) ) if filename.endswith(".bmp")}    # Creates dictionary from the database

        model = load_model(model_location)
        
        for loopNumber, image in enumerate( supporting_Functions.retrieve_Images_by_Label(d, labels[0]) ):
            print('Image: %d/%d' % (loopNumber+1, len(os.listdir(directory))-1 ))

            image, image_Label = supporting_Functions.image_prep(directory, image)

            model.fit(image, image_Label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    #validation_data=(image, image_Label))
            )
            if loopNumber % 25 == 0: model.save(model_location)
            gc.collect()    # Clears memory

    def test():
        print("Testing Convolutional Neural Network...")

        correct_Guesses = 0
        from PIL import Image
        from matplotlib import pyplot as plt
        import numpy as np
        import random

        from keras.models import load_model 

        model = load_model(model_location)

        d = {loopNumber:filename for loopNumber, filename in enumerate( os.listdir(directory) ) if filename.endswith(".bmp")}    # Creates dictionary from the database
        
        for x in range(1):
            filename = d[ random.randint( 0, len(d)-1 ) ]

            image, image_Label = supporting_Functions.image_prep(directory, filename)

            # evaluate the model
            score = model.evaluate(image, image_Label, verbose=1)

            predicted_label = np.argmax( model.predict(image) )
            label = np.argmax( image_Label )
            if (predicted_label == label): correct_Guesses +=  1

        print('Correct guesses:', correct_Guesses)
        print('Test loss:', score[0])
        print('Test accuracy:', correct_Guesses/(x+1) )



if __name__ == "__main__":
    #main.create_cnn()
    #main.train()
    #main.test()