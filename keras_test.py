import glob
import numpy
import tensorflow as tf

from PIL import Image
from keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras import models, layers, Model
from keras.applications.resnet50 import ResNet50
from keras.initializers import Constant
# Load dataset as train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1000

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

numpy.random.seed(seed_value)

tf.compat.v1.set_random_seed(seed_value)
tf.random.set_seed(seed_value)

def prepareData(thresh, test):
    training_dir = r'C:\Users\ASUS\Documents\Diplomka\data'
    benign_dir = training_dir + '//benign'
    malignant_dir = training_dir + '\malignant'

    # Get the list of all the images
    benign_images = glob.glob(benign_dir + '\*.jpg')
    malignant_images = glob.glob(malignant_dir + '\*.jpg')

    # An empty list. We will insert the data into this list in (img_path, label) format
    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    valid_data = []
    valid_labels = []

    i = 0
    j = 0
    k = 0
    for img in benign_images:
        i += 1
        img = Image.open(img).convert(mode = 'L')
        # img = Image.open(img).convert(mode='RGB')
        # img = img.resize((224, 224))
        # img = img.resize((28, 28))
        img = img.resize((64, 64))
        image = numpy.array(img)
        if i > thresh:
            if test:
                j+=1
                if j > 100:
                    k+=1
                    if k > 100:
                        break
                    else:
                        valid_data.append(image)
                        valid_labels.append(0)
                else:
                    test_data.append(image)
                    test_labels.append(0)
            else:
                break
        else:
            train_data.append(image)
            train_labels.append(0)

    i = 0
    j = 0
    k = 0
    for img in malignant_images:
        i += 1
        img = Image.open(img).convert(mode = 'L')
        # img = Image.open(img).convert(mode='RGB')
        # img = img.resize((224, 224))
        # img = img.resize((28, 28))
        img = img.resize((64, 64))
        image = numpy.array(img)
        if i > thresh:
            if test:
                j+=1
                if j > 100:
                    k+=1
                    if k > 100:
                        break
                    else:
                        valid_data.append(image)
                        valid_labels.append(1)
                else:
                    test_data.append(image)
                    test_labels.append(1)
            else:
                break
        else:
            train_data.append(image)
            train_labels.append(1)

    # i = 0
    # for img in malignant_images:
    #     i += 1
    #     if i > 200:
    #         break
    #     img = Image.open(img).convert(mode = 'L')
    #     # img = img.resize((224, 224))
    #     img = img.resize((28, 28))
    #     image = numpy.array(img)
    #     test_data.append(image)
    #     test_labels.append(1)

    train_data = numpy.array(train_data)
    train_labels = numpy.array(train_labels)
    test_data = numpy.array(test_data)
    test_labels = numpy.array(test_labels)
    valid_data = numpy.array(valid_data)
    valid_labels = numpy.array(valid_labels)
    return train_data, train_labels, test_data, test_labels, valid_data, valid_labels


x_train, y_train, x_test, y_test, x_valid, y_valid = prepareData(500, True)
permutation = numpy.random.permutation(len(x_train))
x_train = x_train[permutation]
y_train = y_train[permutation]
permutation = numpy.random.permutation(len(x_test))
x_test = x_test[permutation]
y_test = y_test[permutation]
permutation = numpy.random.permutation(len(x_valid))
x_valid = x_valid[permutation]
y_valid = y_valid[permutation]

# x_test = mnist.test_images()[:2000]
# y_test = mnist.test_labels()[:2000]
# x_train = mnist.train_images()[:2000]
# y_train = mnist.train_labels()[:2000]
#
# # Set numeric type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_valid = x_valid.astype('float32')
#
# # Normalize value to [0, 1]
x_train /= 255
x_test /= 255
x_valid /= 255
#
# # Transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)
y_valid = np_utils.to_categorical(y_valid, 2)
#
# # Reshape the dataset into 4D array
# x_train = x_train.reshape(x_train.shape[0], 224,224,3)
# x_test = x_test.reshape(x_test.shape[0], 224,224,3)
# x_valid = x_valid.reshape(x_valid.shape[0], 224,224,3)
# x_train = x_train.reshape(x_train.shape[0], 28,28,3)
# x_test = x_test.reshape(x_test.shape[0], 28,28,3)
# x_valid = x_valid.reshape(x_valid.shape[0], 28,28,3)
x_train = x_train.reshape(x_train.shape[0], 64,64,1)
x_test = x_test.reshape(x_test.shape[0], 64,64,1)
x_valid = x_valid.reshape(x_valid.shape[0], 64,64,1)

def leNet(init, act):
    #Instantiate an empty model
    model = Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv2D(6, kernel_initializer=init, kernel_size=(5, 5), strides=(1, 1), input_shape=(28,28,3), padding="same", activation=act))
    model.add(layers.PReLU())
    # model.add(layers.BatchNormalization())

    # S2 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # C3 Convolutional Layer
    model.add(layers.Conv2D(16, kernel_initializer=init, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=act))
    model.add(layers.PReLU())
    # model.add(layers.BatchNormalization())

    # S4 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # C5 Fully Connected Convolutional Layer
    model.add(layers.Conv2D(120, kernel_initializer=init, kernel_size=(4, 4), strides=(1, 1), padding='valid', activation=act))
    model.add(layers.PReLU())
    # model.add(layers.BatchNormalization())
    #Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())

    # FC6 Fully Connected Layer
    model.add(layers.Dense(84, kernel_initializer=init, activation=act))
    model.add(layers.PReLU())
    # model.add(layers.BatchNormalization())

    #Output Layer with softmax activation
    model.add(layers.Dense(2, kernel_initializer=init, activation='softmax'))
    return model

def alexNet2(init, act):
    # Instantiate an empty model
    model = Sequential()
    model.add(layers.Conv2D(6, kernel_initializer=init, kernel_size=(11, 11), strides=(4, 4), input_shape=(224, 224, 3), padding="same", activation=act))
    model.add(layers.PReLU())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(16, kernel_initializer=init, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=act))
    model.add(layers.PReLU())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(32, kernel_initializer=init, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=act))
    model.add(layers.PReLU())
    model.add(layers.Conv2D(32, kernel_initializer=init, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=act))
    model.add(layers.PReLU())
    model.add(layers.Conv2D(16, kernel_initializer=init,  kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=act))
    model.add(layers.PReLU())
    # Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, kernel_initializer=init, activation=act))
    model.add(layers.PReLU())
    model.add(layers.Dense(256, kernel_initializer=init, activation=act))
    model.add(layers.PReLU())
    # Output Layer with softmax activation
    model.add(layers.Dense(2, kernel_initializer=init, activation='softmax'))
    return model

def alexNet():
    # Instantiate an empty model
    model = Sequential()
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), input_shape=(224, 224, 1), padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    # Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    # Output Layer with softmax activation
    model.add(layers.Dense(2, activation='softmax'))
    return model

def vgg16():
    # Instantiate an empty model
    model = Sequential()
    model.add(layers.Conv2D(input_shape=(224, 224, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dense(units=2, activation="softmax"))
    return model

def vgg16_2():
    # Instantiate an empty model
    model = Sequential()
    model.add(layers.Conv2D(input_shape=(224, 224, 1), filters=8, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dense(units=2, activation="softmax"))
    return model

def myNet(n_fils1, n_fils2):
    # Instantiate an empty model
    model = Sequential()
    model.add(layers.Conv2D(n_fils1, kernel_initializer='glorot_uniform', kernel_size=(5, 5), strides=(2, 2), input_shape=(64, 64, 1), padding="same", activation='relu'))
    # model.add(layers.PReLU())
    # model.add(layers.Dropout(rate=0.2))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid'))
    # model.add(layers.Conv2D(16, kernel_initializer='glorot_uniform', kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
    # Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(n_fils2, kernel_initializer='glorot_uniform', activation='relu'))
    # model.add(layers.PReLU())
    # model.add(layers.Dropout(rate=0.2))
    # Output Layer with softmax activation
    model.add(layers.Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))
    # model.add(layers.BatchNormalization())
    return model

# model = myNet()
# model = alexNet()
# model = vgg16()
# model = vgg16_2()
# Compile the model
# ss = keras.optimizers.sgd()
# base_model = ResNet50(weights = None, include_top=False)
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # x = Dropout(0.)(x)
# predictions = layers.Dense(2, activation= 'softmax')(x)
# model = Model(inputs = base_model.input, outputs = predictions)
# mm = keras.metrics.TruePositives()
# model.compile(loss='squared_hinge', optimizer=ss, metrics=["accuracy"])
# model.compile(loss=keras.losses.binary_crossentropy, optimizer=ss, metrics=[keras.metrics.FalseNegatives()])
for fils1 in range(1):
    for fils2 in range(1):
        total_loss = 0.0
        total_acc = 0.0
        reps = 10
        # print("combination fils1 = " + str(fils1+2)  + " fils2 = " + str(fils2+3))
        for i in range(reps):
            ss = tf.keras.optimizers.Adam()
            # model = alexNet2('glorot_normal', None)
            # model = leNet('lecun_normal', None)
            model=myNet(16, 64)
            model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=ss, metrics=["accuracy"])
            hist = model.fit(x=x_train,y=y_train, epochs=30, batch_size=128, validation_data=(x_valid, y_valid), verbose=0)

            ######## EVALUATE ########
            test_score = model.evaluate(x_test, y_test)
            print("Replication " + str(i+1))
            print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))
            total_loss += test_score[0]
            total_acc += test_score[1] * 100
            model.reset_metrics()
            model.reset_states()

        print("Average loss {:.4f}, accuracy {:.2f}%".format((total_loss/reps), (total_acc/reps)))

######## Visualize ########
# f, ax = plt.subplots()
# ax.plot([None] + hist.history['acc'], 'o-')
# ax.plot([None] + hist.history['val_acc'], 'x-')
# # Plot legend and use the best location automatically: loc = 0.
# ax.legend(['Train acc', 'Validation acc'], loc = 0)
# ax.set_title('Training/Validation acc per Epoch')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('acc')
#
# f, ax = plt.subplots()
# ax.plot([None] + hist.history['loss'], 'o-')
# ax.plot([None] + hist.history['val_loss'], 'x-')
# # Plot legend and use the best location automatically: loc = 0.
# ax.legend(['Train Loss', 'Validation Loss'], loc = 0)
# ax.set_title('Training/Validation Loss per Epoch')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')