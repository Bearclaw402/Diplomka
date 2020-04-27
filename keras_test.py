import glob
import keras
import mnist
import numpy
import matplotlib.pyplot as plt

from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential
from keras import models, layers
# Load dataset as train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

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

    i = 0
    for img in benign_images:
        i += 1
        img = Image.open(img).convert(mode = 'L')
        img = img.resize((224, 224))
        # img = img.resize((28, 28))
        image = numpy.array(img)
        if i > thresh:
            if test:
                test_data.append(image)
                test_labels.append(0)
            else:
                break
        else:
            train_data.append(image)
            train_labels.append(0)

    i = 0
    for img in malignant_images:
        i += 1
        img = Image.open(img).convert(mode = 'L')
        img = img.resize((224, 224))
        # img = img.resize((28, 28))
        image = numpy.array(img)
        if i > thresh:
            if test:
                test_data.append(image)
                test_labels.append(1)
            else:
                break
        else:
            train_data.append(image)
            train_labels.append(1)

    i = 0
    for img in malignant_images:
        i += 1
        if i > 200:
            break
        img = Image.open(img).convert(mode = 'L')
        img = img.resize((224, 224))
        # img = img.resize((28, 28))
        image = numpy.array(img)
        test_data.append(image)
        test_labels.append(1)

    return numpy.array(train_data), numpy.array(train_labels), numpy.array(test_data), numpy.array(test_labels)


x_train, y_train, x_test, y_test = prepareData(1000, True)
permutation = numpy.random.permutation(len(x_train))
x_train = x_train[permutation]
y_train = y_train[permutation]
permutation = numpy.random.permutation(len(x_test))
x_test = x_test[permutation]
y_test = y_test[permutation]

# x_test = mnist.test_images()[:2000]
# y_test = mnist.test_labels()[:2000]
# x_train = mnist.train_images()[:2000]
# y_train = mnist.train_labels()[:2000]
#
# # Set numeric type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#
# # Normalize value to [0, 1]
x_train /= 255
x_test /= 255
#
# # Transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)
#
# # Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 224,224,1)
x_test = x_test.reshape(x_test.shape[0], 224,224,1)
# x_train = x_train.reshape(x_train.shape[0], 28,28,1)
# x_test = x_test.reshape(x_test.shape[0], 28,28,1)

def leNet():
    #Instantiate an empty model
    model = Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), input_shape=(28,28,1), padding="same"))

    # S2 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # C3 Convolutional Layer
    model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid'))

    # S4 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # C5 Fully Connected Convolutional Layer
    model.add(layers.Conv2D(120, kernel_size=(4, 4), strides=(1, 1), padding='valid'))
    #Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())

    # FC6 Fully Connected Layer
    model.add(layers.Dense(84, activation='tanh'))

    #Output Layer with softmax activation
    model.add(layers.Dense(2, activation='softmax'))
    return model

def alexNet():
    # Instantiate an empty model
    model = Sequential()
    model.add(layers.Conv2D(6, kernel_size=(11, 11), strides=(4, 4), input_shape=(224, 224, 1), padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    # Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    # Output Layer with softmax activation
    model.add(layers.Dense(2, activation='softmax'))
    return model

model = alexNet()
# model = leNet()
# Compile the model
ss = keras.optimizers.sgd()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=ss, metrics=["accuracy"])
# model.compile(loss='squared_hinge', optimizer=ss, metrics=["accuracy"])

hist = model.fit(x=x_train,y=y_train, epochs=30, batch_size=10, validation_data=(x_test, y_test), verbose=1)

######## EVALUATE ########
test_score = model.evaluate(x_test, y_test)
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))

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