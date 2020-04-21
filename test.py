import mnist
import numpy
import glob
from CNN import CNN
from convolutional import Conv
from pool import Pool
from softmax import Softmax
from cnnLayer import CNNLayer
from PIL import Image
from shutil import copyfile
import csv

def leNet():
  cnn = CNN(10)
  # conv1 = Conv(1,8,3)
  # pool1 = Pool(2,2,'max')
  # sm = Softmax(10, 13 * 13 * 8)
  conv1 = Conv(1,6,5)
  pool1 = Pool(2,2,'avg')
  conv2 = Conv(6,16,5)
  pool2 = Pool(2,2,'avg')
  conv3 = Conv(16,120,4)
  # pool3 = Pool(2,2,'max')
  fc1 = CNNLayer(84, 120, 'tanh')
  sm = Softmax(10, 84)
  cnn.add(conv1)
  cnn.add(pool1)
  cnn.add(conv2)
  cnn.add(pool2)
  cnn.add(conv3)
  # cnn.add(pool3)
  cnn.add(fc1)
  cnn.add(sm)
  return cnn

def alexNet():
  cnn = CNN(2)
  conv1 = Conv(1, 96, 11, 4)
  pool1 = Pool(2, 3, 'max')
  conv2 = Conv(96, 256, 5)
  pool2 = Pool(2, 3, 'max')
  conv3 = Conv(256, 384, 3)
  conv4 = Conv(384, 384, 3)
  conv5 = Conv(384, 256, 3)
  pool3 = Pool(2, 3, 'max')
  fc1 = CNNLayer(256, 256, 'relu')
  fc2 = CNNLayer(256, 256, 'relu')
  sm = Softmax(2, 256)
  cnn.add(conv1)
  cnn.add(pool1)
  cnn.add(conv2)
  cnn.add(pool2)
  cnn.add(conv3)
  cnn.add(conv4)
  cnn.add(conv5)
  cnn.add(pool3)
  cnn.add(fc1)
  cnn.add(fc2)
  cnn.add(sm)
  return cnn

def mnist_test():

  cnn = leNet()

  test_images = mnist.test_images()[:2000]
  test_labels = mnist.test_labels()[:2000]
  train_images = mnist.train_images()[:2000]
  train_labels = mnist.train_labels()[:2000]

  numpy.random.seed(1000)
  for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = numpy.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0
    batch_size = 10
    # for i, (im, label) in enumerate(zip(train_images, train_labels)):
    for i in range(len(train_labels) // batch_size):
      if i > 0 and i % 10 == 0:
      # if i > 0:
        print(
          '[Step %d] Past %d steps: Average Loss %.3f | Accuracy: %d%%' %
          (i * batch_size, i*batch_size, loss / 100.0, num_correct)
        )
        loss = 0
        num_correct = 0.0
      im = train_images[i * batch_size: (i+1)*batch_size]
      im = numpy.swapaxes(im[numpy.newaxis], 0,1)
      label = train_labels[i * batch_size: (i+1)*batch_size]
      label = numpy.swapaxes(label[numpy.newaxis], 0, 1)
      l, acc = cnn.train3(im, label)
      loss += l
      num_correct += acc

  # Test the CNN
  print('\n--- Testing the CNN ---')
  loss = 0
  num_correct = 0
  for im, label in zip(test_images, test_labels):
    _, l, acc = cnn.forwardImage(im, label)
    loss += l
    num_correct += acc
  #
  # # _, loss, num_correct = cnn.forwardBatch(test_images, test_labels)
  #
  num_tests = len(test_images)
  print('Test Loss:', loss / num_tests)
  print('Test Accuracy:', num_correct / num_tests)


def prepareData():
  training_dir = r'C:\Users\ASUS\Documents\Diplomka\data'
  benign_dir = training_dir + '//benign'
  malignant_dir = training_dir + '\malignant'

  # Get the list of all the images
  benign_images = glob.glob(benign_dir + '\*.jpg')
  malignant_images = glob.glob(malignant_dir + '\*.jpg')

  # An empty list. We will insert the data into this list in (img_path, label) format
  train_data = []
  train_labels = []
  i = 0
  thresh = 200
  for img in benign_images:
    i += 1
    if i > thresh:
      break
    img = Image.open(img).convert(mode = 'L')
    img = img.resize((224, 224))
    image = numpy.array(img)
    train_data.append(image)
    train_labels.append(0)

  i = 0
  for img in malignant_images:
    i += 1
    if i > thresh:
      break
    img = Image.open(img).convert(mode = 'L')
    img = img.resize((224, 224))
    image = numpy.array(img)
    train_data.append(image)
    train_labels.append(1)

  return numpy.array(train_data), numpy.array(train_labels)

def skinTest():
  train_images, train_labels = prepareData()
  cnn = alexNet()
  for epoch in range(10):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = numpy.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
      if i > 0 and i % 10 == 9:
        print(
          '[Step %d] Past 10 steps: Average Loss %.3f | Accuracy: %d%%' %
          (i + 1, loss / 10, num_correct)
        )
        loss = 0
        num_correct = 0
      l, acc = cnn.train(im, label)
      loss += l
      num_correct += acc

# skinTest()
mnist_test()

def imageCopy():
  src = 'C://Users\ASUS\Downloads\Data\ISIC-images\HAM10000//'
  dst = 'C://Users\ASUS\Documents\Diplomka\data//'
  with open('C://Users\ASUS\Downloads\Data\ISIC-images\data_komplet.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    firstLine = True
    for row in readCSV:
      if firstLine:
        firstLine = False
      else:
        copyfile(src + row[1] + '.jpg', dst + row[2] +"//" + row[1] + '.jpg')
# imageCopy()