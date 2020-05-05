import mnist
import numpy
import glob
from CNN import CNN
from convolutional import Conv
from pool import Pool
from softmax import Softmax
from cnnLayer import CNNLayer
from dense import Dense
from PIL import Image
from shutil import copyfile
import csv

def leNet():
  cnn = CNN(2)
  # conv1 = Conv(1,8,3)
  # pool1 = Pool(2,2,'max')
  # sm = Softmax(10, 13 * 13 * 8)
  conv1 = Conv(1,6,5,initializer='xavier_gennorm',activation='comb2')
  pool1 = Pool(2,2,'max2')
  conv2 = Conv(6,16,5,initializer='xavier_gennorm',activation='comb2')
  pool2 = Pool(2,2,'max2')
  conv3 = Conv(16,120,4,initializer='xavier_gennorm',activation='comb2')
  # pool3 = Pool(2,2,'max')
  # fc1 = CNNLayer(84, 120, 'tanh')
  fc1 = Dense(84, 120, initializer='xavier_gennorm',activation='comb2')
  sm = Softmax(2, 84,initializer='xavier_gennorm')
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
  # conv1 = Conv(1, 96, 11, 4)
  # pool1 = Pool(2, 3, 'max')
  # conv2 = Conv(96, 256, 5)
  # pool2 = Pool(2, 3, 'max')
  # conv3 = Conv(256, 384, 3)
  # conv4 = Conv(384, 384, 3)
  # conv5 = Conv(384, 256, 3)
  # pool3 = Pool(2, 3, 'max')
  # fc1 = CNNLayer(256, 256, 'relu')
  # fc2 = CNNLayer(256, 256, 'relu')
  # sm = Softmax(2, 256)
  conv1 = Conv(1, 6, 11, 4,initializer='xavier_uniform',activation='relu')
  pool1 = Pool(2, 2, 'max')
  conv2 = Conv(6, 16, 5,initializer='xavier_uniform',activation='relu')
  pool2 = Pool(2, 2, 'max')
  conv3 = Conv(16, 32, 3,initializer='xavier_uniform',activation='relu')
  conv4 = Conv(32, 32, 3,initializer='xavier_uniform',activation='relu')
  conv5 = Conv(32, 16, 3,initializer='xavier_uniform',activation='relu')
  pool3 = Pool(2, 2, 'max')
  # fc1 = CNNLayer(128, 16, 'relu')
  # fc2 = CNNLayer(256, 128, 'relu')
  fc1 = Dense(128, 64,initializer='xavier_uniform',activation='relu')
  fc2 = Dense(256, 128,initializer='xavier_uniform',activation='relu')
  sm = Softmax(2, 256,initializer='xavier_uniform')
  cnn.add(conv1)
  cnn.add(pool1)
  cnn.add(conv2)
  cnn.add(pool2)
  cnn.add(conv3)
  cnn.add(conv4)
  cnn.add(conv5)
  # cnn.add(pool3)
  cnn.add(fc1)
  cnn.add(fc2)
  cnn.add(sm)
  return cnn

def mnist_test():
    numpy.random.seed(1000)
    test_images = mnist.test_images()[:500]
    test_labels = mnist.test_labels()[:500]
    train_images = mnist.train_images()[:500]
    train_labels = mnist.train_labels()[:500]
    all_images = numpy.concatenate((train_images, test_images))
    all_labels = numpy.concatenate((train_labels, test_labels))
    permutation = numpy.random.permutation(len(all_images))
    all_images = all_images[permutation]
    all_labels = all_labels[permutation]
    k = 5
    total_loss = 0.0
    total_acc = 0.0
    part_size = len(all_images) // k
    for i in range(k):
        cnn = leNet()
        train_images = numpy.concatenate((all_images[:part_size*i], all_images[part_size*(i+1):]))
        train_labels = numpy.concatenate((all_labels[:part_size*i], all_labels[part_size*(i+1):]))
        test_images = all_images[part_size*i:part_size*(i+1)]
        test_labels = all_labels[part_size*i:part_size*(i+1)]
        for epoch in range(10):
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
                # label = numpy.swapaxes(label[numpy.newaxis], 0, 1)
                l, acc = cnn.train3(im, label)
                loss += l
                num_correct += acc

        # Test the CNN
        print('\n--- Testing the CNN ---')
        loss = 0
        num_correct = 0
        # for im, label in zip(test_images, test_labels):
        #   _, l, acc = cnn.forwardImage(im, label)
        #   loss += l
        #   num_correct += acc
        #
        test_images = numpy.swapaxes(test_images[numpy.newaxis], 0,1)
        # test_labels = numpy.swapaxes()
        _, loss, num_correct = cnn.forwardBatch(test_images, test_labels)
        #
        num_tests = len(test_images)
        print('Test Loss:', loss / num_tests)
        print('Test Accuracy:', num_correct / num_tests)
        total_loss += loss / num_tests
        total_acc += num_correct / num_tests
    print("Average loss {:.4f}, accuracy {:.2f}%".format((total_loss / k), (total_acc*100.0 / k)))


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
    j = 0
    for img in benign_images:
        i += 1
        img = Image.open(img).convert(mode = 'L')
        # img = Image.open(img).convert(mode = 'RGB')
        # img = img.resize((224, 224))
        img = img.resize((28, 28))
        image = numpy.array(img)
        if i > thresh:
            if test:
                j+=1
                if j > 100:
                    break
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
    for img in malignant_images:
        i += 1
        img = Image.open(img).convert(mode = 'L')
        # img = Image.open(img).convert(mode = 'RGB')
        # img = img.resize((224, 224))
        img = img.resize((28, 28))
        image = numpy.array(img)
        if i > thresh:
            if test:
                j+=1
                if j > 100:
                    break
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

    return numpy.array(train_data), numpy.array(train_labels), numpy.array(test_data), numpy.array(test_labels)

def skinTest():
    numpy.random.seed(1000)
    train_images, train_labels, test_images, test_labels = prepareData(525, True)
    all_images = numpy.concatenate((train_images, test_images))
    all_labels = numpy.concatenate((train_labels, test_labels))
    permutation = numpy.random.permutation(len(all_images))
    all_images = all_images[permutation]
    all_labels = all_labels[permutation]
    k = 5
    total_loss = 0.0
    total_acc = 0.0
    for i in range(k):
        print()
        print('K - %d' % (i+1))
        part_size = len(all_images) // k
        train_images = numpy.concatenate((all_images[:part_size*i], all_images[part_size*(i+1):]))
        train_labels = numpy.concatenate((all_labels[:part_size*i], all_labels[part_size*(i+1):]))
        test_images = all_images[part_size*i:part_size*(i+1)]
        test_labels = all_labels[part_size*i:part_size*(i+1)]
        # cnn = alexNet()
        cnn = leNet()
        for epoch in range(30):
            print('--- Epoch %d ---' % (epoch + 1))
            # Shuffle the training data
            permutation = numpy.random.permutation(len(train_images))
            train_images = train_images[permutation]
            train_labels = train_labels[permutation]
            permutation = numpy.random.permutation(len(test_images))
            test_images = test_images[permutation]
            test_labels = test_labels[permutation]
            loss = 0
            num_correct = 0
            samples = 0
            batch_size = 128
            steps = int(numpy.ceil((len(train_labels) / batch_size)))
            for step in range(steps):
                if (step + 1) * batch_size > len(train_images):
                    im = train_images[step * batch_size:]
                    label = train_labels[step * batch_size:]
                    samples = len(train_images)
                else:
                    im = train_images[step * batch_size: (step + 1) * batch_size]
                    label = train_labels[step * batch_size: (step + 1) * batch_size]
                    samples = (step+1)*batch_size
                im = numpy.swapaxes(im[numpy.newaxis], 0, 1)
                # im = numpy.swapaxes(im, 3, 1)
                # im = numpy.swapaxes(im, 2, 3)
                # label = numpy.swapaxes(label[numpy.newaxis], 0, 1)
                l, acc = cnn.train3(im, label)
                loss += l
                num_correct += acc
                print('[Step %d] Loss %.3f | Accuracy: %.2f%%' %
                    (samples, l, acc * 100.0))
            print('Past %d steps: Average Loss %.3f | Accuracy: %.2f%%' %
                (samples, loss / steps, (num_correct / steps)* 100.0))

        # Test the CNN
        print('\n--- Testing the CNN ---')
        test_images = numpy.swapaxes(test_images[numpy.newaxis], 0, 1)
        # test_images = numpy.swapaxes(test_images, 3, 1)
        # test_images = numpy.swapaxes(test_images, 2, 3)
        _, loss, num_correct = cnn.forwardBatch(test_images, test_labels)
        # num_tests = len(test_images)
        print('Test Loss:', loss)
        print('Test Accuracy:', num_correct)
        total_loss += loss
        total_acc += num_correct
    print("Average loss {:.4f}, accuracy {:.2f}%".format((total_loss / k), (total_acc*100.0 / k)))

skinTest()
# mnist_test()

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