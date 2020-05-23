import numpy
import glob
from CNN import CNN
from Convolutional import Conv
from Pool import Pool
from Softmax import Softmax
from Dense import Dense
from PIL import Image


def LeNet():
    cnn = CNN(2, loss='sbce')
    conv1 = Conv(1, 6, 5, initializer='xavier_gennorm', activation='sqcbrt')
    pool1 = Pool(2, 2, 'max2')
    conv2 = Conv(6, 16, 5, initializer='xavier_gennorm', activation='sqcbrt')
    pool2 = Pool(2, 2, 'max2')
    conv3 = Conv(16, 120, 4, initializer='xavier_gennorm', activation='sqcbrt')
    fc1 = Dense(84, 120, initializer='xavier_gennorm', activation='sqcbrt')
    sm = Softmax(2, 84, initializer='xavier_gennorm')
    cnn.add(conv1)
    cnn.add(pool1)
    cnn.add(conv2)
    cnn.add(pool2)
    cnn.add(conv3)
    cnn.add(fc1)
    cnn.add(sm)
    return cnn


def BranoNet():
    cnn = CNN(2, loss='sensce')
    conv1 = Conv(1, 16, 3, stride=2, initializer='he_uniform', activation='leakytrelu')
    pool1 = Pool(3, 3, 'max')
    fc1 = Dense(64, 16 * 10 * 10, initializer='he_uniform', activation='leakytrelu')
    sm = Softmax(2, 64, initializer='he_uniform')
    cnn.add(conv1)
    cnn.add(pool1)
    cnn.add(fc1)
    cnn.add(sm)
    return cnn


def prepareData(malignant=0, benign=0, rgb=False, size=64):
    """Metóda, ktorá načíta zadaný počet dát zo zložky data.

        # Arguments
            malignant: počet obrázkov zhubných znamienok. (max=1113)
                     pri hodnote -1 sa načítajú všetky
            benign: počet obrázkov nezhubných znamienok. (max=6705)
                  pri hodnote -1 sa načítajú všetky
            rgb: True - obrázky sa načítajú farebné
                False - obrázky sa načítajú čiernobiele
            size: veľkosť, na ktorú sa načítané obrázky zmenšia.
      """
    training_dir = r'.\data'
    benign_dir = training_dir + '//benign'
    malignant_dir = training_dir + '\malignant'

    benign_images = glob.glob(benign_dir + '\*.jpg')
    malignant_images = glob.glob(malignant_dir + '\*.jpg')

    data = []
    labels = []

    i = 0
    for img in benign_images:
        i += 1
        if rgb:
            img = Image.open(img).convert(mode='RGB')
        else:
            img = Image.open(img).convert(mode='L')
        img = img.resize((size, size))
        image = numpy.array(img)
        if i > benign and benign >= 0:
            break
        data.append(image)
        labels.append(0)

    i = 0
    for img in malignant_images:
        i += 1
        if rgb:
            img = Image.open(img).convert(mode='RGB')
        else:
            img = Image.open(img).convert(mode='L')
        img = img.resize((size, size))
        image = numpy.array(img)
        if i > malignant and malignant >= 0:
            break
        data.append(image)
        labels.append(1)
    return numpy.array(data), numpy.array(labels)


def skinTest(epochs, batch_size, model='BranoNet'):
    """Metóda, ktorá vykoná 5-násobnú krížovú validáciu na zadanom modely,
    so zadaným počtom epoch a zadanou veľkosťou várky.
    Ak chcete zmeniť počet krížovej validácie, treba zmeniť parameter k.
    Dáta sa načítavajú metódou prepareData.

      # Arguments
          epochs: počet epoch pre každý beh trénovania.
          batch_size: veľkosť várky, po ktorej sa budú vrstvy trénovať
          model: model, ktorý sa má trénovať. (sú preddefinované len BranoNet a LeNet)
      """
    numpy.random.seed(1000)
    all_images, all_labels = prepareData(benign=-1, malignant=-1, rgb=False, size=64)
    permutation = numpy.random.permutation(len(all_images))
    all_images = all_images[permutation]
    all_labels = all_labels[permutation]
    k = 5
    total_loss = 0.0
    total_acc = 0.0
    total_stats = numpy.zeros((2, 2))
    for i in range(k):
        print()
        print('K - %d' % (i + 1))
        part_size = len(all_images) // k
        train_images = numpy.concatenate((all_images[:part_size * i], all_images[part_size * (i + 1):]))
        train_labels = numpy.concatenate((all_labels[:part_size * i], all_labels[part_size * (i + 1):]))
        test_images = all_images[part_size * i:part_size * (i + 1)]
        test_labels = all_labels[part_size * i:part_size * (i + 1)]
        if model == 'BranoNet':
            cnn = BranoNet()
        else:
            cnn = LeNet()
        for epoch in range(epochs):
            print('--- Epoch %d ---' % (epoch + 1))
            permutation = numpy.random.permutation(len(train_images))
            train_images = train_images[permutation]
            train_labels = train_labels[permutation]
            permutation = numpy.random.permutation(len(test_images))
            test_images = test_images[permutation]
            test_labels = test_labels[permutation]
            loss = 0
            num_correct = 0
            samples = 0
            stats = numpy.zeros((2, 2))
            steps = int(numpy.ceil((len(train_labels) / batch_size)))
            for step in range(steps):
                if (step + 1) * batch_size > len(train_images):
                    im = train_images[step * batch_size:]
                    label = train_labels[step * batch_size:]
                    samples = len(train_images)
                else:
                    im = train_images[step * batch_size: (step + 1) * batch_size]
                    label = train_labels[step * batch_size: (step + 1) * batch_size]
                    samples = (step + 1) * batch_size
                im = numpy.swapaxes(im[numpy.newaxis], 0, 1)
                l, accs = cnn.train(im, label, optimizer='Adam')
                stats += accs
                acc = (accs[0, 0] + accs[1, 1]) / numpy.sum(accs)
                loss += l
                num_correct += acc
                print('[Step %d] Loss %.3f | Accuracy: %.2f%%' %
                      (samples, l, acc * 100.0))
            print('Past %d steps: Average Loss %.3f | Accuracy: %.2f%%' %
                  (samples, loss / steps, (num_correct / steps) * 100.0))
            print('False Negative %d' % (stats[1, 0]))

        # Test the CNN
        print('\n--- Testing the CNN ---')
        test_images = numpy.swapaxes(test_images[numpy.newaxis], 0, 1)
        _, loss, num_correct = cnn.forwardBatch(test_images, test_labels)
        total_stats += num_correct
        acc = (num_correct[0, 0] + num_correct[1, 1]) / numpy.sum(num_correct)
        print('Test Loss:', loss)
        print('Test Accuracy:', acc)
        total_loss += loss
        total_acc += acc
        print("TN  |  FP")
        print(num_correct[0])
        print("FN  |  TP")
        print(num_correct[1])
        print("Sensitivity {:.2f}".format(num_correct[1, 1] / (num_correct[1, 1] + num_correct[1, 0])))
        print("Specificity {:.2f}".format(num_correct[0, 0] / (num_correct[0, 0] + num_correct[0, 1])))
    print("Average loss {:.4f}, accuracy {:.2f}%".format((total_loss / k), (total_acc * 100.0 / k)))
    print("TN  |  FP")
    print(total_stats[0] / k)
    print("FN  |  TP")
    print(total_stats[1] / k)
    print("Sensitivity {:.2f}%".format((total_stats[1, 1] / (total_stats[1, 1] + total_stats[1, 0])) * 100.0))
    print("Specificity {:.2f}%".format((total_stats[0, 0] / (total_stats[0, 0] + total_stats[0, 1])) * 100.0))

#stačí spustiť túto metódu a vykoná sa 5-násobná krížová validácia na všetkých dátach na sieti BranoNet
skinTest(30, 128, model='BranoNet')