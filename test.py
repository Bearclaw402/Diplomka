import mnist
import numpy
from CNN import CNN
from convolutional import Conv
from pool import Pool
from softmax import Softmax
from cnnLayer import CNNLayer

test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]
train_images = mnist.train_images()[:2000]
train_labels = mnist.train_labels()[:2000]
cnn = CNN(10)
conv1 = Conv(1,6,5)
pool1 = Pool(2,2,'avg')
conv2 = Conv(6,16,5)
pool2 = Pool(2,2,'avg')
conv3 = Conv(16,120,4)
fc1 = CNNLayer(84, 120, 'tanh')
sm = Softmax(10, 84)
cnn.add(conv1)
cnn.add(pool1)
cnn.add(conv2)
cnn.add(pool2)
cnn.add(conv3)
cnn.add(fc1)
cnn.add(sm)

# Train the CNN for 3 epochs
for epoch in range(10):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = numpy.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i > 0 and i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0
    l, acc = cnn.train(im, label)
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

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)