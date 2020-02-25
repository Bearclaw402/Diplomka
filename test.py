import mnist
import numpy

import CNN
import convolutional
import pool
import softmax

test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
cnn = CNN.CNN(10)
conv = convolutional.Conv(8,3)
pool = pool.Pool()
sm = softmax.Softmax(10, 13 * 13 * 8)
cnn.add(conv)
cnn.add(pool)
cnn.add(sm)
# Train!
# loss = 0
# num_correct = 0
# for i, (im, label) in enumerate(zip(train_images, train_labels)):
#   if i > 0 and (i + 1) % 100 == 0:
#     print(
#       '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
#       (i + 1, loss / 100, num_correct)
#     )
#     loss = 0
#     num_correct = 0
#
#   l, acc = cnn.train(im, label)
#   loss += l
#   num_correct += acc
# #print(output)

# Train the CNN for 3 epochs
for epoch in range(3):
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