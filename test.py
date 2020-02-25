import mnist
import numpy

import CNN
import convolutional
import pool
import softmax


def asd():
    filters = []
    for i in range(3):
        filters.append([[k + j for k in range(5)] for j in range(5)])
    print(filters)
    rows = filters[0][0:3]
    print(len(rows))
    for row in range(len(rows)):
        print(rows[row][0:3])


# asd()

cc = convolutional.Conv(2, 3, 2, 1)
input = numpy.array([[[0, 0, 0, 0, 0, 0, 0],
                      [0, 2, 1, 1, 1, 0, 0],
                      [0, 2, 2, 0, 0, 0, 0],
                      [0, 2, 1, 1, 2, 1, 0],
                      [0, 1, 0, 1, 0, 0, 0],
                      [0, 2, 0, 2, 2, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0]],
                     [[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 2, 1, 0, 0],
                      [0, 1, 0, 2, 2, 1, 0],
                      [0, 1, 1, 2, 2, 2, 0],
                      [0, 2, 2, 0, 2, 0, 0],
                      [0, 2, 2, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]],
                     [[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 2, 0, 0],
                      [0, 1, 2, 0, 2, 0, 0],
                      [0, 1, 1, 2, 0, 2, 0],
                      [0, 0, 0, 0, 1, 2, 0],
                      [0, 1, 1, 2, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]])
input2 = numpy.array([[[2, 1, 1, 1, 0],
                       [2, 2, 0, 0, 0],
                       [2, 1, 1, 2, 1],
                       [1, 0, 1, 0, 0],
                       [2, 0, 2, 2, 1]],
                      [[1, 1, 2, 1, 0],
                       [1, 0, 2, 2, 1],
                       [1, 1, 2, 2, 2],
                       [2, 2, 0, 2, 0],
                       [2, 2, 0, 0, 0]],
                      [[1, 1, 0, 2, 0],
                       [1, 2, 0, 2, 0],
                       [1, 1, 2, 0, 2],
                       [0, 0, 0, 1, 2],
                       [1, 1, 2, 1, 0]]])
filters = []
bias = []
filter = numpy.array([[[1, -1, 1],
                       [0, 0, -1],
                       [-1, 1, 1]],
                      [[1, 0, -1],
                       [-1, -1, 1],
                       [1, -1, 1]],
                      [[1, 1, 1],
                       [-1, -1, 0],
                       [0, 1, -1]]])
filters.append(filter)
bias.append(1)
filter2 = numpy.array([[[0, -1, 0],
                        [0, -1, 0],
                        [0, -1, -1]],
                       [[-1, 1, -1],
                        [1, 0, -1],
                        [-1, 0, 0]],
                       [[-1, 1, 0],
                        [1, -1, -1],
                        [0, -1, 1]]])
filters.append(filter2)
bias.append(0)
#cc.setBias(bias)
#cc.setFilters(filters)
#output = cc.convlove(input2)
#print(output)
pool = pool.Pool()
input3 = numpy.array([[[1, 1, 2, 4],
                       [5, 6, 7, 8],
                       [3, 2, 1, 0],
                       [1, 2, 3, 4]]])
output = pool.maxPooling(input3)
#print(output)
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
cnn = CNN.CNN(10)
conv = convolutional.Conv(8,3)
sm = softmax.Softmax(10, 13 * 13 * 8)
cnn.add(conv)
cnn.add(pool)
cnn.add(sm)
#output = cnn.forwardImage(test_images)
# Train!
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
  if i > 0 and (i + 1) % 100 == 0:
    print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i + 1, loss / 100, num_correct)
    )
    loss = 0
    num_correct = 0

  l, acc = cnn.train(im, label)
  loss += l
  num_correct += acc
#print(output)