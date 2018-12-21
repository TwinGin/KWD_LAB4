# dodac score, sprawdzic tshirt

import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import seaborn as sns;

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train[0].shape)

images_train = []
for image_train in x_train:
    images_train.append(image_train.flatten())

images_test = []

for image_test in x_test:
    images_test.append(image_test.flatten())

images_train = np.array(images_train)
images_test = np.array(images_test)

fashion_classifier = OneVsRestClassifier(LogisticRegression(verbose=1, max_iter=4))

fashion_classifier.fit(images_train, y_train);
conf_matrix1 = confusion_matrix(y_test, fashion_classifier.predict(images_test))
multi_class_fashion_classifier = LogisticRegression(verbose=1, max_iter=4, multi_class="multinomial", solver="sag")

multi_class_fashion_classifier.fit(images_train, y_train)

conf_matrix2 = confusion_matrix(y_test, multi_class_fashion_classifier.predict(images_test))
sns.heatmap(conf_matrix2)

conf_matrix = confusion_matrix(y_test, multi_class_fashion_classifier.predict(images_test))
from keras.preprocessing import image
image_file = 'tshirt.jpg'
img = image.load_img(image_file,target_size=(28,28))
x = image.img_to_array(img)
print("Image from stock (tshirt)")
print(x.shape)
print(x.flatten())
print(multi_class_fashion_classifier.predict(x.flatten().reshape(1,-1)))
print("Confusion_matrix one vs rest:")
fashion_classifier.score(images_test, y_test)
print(conf_matrix1)
sns.heatmap(conf_matrix1)
print("Confusion_matrix multi:")
multi_class_fashion_classifier.score(images_test, y_test)
print(conf_matrix2)
sns.heatmap(conf_matrix2)