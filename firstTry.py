import os
import re
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

from math import sqrt
from random import randrange
from mnist import MNIST


# from PIL import Image


class Statistics:

    def pca_3d(self, x, y):
        fig = plt.figure()
        pca = PCA(n_components=3)
        pca_res = pca.fit_transform(x)
        ax = plt.axes(projection='3d')
        ax.scatter3D(pca_res[:, 0], pca_res[:, 1], pca_res[:, 2], c=y)
        plt.show()


class Transformation:

    def nystroem(self, x):
        print("--------------- STARTING TRANSFORMATION ---------------")
        start_time = time.time()
        feature_map_nystroem = Nystroem(gamma=0.01, random_state=1, n_components=500)
        data_transformed = feature_map_nystroem.fit_transform(x)
        print("--------------- FINISHED : %s SECONDS ---------------" % (time.time() - start_time))
        return data_transformed

    def pca(self, x, y):
        print("--------------- STARTING TRANSFORMATION ---------------")
        start_time = time.time()
        pca = PCA(n_components=500)
        pca_res = pca.fit_transform(x)
        print("--------------- FINISHED : %s SECONDS ---------------" % (time.time() - start_time))
        return pca_res


class Sklearn:

    def __init__(self, data):
        self.np_images_training = data["np_images_training"]
        self.np_labels_training = data["np_labels_training"]
        self.np_images_testing = data["np_images_testing"]
        self.np_labels_testing = data["np_labels_testing"]

    def get_scores(self, model):
        train_score = model.score(self.np_images_training, self.np_labels_training) * 100
        test_score = model.score(self.np_images_testing, self.np_labels_testing) * 100
        print('\n\n--- Training score : %.3f' % (train_score))
        print('\n--- Testing score : %.3f \n\n' % (test_score))

    def get_predictions(self, model):
        return model.predict(self.np_images_testing)

    def get_matrix(self, model, predicted):
        print(f"Classification report for classifier {model}:\n"
              f"{metrics.classification_report(self.np_labels_testing, predicted)}\n")

        disp = metrics.plot_confusion_matrix(model, self.np_images_testing, self.np_labels_testing)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.show()

    def svc(self):
        clf = svm.SVC(verbose=True)
        clf.fit(self.np_images_training, self.np_labels_training)
        predicted = clf.predict(self.np_images_testing)

        print(f"Classification report for classifier {clf}:\n"
              f"{metrics.classification_report(self.np_labels_testing, predicted)}\n")

        disp = metrics.plot_confusion_matrix(clf, self.np_images_testing, self.np_labels_testing)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.show()

    def svc_linear(self):
        clf = svm.LinearSVC(verbose=True)
        clf.fit(self.np_images_training, self.np_labels_training)
        score = clf.score(self.np_images_training, self.np_labels_training) * 100
        predict = clf.predict(self.np_images_testing)
        i = 0
        good = 0
        for item in predict:
            if item == self.np_labels_testing[i]:
                good += 1
            i += 1
        print((good / self.np_images_testing.size) * 100)

        return score

    def mlp_classifier(self):
        clf = MLPClassifier(verbose=True, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(784, 3), random_state=1)
        clf.fit(self.np_images_training, self.np_labels_training)
        print(clf.predict(self.np_images_testing), '\n', clf.score(self.np_images_testing, self.np_labels_testing))
        return clf.predict(self.np_images_testing), clf.score(self.np_images_testing, self.np_labels_testing)

    def prediction_knn(self):
        # Create KNN Classifier
        k = round(sqrt(self.np_labels_training.size))  # k does be square root of the training set records
        knn = KNeighborsClassifier(n_neighbors=k)  # call the k nearest neighbors

        # Train the model using the training sets
        knn.fit(self.np_images_training, self.np_labels_training)

        # Predict the response for test dataset
        predicted = knn.predict(self.np_images_testing)

        print("Accuracy:", metrics.accuracy_score(self.np_labels_testing, predicted))

    def naive_bayes(self):
        model = GaussianNB()
        # fit the model with the training data
        model.fit(self.np_images_training, self.np_labels_training)

        # predict the target on the train dataset
        predict_train = model.predict(self.np_images_training)
        print('accuracy_score on train dataset : ', accuracy_score(self.np_labels_training, predict_train))

        # predict the target on the test dataset
        predict_test = model.predict(self.np_images_testing)
        print('accuracy_score on test dataset : ', accuracy_score(self.np_labels_testing, predict_test))

    def decision_tree_classifier(self, max_depth):
        # max_depth = nombre de niveau dans l'arbre (+ grand = + precis (jusqu'a un certain point), - graph lisible)
        # entropy = par rapport au gain
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_split=2, random_state=0)
        clf = clf.fit(self.np_images_training, self.np_labels_training)

        # scores = cross_val_score(clf, self.np_images_training, self.np_labels_training, cv=5)
        # print(scores.mean())

        # export to pdf the training tree classification schema
        dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("Tree_Graph")

        predicted = clf.predict(self.np_images_testing)
        print("Accuracy:", metrics.accuracy_score(self.np_labels_testing, predicted))

    def random_tree_forest(self, estimators, max_depth):
        # n_estimators = + le nombre est grand, + les performances seront bonnes mais le code sera ralenti
        clf = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth, min_samples_split=2, random_state=0)
        clf = clf.fit(self.np_images_training, self.np_labels_training)

        # scores = cross_val_score(clf, self.np_images_training, self.np_labels_training, cv=5)
        # print(scores.mean())

        # create graph in a pdf
        # take a random tree in the forest and display it !!!
        estimator = clf.estimators_[randrange(estimators)]
        dot_data = export_graphviz(estimator, out_file=None, filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("Tree_Forest_Graph")

        # Predict the response for test dataset
        predicted = clf.predict(self.np_images_testing)
        print("Accuracy:", metrics.accuracy_score(self.np_labels_testing, predicted))

    def extremely_randomized_trees(self, estimators, max_depth):
        # ExtraTrees classifier always tests random splits over fraction of features
        # (in contrast to RandomForest, which tests all possible splits over fraction of features)

        clf = ExtraTreesClassifier(n_estimators=estimators, max_depth=max_depth, min_samples_split=2, random_state=0)
        clf = clf.fit(self.np_images_training, self.np_labels_training)

        # scores = cross_val_score(clf, self.np_images_training, self.np_labels_training, cv=5)
        # print(scores.mean())

        # Predict the response for test dataset
        predicted = clf.predict(self.np_images_testing)
        print("Accuracy:", metrics.accuracy_score(self.np_labels_testing, predicted))


# resize img and change rgb to black and white
# for file in listing:
#     image = cv2.imread(path + file, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
#     image = np.array(image)
#
#     if image.size != 40000:
#         rgb_weights = [0.2989, 0.5870, 0.1140]
#         image = np.dot(image[..., :3], rgb_weights)
#         print(image.shape)
#     image = np.reshape(image, 40000)
#     image = image.astype('float32')
#     img_data = image / 255
#     # img_data = img.imread(path + file)
#     # img_data = Image.open(path + file)
#     img_table.append(img_data)
#
#     if "virus" in file:
#         label_table.append("virus")
#     elif "bacteria" in file:
#         label_table.append("bacteria")
#     else:
#         label_table.append("normal")
#         # TODO : problème de RGB

# np_img = np.array(img_table)
# np_label = np.array(label_table)
# return np_img, np_label


def crop(img, size):
    middleH = img.shape[0] / 2
    middleW = img.shape[1] / 2

    lowH = middleH - (size / 2)
    maxH = middleH + (size / 2)

    lowW = middleW - (size / 2)
    maxW = middleW + (size / 2)

    cropped = img[int(lowH):int(maxH), int(lowW):int(maxW)]
    return cropped


def data_formatting(path):
    # List all name of images file in a table
    images_name = [f for f in os.listdir(path) if re.match(r'.*\.jpeg', f)]

    # Create table of label and image path
    labels = []
    images_path = []
    counter = 0

    # Create a table of reformatted images
    images = []
    good_images = []
    bad_images = []

    for image_name in images_name:
        img = mpimg.imread(path + image_name)
        if img.ndim == 2:
            cropped = crop(img, 200)
            cropped = cropped / 255
            cropped = np.reshape(cropped, 40000)
            good_images.append(cropped)
            if "virus" in image_name:
                labels.append("virus")
            elif "bacteria" in image_name:
                labels.append("bacteria")
            else:
                labels.append("normal")
        elif img.ndim == 3:
            bad_images.append(img)

    #         # TODO : problème de RGB

    # np_img = np.array(img_table)
    # np_label = np.array(label_table)
    # return np_img, np_label

    # for image_name in images_name:
    #     counter += 1
    #     if "virus" in image_name:
    #         labels.append("virus")
    #         images_path.append(path + image_name)
    #     elif "bacteria" in image_name:
    #         labels.append("bacteria")
    #         images_path.append(path + image_name)
    #     else:
    #         labels.append("normal")
    #         images_path.append(path + image_name)

    # for i in range(len(images_path)):
    #     images.append(mpimg.imread(images_path[i]))
    #     if images[i].ndim == 2:
    #         good_images.append(images[i])
    #     elif images[i].ndim == 3:
    #         bad_images.append(images[i])
    #     i -= 1

    return good_images, labels


def load_dataset(type_data):
    if type_data == 'mnist':
        mn_data = MNIST('./mnist/')
        images_training, labels_training = mn_data.load_training()
        images_testing, labels_testing = mn_data.load_testing()
        data = {
            'np_images_training': np.array(images_training),
            'np_labels_training': np.array(labels_training),
            'np_images_testing': np.array(images_testing),
            'np_labels_testing': np.array(labels_testing)
        }
        return data

    elif type_data == 'pneumonia':

        train_images, train_labels = data_formatting("data ia sorted/all/train/")
        test_images, test_labels = data_formatting("data ia sorted/all/test/")
        validation_images, validation_labels = data_formatting("data ia sorted/all/validation/")

        # plt.imshow(train_images[0], cmap=plt.get_cmap("gray"))
        # plt.show()

        data = {
            'np_images_training': np.array(train_images),
            'np_labels_training': np.array(train_labels),
            'np_images_testing': np.array(test_images),
            'np_labels_testing': np.array(test_labels),
            "np_images_validation": np.array(validation_images),
            "np_labels_validation": np.array(validation_labels),
        }
        return data


def main():
    print("--------------- START LOAD_DATASET ---------------")
    start_time = time.time()
    data = load_dataset('pneumonia')
    print("--------------- FINISH : %s SECONDS ---------------" % (time.time() - start_time))

    models = Sklearn(data)

    # visualize = Statistics()
    # visualize.pca_3d(models.np_images_training, models.np_labels_training)

    # Les transform
    print("--------------- START TRANSFORM ---------------")
    # start_time = time.time()
    # transform = Transformation()
    # train_data_transform = transform.nystroem(models.np_images_training)
    # test_data_transform = transform.nystroem(models.np_images_testing)
    # models.np_images_training = train_data_transform
    # models.np_images_testing = test_data_transform
    print("--------------- FINISH : %s SECONDS ---------------" % (time.time() - start_time))

    # Les algo
    print("--------------- START TRAINING ---------------")
    start_time = time.time()
    # models.svc()
    # print(models.svc_linear())
    # my_model = models.mlp_classifier()

    # models.prediction_knn()
    # models.naive_bayes()
    # models.decision_tree_classifier(5)
    models.random_tree_forest(100, 5)
    # models.extremely_randomized_trees(100, 10)

    print("--------------- FINISH : %s SECONDS ---------------" % (time.time() - start_time))

    # Les stats
    # models.get_scores(my_model)
    # predictions = models.get_predictions(my_model)
    # models.get_matrix(my_model, predictions)


if __name__ == '__main__':
    main()
