from mnist import MNIST
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from math import sqrt
from datetime import datetime
import numpy as np
import os
import re
import cv2  # pip install opencv-python
# installer le package sur le PC
# sudo apt-get install graphviz
# installer la lib
# pip install graphviz
import graphviz


def load_path(path):
    listing = [f for f in os.listdir(path) if re.match(r'.*\.jpeg', f)]
    img_table = []
    label_table = []

    # resize img and change rgb to black and white
    for file in listing:
        image = cv2.imread(path + file, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
        image = np.array(image)

        if image.size != 40000:
            rgb_weights = [0.2989, 0.5870, 0.1140]
            image = np.dot(image[..., :3], rgb_weights)
            print(image.shape)
        image = np.reshape(image, 40000)
        image = image.astype('float32')
        img_data = image / 255
        # img_data = img.imread(path + file)
        # img_data = Image.open(path + file)
        img_table.append(img_data)

        if "virus" in file:
            label_table.append("virus")
        elif "bacteria" in file:
            label_table.append("bacteria")
        else:
            label_table.append("normal")
            # TODO : problème de RGB

    np_img = np.array(img_table)
    np_label = np.array(label_table)
    return np_img, np_label


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

        path_img_test = "data ia sorted/all/test/"
        path_img_train = "data ia sorted/all/train/"
        # path_img_validation = "data ia sorted/all/validation/"

        np_img_test, np_label_test = load_path(path_img_test)
        np_img_train, np_label_train = load_path(path_img_train)
        # np_img_validation, np_label_validation = load(path_img_validation)

        data = {
            'np_images_training': np_img_train,
            'np_labels_training': np_label_train,
            'np_images_testing': np_img_test,
            'np_labels_testing': np_label_test
        }
        return data


class ExampleML:

    def __init__(self, data):
        self.np_images_training = data["np_images_training"]
        self.np_labels_training = data["np_labels_training"]
        self.np_images_testing = data["np_images_testing"]
        self.np_labels_testing = data["np_labels_testing"]

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

    def decision_tree_classifier(self):
        # classifier
        # max_depth correspond au nombre de niveau créer dans l'arbre
        # (+ il est elever, plus la precision est bonne jusu'a une certaine valeur, - la visibilite du pdf sera bonne)
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
        clf = clf.fit(self.np_images_training, self.np_labels_training)

        # export to pdf the training tree classification
        dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("Tree_Graph")

        # prediction
        predicted = clf.predict(self.np_images_testing)
        print("Accuracy:", metrics.accuracy_score(self.np_labels_testing, predicted))

    def random_tree_forest(self, estimators):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=estimators)  # plus le nombre est grand, plus les performances seront bonnes mais le code sera ralenti
        clf = clf.fit(self.np_images_training, self.np_labels_training)

        # Predict the response for test dataset
        predicted = clf.predict(self.np_images_testing)
        print("Accuracy:", metrics.accuracy_score(self.np_labels_testing, predicted))

    # -----FONCTIONNEL mais revoir pour des param plus precis-------
    def extremely_randomized_trees(self, estimators):
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_blobs
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.tree import DecisionTreeClassifier


        # clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=2, random_state=0)
        # scores = cross_val_score(clf, self.np_images_training, self.np_labels_training, cv=5)
        # print(scores.mean())

        # ExtraTrees classifier always tests random splits over fraction of features
        # (in contrast to RandomForest, which tests all possible splits over fraction of features)

        # clf = RandomForestClassifier(n_estimators=estimators, max_depth=10, min_samples_split=2, random_state=0)
        # scores = cross_val_score(clf, self.np_images_training, self.np_labels_training, cv=5)
        # print(scores.mean())

        clf = ExtraTreesClassifier(n_estimators=estimators, max_depth=10, min_samples_split=2, random_state=0)
        scores = cross_val_score(clf, self.np_images_training, self.np_labels_training, cv=5)
        print(scores.mean())


def main():
    print("loading datasets : " + str(datetime.now().time()))
    # data = load_dataset('mnist')
    data = load_dataset('pneumonia')

    ml = ExampleML(data)

    print("launch code : " + str(datetime.now().time()))

    # ml.prediction_knn()
    # ml.naive_bayes()
    # ml.decision_tree_classifier()
    # ml.random_tree_forest(100)
    ml.extremely_randomized_trees(5)

    print("finished at : " + str(datetime.now().time()))


if __name__ == '__main__':
    main()
