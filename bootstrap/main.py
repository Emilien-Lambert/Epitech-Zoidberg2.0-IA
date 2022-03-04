import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


class DatasetsBootStrap:
    mn_data = MNIST('./mnist/')
    images_training, labels_training = mn_data.load_training()
    images_testing, labels_testing = mn_data.load_testing()
    np_images_training = np.array(images_training)
    np_labels_training = np.array(labels_training)
    np_images_testing = np.array(images_testing)
    np_labels_testing = np.array(labels_testing)

    def show_pics(self, value):
        index = self.get_index(value)
        reshape_images = self.np_images_training[index[1]].reshape((28, 28))
        plt.imshow(reshape_images, cmap="Greys")
        plt.show()

    def show_statistics(self):
        occurrences = {}
        for index in range(10):
            occurrences[index] = np.count_nonzero(self.np_labels_training == index)
        total_digits = len(self.np_labels_training)
        print("Nombre d'occurrences :", occurrences)
        print("Nombre total :", total_digits)

    def get_index(self, value):
        index = np.where(self.np_labels_training == value)[0]
        return index

    def mean_digit(self, value):
        index = self.get_index(value)  # Tableau d'index de tous les nombres = value
        total = np.zeros(len(self.np_images_training[0]))  # Tableau de 0 de la taille d'une image
        for item in index:
            total += self.np_images_training[item]
        mean = total / len(index)
        return mean

    def show_mean_pics(self, value):
        mean = self.mean_digit(value)
        reshape_images = mean.reshape((28, 28))
        plt.imshow(reshape_images, cmap="Greys")
        plt.show()


class SklearnSVC:
    mn_data = MNIST('./mnist/')
    images_training, labels_training = mn_data.load_training()
    images_testing, labels_testing = mn_data.load_testing()
    np_images_training = np.array(images_training)
    np_labels_training = np.array(labels_training)
    np_images_testing = np.array(images_testing)
    np_labels_testing = np.array(labels_testing)

    def classification(self):
        # Create a classifier: a support vector classifier
        clf = svm.SVC(verbose=True)

        # Learn the digit on the train subset
        clf.fit(self.np_images_training, self.np_labels_training)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(self.np_images_testing)

        print(f"Classification report for classifier {clf}:\n"
              f"{metrics.classification_report(self.np_labels_testing, predicted)}\n")

        disp = metrics.plot_confusion_matrix(clf, self.np_images_testing, self.np_labels_testing)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")

        plt.show()

class SklearnLinearSVC:


def main():
    # m = DatasetsBootStrap()
    # m.show_pics(value)
    # m.show_statistics()
    # m.mean_digit()
    # m.show_mean_pics(value)

    s = SklearnSVC()
    # s.digits_dataset()
    s.classification()



if __name__ == '__main__':
    main()