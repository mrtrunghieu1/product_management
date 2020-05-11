import numpy as np
import json
import pickle
import random


def categorical(n0, n_classes, y_label):
    Y = np.zeros((n0, n_classes))
    for i0 in range(n0):
        for m in range(n_classes):
            if y_label[i0] == (m + 1):
                Y[i0][m] = 1
    return Y


def write_file_cv(array, folder, filename):
    array_mean = np.mean(array)
    array_var = np.var(array)
    np.savetxt(folder + "/" + filename, array, delimiter=',', fmt='%0.6e')
    f = open(folder + "/" + filename, "a")
    f.write("----------\n")
    f.write("Mean:\n")
    f.write("{0:6E}\n".format(array_mean))
    f.write("Variance:\n")
    f.write("{0:6E}".format(array_var))
    f.close()


def write_file(result, folder, filename):
    f = open(folder + "/" + filename, 'w')
    f.write(str(result))
    f.close()


def dump_file(result, folder, filename):
    file = open(folder + "/" + filename, 'wb')
    pickle.dump(result, file)
    file.close()


def generate_gauss_noise(sample, n_classifiers, n_classes, seed):
    # set seed so that the output for each sample is fixed
    np.random.seed(seed)

    random_matrix = np.random.normal(0, 0.1, (n_classifiers, n_classes))
    random_binary_matrix = np.random.randint(2, size=(n_classifiers, n_classes))
    gaussian_matrix = sample + random_matrix * random_binary_matrix
    return gaussian_matrix


def generate_randomness(sample, n_classifiers, num_classes, seed):
    np.random.seed(seed)

    random_matrix = np.random.uniform(low=-0.1, high=0.1, size=(n_classifiers, num_classes))
    random_binary_matrix = np.random.randint(2, size=(n_classifiers, num_classes))
    gaussian_matrix = sample + random_matrix * random_binary_matrix
    return gaussian_matrix
