import numpy as np
import itertools, random
import math
# from keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier

import util
from model.model_cnn import get_model, calc_mf1
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from model.model_vgg16 import vgg16_model
from util import write_file, dump_file


def data_augmentation_factorial(X, y, algo_list):
    # Create new K! * N samples
    li1 = list(itertools.permutations(X[0], len(algo_list)))
    X_new = np.array(li1)
    y_new = [y[0]] * math.factorial(len(algo_list))  # Caculate factorial of desired number.
    for i in range(1, len(X)):
        li = list(itertools.permutations(X[i], len(algo_list)))
        np_li = np.array(li)
        X_new = np.vstack((X_new, np_li))
        y_new.extend([y[i]] * math.factorial(len(algo_list)))
    return X_new, y_new


def selection_random_sample(X_fact, y_fact, num_samples):
    np.random.seed(4)
    perm = np.random.permutation(X_fact.shape[0])
    perm = perm[:num_samples]

    new_samples = X_fact[perm, :, :]
    y_fact_np = np.array(y_fact)
    new_labels = y_fact_np[perm]

    return new_samples, new_labels


def data_augmentation(metadata_x, y, num_classifier, num_samples, num_classes, method, seed):
    np.random.seed(seed)
    MAX_SAMPLES = 10000

    choices = np.random.choice(metadata_x.shape[0], MAX_SAMPLES)
    choices = choices[:num_samples]

    new_samples = []
    new_labels = []
    for i in range(num_samples):
        id = choices[i]
        if 'GN' in method:
            matrix = util.generate_gauss_noise(metadata_x[id, :, :], num_classifier, num_classes, seed=i)
        elif 'RN' in method:
            matrix = util.generate_randomness(metadata_x[id, :, :], num_classifier, num_classes, seed=i)
        new_samples.append(matrix)
        new_labels.append(y[id])

    return new_samples, new_labels


def split_per_class(train_md_x, train_md_y):
    classes = np.unique(train_md_y)

    res_x = []
    res_y = []
    for c in classes:
        ids = (train_md_y == c)
        res_x.append(train_md_x[ids])
        res_y.append(train_md_y[ids])

    return res_x, res_y


# ================================== CNN Combining ======================================================


def model_combination(train_md_x, train_md_y, test_md_x, test_md_y, e, num_samples, aug_sel, num_classes, model):
    if aug_sel == "Fact":
        # Factorial Augmentation
        X_new, y_new = data_augmentation_factorial(train_md_x, train_md_y, e)
        X_train_new, y_train_new = selection_random_sample(X_new, y_new, num_samples)
    if aug_sel == "GN" or aug_sel == "RN":
        # Gaussian Noise Augmentation
        X_train_new, y_train_new = data_augmentation(train_md_x, train_md_y, len(e),
                                                     num_samples, num_classes, method=aug_sel, seed=1)

    if aug_sel == "Balanced_Fact":
        # 1. Split train_md_x based on train_md_y --> M sets
        md_x_per_class, md_y_per_class = split_per_class(train_md_x, train_md_y)

        X_train_new = None
        y_train_new = None

        # 2. Augment each of M sets
        for i_c in range(len(md_x_per_class)):
            X_fact, y_fact = data_augmentation_factorial(md_x_per_class[i_c], md_y_per_class[i_c], e)
            X_sample, y_sample = selection_random_sample(X_fact, y_fact, num_samples)
            X_train_new = X_sample if X_train_new is None \
                else np.concatenate((X_train_new, X_sample), axis=0)
            y_train_new = y_sample if y_train_new is None \
                else np.concatenate((y_train_new, y_sample))

        num_samples = X_train_new.shape[0]

    if aug_sel == "Balanced_GN" or aug_sel == "Balanced_RN":
        # 1. Split train_md_x based on train_md_y --> M sets
        md_x_per_class, md_y_per_class = split_per_class(train_md_x, train_md_y)

        X_train_new = None
        y_train_new = None

        # 2. Augment each of M sets
        for i_c in range(len(md_x_per_class)):
            X_sample, y_sample = data_augmentation(md_x_per_class[i_c], md_y_per_class[i_c],
                                                   len(e), num_samples, num_classes,
                                                   method=aug_sel, seed=i_c)

            X_train_new = X_sample if X_train_new is None \
                else np.concatenate((X_train_new, X_sample), axis=0)
            y_train_new = y_sample if y_train_new is None \
                else np.concatenate((y_train_new, y_sample))

        num_samples = X_train_new.shape[0]

    if aug_sel == "No":
        X_train = train_md_x
        y_train = train_md_y

    if aug_sel != "No":
        X_train_new = np.reshape(X_train_new, (num_samples, len(e), num_classes))

        X_train = np.vstack((X_train_new, train_md_x))
        y_train = np.append(y_train_new, train_md_y)

    if model == 'fcnn':
        X_train = X_train.reshape(X_train.shape[0], len(e) * num_classes)
        X_test = test_md_x.reshape(test_md_x.shape[0], len(e) * num_classes)
        clf = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=1e-5, random_state=1)
        clf.fit(X_train, y_train)
        predict_label = clf.predict(X_test)

    else:
        X = X_train.reshape(X_train.shape[0], len(e), num_classes, 1)
        X_test = test_md_x.reshape(test_md_x.shape[0], len(e), num_classes, 1)

        # Reshaping of y
        Y = util.categorical(n0=X_train.shape[0], n_classes=num_classes, y_label=y_train)
        lable_y = util.categorical(n0=len(test_md_y), n_classes=num_classes, y_label=np.array(test_md_y))
        # print('validate_y',validate_y.shape)
        if model == 'cnn':
            clf = get_model(len(e), num_classes)
        elif model == 'vgg16':
            clf = vgg16_model(len(e), num_classes)

        clf.fit(X, Y, epochs=10, batch_size=16, verbose=1)
        # predictions = clf.predict(X_test, batch_size=16)
        predict_class = clf.predict_classes(X_test, batch_size=16)
        predict_label = [i + 1 for i in predict_class]

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_md_y, predict_label, average='macro')
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(test_md_y, predict_label, average='micro')
    accuracy = accuracy_score(test_md_y, predict_label)

    print('ACCURACY', accuracy)

    return p_macro, r_macro, f1_macro, p_micro, r_micro, \
           f1_micro, accuracy, X_train, y_train
