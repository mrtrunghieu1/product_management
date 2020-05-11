import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import datetime
import scipy.io as sio
import numpy as np
import os
import pickle
import argparse
# from util import write_filconda install -c conda-forge kerase
from data_helper import file_list, data_folder, cv_folder
from util import write_file, dump_file
from train_base import Train_Base
from data_augmentation import model_combination

try:
    from_id = int(sys.argv[1])
    to_id = int(sys.argv[2])
except:
    from_id = 0
    to_id = len(file_list)

# model = sys.argv[3]
model = 'cnn'
'''Choose DeepLearing model: 1.cnn combining or 2.vgg16 or 3. fncc (fully connected nn) combining '''

for i_file in range(from_id, to_id):
    file_name = file_list[i_file]
    print(datetime.datetime.now(), ' File {}: '.format(i_file), file_name)

    '''-------------------------Data Loader -----------------------'''
    D_train = np.loadtxt(data_folder + '/train1/' + file_name + '_train1.dat', delimiter=',')
    D_val = np.loadtxt(data_folder + '/val/' + file_name + '_val.dat', delimiter=',')
    D_test = np.loadtxt(data_folder + '/test/' + file_name + '_test.dat', delimiter=',')

    X_train_full = np.concatenate((D_train, D_val), axis=0)
    X_test = D_test

    '''----------------------------- Initial parameters --------------------------------'''
    label_X_test = X_test[:, -1].astype(np.int32)
    classes = np.unique(label_X_test)
    n_classes = len(classes)
    binary_classifiers = [1, 1, 1, 1, 1, 0]  # 1:Activate | 0:Deactivate  Classifiers
    n_folds = 5
    flag = [i for i, clsf in enumerate(binary_classifiers) if clsf == 1]
    review_flag = False

    base_parameters = {
        'X_train_full': X_train_full,
        'X_test': X_test,
        'nfolds': n_folds,
        'classifiers': binary_classifiers,
    }
    '''----------------------- Meta data generation--------------------------------'''
    metadata_folder = "Meta_data/{}/Without_Augmentation".format(file_name)
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder)

    if review_flag is False:
        train_base = Train_Base(base_parameters)
        train_meta_data, md_y = train_base.meta_data_generation()
        test_meta_data = train_base.test_metadata_generation()

        dump_file(train_meta_data, metadata_folder, 'train_meta_data.pkl')
        dump_file(md_y, metadata_folder, 'label_meta_data.pkl')
        dump_file(test_meta_data, metadata_folder, 'test_meta_data.pkl')
    else:
        train_md = open(metadata_folder + '/train_meta_data.pkl', 'rb')
        lable_file = open(metadata_folder + '/label_meta_data.pkl', 'rb')
        test_md = open(metadata_folder + '/test_meta_data.pkl', 'rb')
        train_meta_data = pickle.load(train_md)
        md_y = pickle.load(lable_file)
        test_meta_data = pickle.load(test_md)

    '''----------------------Training without Augmentation-----------------------------'''
    p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro, accuracy, _, _ = model_combination(
        train_md_x=train_meta_data,
        train_md_y=md_y, test_md_x=test_meta_data, test_md_y=label_X_test,
        e=flag, num_samples=0, aug_sel='No', num_classes=n_classes, model=model)
    result_folder = "result/{}/Without_Augmentation".format(file_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    write_file(accuracy, result_folder, 'accuracy')

    write_file(p_macro, result_folder, 'precision_macro')
    write_file(r_macro, result_folder, 'recall_macro')
    write_file(f1_macro, result_folder, 'f1_macro')

    write_file(p_micro, result_folder, 'precision_micro')
    write_file(r_micro, result_folder, 'recall_micro')
    write_file(f1_micro, result_folder, 'f1_micro')

    '''----------------------Training with Augmentation-----------------------------'''
    samples = np.array([100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
    # aug_sels = ["Fact", "GN", "RN"]
    aug_sels = ["Balanced_GN", "Balanced_RN", "Balanced_Fact", "Fact", "GN", "RN"]
    # aug_sels = ["GN"]
    mkdir = ['result', 'Meta_data']
    for dir in mkdir:
        folder_aug = "{}/{}/Data_Augmentation".format(dir, file_name)
        if not os.path.exists(folder_aug):
            os.makedirs(folder_aug)
        for aug in aug_sels:
            dir_aug = folder_aug + "/{}".format(aug)
            if not os.path.exists(dir_aug):
                os.makedirs(dir_aug)

    for aug_sel in aug_sels:
        dt_result = 'result/{}/Data_Augmentation/{}/'.format(file_name, aug_sel)
        dt_md = 'Meta_data/{}/Data_Augmentation/{}/'.format(file_name, aug_sel)
        for aug_num_samples in samples:
            # CNN with augmentation
            dir_sample = dt_md + '{}'.format(aug_num_samples)
            if not os.path.exists(dir_sample):
                os.makedirs(dir_sample)

            file_aug = {"Number_Samples": aug_num_samples}

            p_macro, r_macro, f1_macro, p_micro, r_micro, \
            f1_micro, accuracy, X, y = model_combination(
                train_md_x=train_meta_data, train_md_y=md_y, test_md_x=test_meta_data,
                test_md_y=label_X_test, e=flag, num_samples=aug_num_samples,
                aug_sel=aug_sel, num_classes=n_classes, model=model)

            file_aug.update({'p_macro': p_macro, 'r_macro': r_macro, 'f1_macro': f1_macro,
                             'p_micro': p_micro, 'r_micro': r_micro, 'f1_micro': f1_micro, 'accuracy': accuracy})

            write_file(file_aug, dt_result, '{}'.format(aug_num_samples))
            dump_file(X, dir_sample, 'train_meta_data.pkl'.format())
            dump_file(md_y, dir_sample, 'label_meta_data.pkl')
