import numpy as np
from sklearn.model_selection import KFold
import Posterior


class Train_Base(object):
    def __init__(self, args):
        self.X_train_full = args['X_train_full']
        self.X_test = args['X_test']
        self.nfolds = args['nfolds']
        self.classifiers = args['classifiers']

    def meta_data_generation(self):
        self.n0 = self.X_train_full.shape[0]

        self.features_X_train_full = self.X_train_full[:, :-1]
        self.labels_y_train_full = self.X_train_full[:, -1].astype(np.int32)

        self.classes = np.unique(self.labels_y_train_full)
        self.n_classes = len(self.classes)
        self.flag = [i for i, clsf in enumerate(self.classifiers) if clsf == 1]
        self.n_classifiers = len(self.flag)

        kf = KFold(n_splits=self.nfolds, shuffle=True)
        self.kf_split = list(kf.split(self.X_train_full))

        # -------------------------- Cross Validation T-Fold -------------------------------------
        self.meta_data_x = []
        self.meta_data_y = []

        for train_id, test_id in self.kf_split:

            sample_test = self.features_X_train_full[test_id, :]
            sample_train = self.features_X_train_full[train_id, :]
            label_train = self.labels_y_train_full[train_id]
            label_test = self.labels_y_train_full[test_id]

            P_tempt = []
            for i in range(self.n_classifiers):
                Pr = Posterior.Posterior(sample_train, label_train, sample_test, self.flag[i], self.n_classes)
                P_tempt.append(Pr)
            self.P = np.array(P_tempt)
            self.meta_data = np.swapaxes(self.P, 0, 1)
            # print('Train cross valid meta-data')
            # TODO: meta_data_x khong dung thu tu ban dau
            self.meta_data_x.extend(self.meta_data)
            self.meta_data_y.extend(label_test)

        self.meta_data_x = np.array(self.meta_data_x)
        self.meta_data_y = np.array(self.meta_data_y)

        return self.meta_data_x, self.meta_data_y

    def test_metadata_generation(self):
        train_models = []
        for i in range(self.n_classifiers):
            model = Posterior.PosteriorModel(self.features_X_train_full,
                                             self.labels_y_train_full, self.flag[i], self.n_classes)
            train_models.append(model)

        n_test = self.X_test.shape[0]
        features_test = self.X_test[:, :-1]
        labels_test = self.X_test[:, -1]

        P_temp = []
        for i in range(self.n_classifiers):
            Pr_test = Posterior.PosteriorTest(train_models[i], features_test)
            P_temp.append(Pr_test)
        self.P_test = np.array(P_temp)
        self.test_md = np.swapaxes(self.P_test, 0, 1)
        return self.test_md
