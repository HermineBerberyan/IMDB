import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


class ModelPreparation:

    def __init__(self, filename):
        self.data = np.load(filename)
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.models = {'LR': LogisticRegression, 'SVC': SVC, 'RF': RandomForestClassifier}
        self.predictions = {'LR': [], 'SVC': [], 'RF': []}

    def divide_train_test(self, percent_train=0.3):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data['x'], self.data['y'],
                                                                                test_size=percent_train, random_state=17, stratify=self.data['y'])

    def train_models(self, **parameters):
        for imodel in self.models.keys():
            model = self.models[imodel](**parameters['parameters'][imodel])
            model.fit(self.x_train, self.y_train)
            self.predictions[imodel] = model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, self.predictions[imodel])
            print(f'{imodel} had an accuracy score of {accuracy}')
            pickle.dump(model, open(imodel+".p", "wb"))


if __name__ == "__main__":
    params = {'LR': {'max_iter': 500}, 'SVC': {'kernel': 'linear'}, 'RF': {'n_estimators': 100}}
    fname = 'train_test.npz'
    model_train = ModelPreparation(fname)
    model_train.divide_train_test()
    model_train.train_models(parameters=params)


