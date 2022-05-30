import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTester:

    def __init__(self, filename, model_name):
        self.data = np.load(filename)
        self.x = self.data['x']
        self.y = self.data['y']
        self.model = pickle.load(open(model_name, 'rb'))
        self.prediction = []

    def predict_data(self):
        self.prediction = self.model.predict(self.x)
        accuracy = accuracy_score(self.y, self.prediction)
        print(f'The trained model had an accuracy score of: {accuracy}')

    def visualize_confusion_matrix(self, fsize=18, annotation=True, color_bar=False):
        cm = confusion_matrix(self.y, self.prediction)
        sns.heatmap(cm, annot=annotation, cbar=color_bar)
        plt.title('Confusion matrix', fontsize=fsize)
        plt.show()


if __name__ == "__main__":
    fname = 'live_test.npz'
    best_model = 'LR.p'
    model_fitting = ModelTester(fname, best_model)
    model_fitting.predict_data()
    model_fitting.visualize_confusion_matrix()











