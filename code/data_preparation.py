import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

class DataInspector:

    def __init__(self, file_path, file_name):
        self.file_name = file_name
        self.file_path = file_path
        self.data = pd.read_csv(os.path.join(self.file_path, self.file_name), sep=',')
        self.X = []
        self.y = []

    def check_dataset(self, n_rows):
        print(self.data.head(n_rows))
        print(f'The original data has the following shape: {self.data.shape}')
        print(f'The number of missing values is: {self.data.isnull().sum()}')
        print(f'Data has the following datatypes: {self.data.dtypes}')

    def show_countplot(self, fsize=18, col_palette=None):
        y_plot = sns.countplot(data=self.data, x='sentiment', palette=col_palette)
        y_plot.set_xlabel('Sentiment polarity labels', fontsize=fsize)
        y_plot.set_xticklabels(['negative', 'positive'])
        plt.show()

    def subset_data(self, subset_ratio):
        rows_subset = int(self.data.shape[0] / subset_ratio)
        self.data = self.data.iloc[:rows_subset, :]
        print(f'The new data has the following shape: {self.data.shape}')

    def change_dtypes(self):
        encoder = LabelEncoder()
        vectorizer = CountVectorizer()
        self.y = encoder.fit_transform(self.data['sentiment'])
        vectorizer.fit(self.data['review'].values)
        self.X = vectorizer.transform(self.data['review']).toarray()

    def inspect_unique_values(self):
        print(f'Unique values for X are: {np.unique(self.X)}')
        print(f'Unique values for y are: {np.unique(self.y)}')

    def split_train_live(self, percent):
        split_idx = int(self.X.shape[0] * percent)
        x_trainset, x_liveset = self.X[:split_idx, :], self.X[split_idx:, :]
        y_trainset, y_liveset = self.y[:split_idx], self.y[split_idx:]
        np.savez('train_test.npz', x=x_trainset, y=y_trainset)
        np.savez('live_test.npz', x=x_liveset, y=y_liveset)


if __name__ == "__main__":
    fpath = '../data/'
    fname = 'IMDB Dataset.csv'
    datafile = DataInspector(fpath, fname)
    datafile.check_dataset(3)
    datafile.show_countplot()
    datafile.subset_data(20)
    datafile.change_dtypes()
    datafile.inspect_unique_values()
    datafile.split_train_live(0.9)











