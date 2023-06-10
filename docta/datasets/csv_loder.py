import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TabularDataset(object):
    def __init__(self, root_path, preprocess = None):
        self.root_path = root_path
        self.df = pd.read_csv(self.root_path)
        self.preprocess = preprocess 
        self.preprocess_csv()
        # self.index = range(len(self.feature))
        self.index = np.array(range(len(self.feature)))


    def preprocess_csv(self, normalize=True):
        """
        This function servers to preporcess your dataset for docta use.
        The code below will do some basic cleaning stuff, then return you the features and the labels.
        Note: this preprocess function is case by case, so please update/revise accoridngly for your personal use.
        Reference:
        Docta will automatically view the column "target" as labels of samples;
        Remaining columns will be the features of samples.
        """
        self.feature = self.df.drop(['target'], axis=1).values
        self.label = self.df.target.values
        sc = StandardScaler()
        if normalize:
            self.feature = sc.fit_transform(self.feature)
        return self.feature, self.label

    def __getitem__(self, index):
        idx = self.index[index]
        feature = self.feature[idx]
        
        if self.preprocess is not None:
            feature = self.preprocess(feature)
            if feature.shape[0] == 1: # output of tokenizer
                feature = feature.reshape(feature.shape[1:])

        return feature, self.label[idx], idx
    
    def __len__(self) -> int:
        return len(self.feature)


def onehot(df, cols):
    dummies = [pd.get_dummies(df[col]) for col in cols]
    df.drop(cols, axis=1, inplace=True)
    df = pd.concat([df] + dummies, axis=1)
    return df