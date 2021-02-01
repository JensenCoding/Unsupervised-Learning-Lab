import numpy as np
import pandas as pd
import os
from pathlib import Path
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler


class DataLoader:

    def __init__(self, file, output_path, name, seed=7141, test_size=0.2):
        abs_dir = os.path.dirname(__file__)
        self._file = Path(os.path.join(abs_dir, file))
        self._test_size = test_size

        self._data = pd.DataFrame()
        self.output_path = Path(os.path.join(abs_dir, output_path))
        self.seed = seed
        self.data_name = name
        self.X = None
        self.Y = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.num_records = None
        self.num_features = None

        if not os.path.exists(output_path):
            os.mkdir(output_path)

    def load_data(self):
        self._data = pd.read_csv(self._file, header=None)
        self.X = np.array(self._data.iloc[:, :-1])
        self.Y = np.array(self._data.iloc[:, -1])
        self.num_records, self.num_features = self.X.shape
        self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
            self.X, self.Y, test_size=self._test_size, random_state=self.seed, stratify=self.Y
        )

    def scaled_data(self):
        self.X = StandardScaler().fit_transform(self.X)
        self.training_x = StandardScaler().fit_transform(self.training_x)
        self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def get_split_data(self):
        return self.training_x, self.training_y, self.testing_x, self.testing_y

    def get_data(self):
        return self.X, self.Y
