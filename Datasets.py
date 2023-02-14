import os
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement
import pandas as pd
import open3d as o3d
from tqdm import tqdm

class PLYDatasetPlaneCount(Dataset):
    def __init__(self, root_dir = 'my_dataset_dir', features=None, labels_file='plane_count.csv', normalize=False):
        super().__init__()
        self.root_dir = root_dir

        if features is None:
            self.features = [0, 1, 2]
        else:
            self.features = features

        self.labelsFile = labels_file
        self.normalize = normalize

        self.dataset = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".ply"):
                self.dataset.append(file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        filename = file.split('.')[0]
        path_to_file = os.path.join(self.root_dir, file)
        labels_path = os.path.join(self.root_dir, self.labelsFile)
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        features = data[:, self.features]

        if self.normalize:
            # XYZ suposed to be 3 first features
            xyz = data[:, [0,1,2]]
            centroid = np.mean(xyz, axis=0)
            xyz -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
            xyz /= furthest_distance
            features[:, [0, 1, 2]] = xyz

        # Get number of planes from csv file
        data_frame = pd.read_csv(labels_path, delimiter=',')
        labels_dict = data_frame.set_index('File')['Planes'].to_dict()
        label = labels_dict.get(int(filename))

        return features, label, filename


class PLYDataset(Dataset):
    def __init__(self, root_dir = 'my_dataset_dir', features=None, labels=None, normalize=False, binary = False, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.weights = []

        if features is None:
            self.features = [0, 1, 2]
        else:
            self.features = features

        if labels is None:
            self.labels = [-1]
        else:
            self.labels = labels

        self.normalize = normalize
        self.binary = binary
        self.transform = transform

        self.dataset = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".ply"):
                self.dataset.append(file)

        print('-'*50)
        print("COMPUTING LABEL WEIGHTS")
        for file in tqdm(self.dataset):
            file = '07782.ply'
            path_to_file = os.path.join(self.root_dir, file)
            ply = PlyData.read(path_to_file)
            data = ply["vertex"].data
            # nm.memmap to np.ndarray
            data = np.array(list(map(list, data)))

            labels = data[:, self.labels]

            if self.binary:
                labels[labels > 0] = 1

            # print(file)
            # COMPUTE WEIGHTS FOR EACH LABEL
            labels = np.sort(labels, axis=None)
            pesos, weights = np.unique(labels, return_counts=True)
            if len(pesos < 2):
                if pesos == 0:
                    weights = np.array([1, 0])
                else:
                    weights = np.array([0, 1])

            weights = weights / len(labels)
            if len(self.weights) == 0:
                self.weights = weights
            else:
                self.weights = np.vstack((self.weights, weights))

        self.weights = np.mean(self.weights, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        path_to_file = os.path.join(self.root_dir, file)
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        features = data[:, self.features]
        labels = data[:, self.labels]

        if self.normalize:
            # XYZ suposed to be 3 first features
            xyz = features[:, [0,1,2]]
            centroid = np.mean(xyz, axis=0)
            xyz -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
            xyz /= furthest_distance
            features[:, [0,1,2]] = xyz

        if self.binary:
            labels[labels > 0] = 1

        # COMPUTE WEIGHTS FOR EACH LABEL
        # labels = np.sort(labels, axis=None)
        # _, weights = np.unique(labels, return_counts=True)
        # weights = weights/len(labels)

        if self.transform is not None:
            features, labels = self.transform(features, labels)

        return features, labels, file.split('.')[0]


class RandDataset(Dataset):
  def __init__(self, n_clouds=50, n_points=3000, n_features=3):
    super(RandDataset, self).__init__()
    # do stuff here?
    self.values = np.random.rand(n_clouds, n_points, n_features)
    self.labels = np.random.rand(n_clouds, n_points)

  def __len__(self):
    return len(self.values)  # number of samples in the dataset

  def __getitem__(self, index):
    return self.values[index], self.labels[index]


if __name__ == '__main__':

    ROOT_DIR = os.path.abspath('/media/arvc/data/datasets/ARVCTRUSS/train/ply_xyzlabelnormal')

    dataset = PLYDataset(root_dir = ROOT_DIR,
                         features = [0, 1, 2],
                         labels = 3,
                         normalize = True,
                         binary = True,
                         transform = None)

    print(dataset.weights)

    # train_loader = torch.utils.data.DataLoader(dataset = dataset,
    #                                            batch_size = 2,
    #                                            shuffle = True,
    #                                            num_workers = 1,
    #                                            pin_memory = True)
    #
    #
    #
    # for i, (points, label, weights, filename) in enumerate(train_loader):
    #     print(f'{weights[0].detach().cpu().numpy()} - {filename[0]}')
    #     print(f'{weights[1].detach().cpu().numpy()} - {filename[1]}')


