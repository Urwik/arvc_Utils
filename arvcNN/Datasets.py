import os
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement
# import pandas as pd
# import open3d as o3d
# from tqdm import tqdm

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
    def __init__(self, root_dir = 'my_dataset_dir', features=None, labels=None, normalize=False, binary = False, add_range_=False,compute_weights=False):
        super().__init__()
        self.root_dir = root_dir
        self.add_range = add_range_

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

        self.dataset = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".ply"):
                self.dataset.append(file)

        if compute_weights:
            self.weights = []
            # COMPUTE WEIGHTS FOR EACH LABEL IN THE WHOLE DATASET
            print('-'*50)
            print("COMPUTING LABEL WEIGHTS")
            for file in tqdm(self.dataset):
                # READ THE FILE
                path_to_file = os.path.join(self.root_dir, file)
                ply = PlyData.read(path_to_file)
                data = ply["vertex"].data
                data = np.array(list(map(list, data)))

                # CONVERT TO BINARY LABELS
                labels = data[:, self.labels]
                if self.binary:
                    labels[labels > 0] = 1

                labels = np.sort(labels, axis=None)
                k_lbl, weights = np.unique(labels, return_counts=True)
                # SI SOLO EXISTE UNA CLASE EN LA NUBE (SOLO SUELO)
                if k_lbl.size < 2:
                    if k_lbl[0] == 0:
                        weights = np.array([1, 0])
                    else:
                        weights = np.array([0, 1])
                else:
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

        if self.add_range:
            xyz = features[:, [0,1,2]]
            D = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            D = D[:, None]
            features = np.hstack((features, D))

        if self.binary:
            labels[labels > 0] = 1

        # COMPUTE WEIGHTS FOR EACH LABEL
        # labels = np.sort(labels, axis=None)
        # _, weights = np.unique(labels, return_counts=True)
        # weights = weights/len(labels)

        return features, labels, file.split('.')[0]


class minkDataset(Dataset):

    def __init__(self, mode_='train', root_dir = 'my_dataset_dir', features=None, labels=None, normalize=False, binary = False, add_range_=False, voxel_size_=0.05):
        super().__init__()
        self.root_dir = root_dir
        self.add_range = add_range_
        self.coords = []
        self.voxel_size = voxel_size_
        self.mode = mode_

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

        self.dataset = []
        for file in os.listdir(self.root_dir):
            if file.endswith(".ply"):
                self.dataset.append(file)


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
        self.coords = features[:, [0,1,2]]

        if self.normalize:
            # XYZ suposed to be 3 first features
            xyz = features[:, [0,1,2]]
            centroid = np.mean(xyz, axis=0)
            xyz -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
            xyz /= furthest_distance
            features[:, [0,1,2]] = xyz

        if self.add_range:
            xyz = features[:, [0,1,2]]
            D = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            D = D[:, None]
            features = np.hstack((features, D))

        if self.mode == 'test_no_labels':
            return self.coords.astype(np.float32)/self.voxel_size, features.astype(np.float32)
        
        else:
            labels = data[:, self.labels]
            if self.binary:
                labels[labels > 0] = 1

            return self.coords.astype(np.float32) / self.voxel_size, features.astype(np.float32), labels.astype(np.int32)


class vis_minkDataset(Dataset):

    def __init__(self, root_dir = 'my_dataset_dir',  common_clouds_dir='', extend_clouds=[], features=None, labels=None, normalize=False, binary = False, add_range_=False, voxel_size_=0.05):
        super().__init__()
        self.root_dir = root_dir
        self.add_range = add_range_
        self.coords = []
        self.voxel_size = voxel_size_

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

        self.dataset = []
        for file in os.listdir(common_clouds_dir):
            if file.endswith(".ply"):
                self.dataset.append(os.path.join(common_clouds_dir, file))
        if extend_clouds:
            for file in extend_clouds:
                self.dataset.append(os.path.join(self.root_dir, file))


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        path_to_file = os.path.abspath(self.dataset[index]) #type: os.path
        ply = PlyData.read(path_to_file)
        data = ply["vertex"].data
        # nm.memmap to np.ndarray
        data = np.array(list(map(list, data)))

        features = np.copy(data[:, self.features])
        labels = np.copy(data[:, self.labels])
        self.coords = np.copy(features[:, [0,1,2]])

        if self.normalize:
            # XYZ suposed to be 3 first features
            xyz = features[:, [0,1,2]]
            centroid = np.mean(xyz, axis=0)
            xyz -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(xyz) ** 2, axis=-1)))
            xyz /= furthest_distance
            features[:, [0,1,2]] = xyz

        if self.add_range:
            xyz = features[:, [0,1,2]]
            D = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            D = D[:, None]
            features = np.hstack((features, D))

        if self.binary:
            labels[labels > 0] = 1


        return self.coords.astype(np.float32) / self.voxel_size, features.astype(np.float32), labels.astype(np.int32)

class vis_Test_Dataset(Dataset):
    def __init__(self, root_dir = 'my_dataset_dir', common_clouds_dir='', extend_clouds=[], features=None, labels=None, normalize=False, binary = False, add_range_=False, compute_weights=False):
        super().__init__()
        self.root_dir = root_dir
        self.add_range = add_range_

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

        self.dataset = []
        for file in os.listdir(common_clouds_dir):
            if file.endswith(".ply"):
                self.dataset.append(os.path.join(common_clouds_dir, file))
        for file in extend_clouds:
            self.dataset.append(os.path.join(self.root_dir, file))

        if compute_weights:
            self.weights = []
            # COMPUTE WEIGHTS FOR EACH LABEL IN THE WHOLE DATASET
            print('-'*50)
            print("COMPUTING LABEL WEIGHTS")
            for file in tqdm(self.dataset):
                # READ THE FILE
                path_to_file = os.path.join(self.root_dir, file)
                ply = PlyData.read(path_to_file)
                data = ply["vertex"].data
                data = np.array(list(map(list, data)))

                # CONVERT TO BINARY LABELS
                labels = data[:, self.labels]
                if self.binary:
                    labels[labels > 0] = 1

                labels = np.sort(labels, axis=None)
                k_lbl, weights = np.unique(labels, return_counts=True)
                # SI SOLO EXISTE UNA CLASE EN LA NUBE (SOLO SUELO)
                if k_lbl.size < 2:
                    if k_lbl[0] == 0:
                        weights = np.array([1, 0])
                    else:
                        weights = np.array([0, 1])
                else:
                    weights = weights / len(labels)

                if len(self.weights) == 0:
                    self.weights = weights
                else:
                    self.weights = np.vstack((self.weights, weights))

            self.weights = np.mean(self.weights, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path_to_file = os.path.abspath(self.dataset[index]) #type: os.path
        # path_to_file = os.path.join(self.root_dir, file)
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

        if self.add_range:
            xyz = features[:, [0,1,2]]
            D = np.sqrt(np.sum(abs(xyz) ** 2, axis=-1))
            D = D[:, None]
            features = np.hstack((features, D))

        if self.binary:
            labels[labels > 0] = 1

        # COMPUTE WEIGHTS FOR EACH LABEL
        # labels = np.sort(labels, axis=None)
        # _, weights = np.unique(labels, return_counts=True)
        # weights = weights/len(labels)

        return features, labels, os.path.basename(path_to_file).split('.')[0]


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

    ROOT_DIR = os.path.abspath('/media/arvc/data/experiments/ouster/real/entrance_innova/ply_xyznormal')

    dataset = PLYDataset(root_dir = ROOT_DIR,
                         features = [0,1,2,3,4,5],
                         labels = [],
                         normalize = True,
                         binary = False,
                         add_range_=False,
                         compute_weights=False)

    train_loader = torch.utils.data.DataLoader(dataset = dataset,
                                               batch_size = 1,
                                               shuffle = True,
                                               num_workers = 1,
                                               pin_memory = True,
                                               drop_last=True)

    for i, (points, label, filename) in enumerate(train_loader):
        # print(points.size())
        points = points.numpy()
        cloud = points[0]
        normals = cloud[:, [3,4,5]]
        print(normals)


