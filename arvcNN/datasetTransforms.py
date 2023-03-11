import h5py
import os
from tqdm import tqdm
import numpy as np
from plyfile import PlyData, PlyElement


def h52ply(root_dir, features, binary=True):
    data_path = os.path.abspath(root_dir)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(root_dir), os.pardir))
    file_info = 'all_files.txt'
    path_to_file = os.path.join(data_path, file_info)

    with open(path_to_file) as f:
        file_list = f.readlines()

    total = 0
    actual = 0
    for file in file_list:
        file_path = os.path.join(data_path, file)
        h5f = h5py.File(file_path.strip(), 'r')
        data = np.array(h5f.get('data'))    # (1000x4096x9)
        labels = np.array(h5f.get('label')) # (1000x4096x1)

        # Add labels to the same np array
        data = np.dstack((data, labels))    # (1000x4096x10)

        for i, cloud in enumerate(tqdm(data, desc=os.path.basename(file_path).strip())):
            # cloud = cloud.flatten()
            # cloud = list(map(tuple, cloud))
            name = 'pc_' + str(total + i)
            np2ply(cloud, features, name, binary)
            actual = i

        total = total + actual

    print(f"{total+1} .ply files stored in {parent_dir + '/ply_dataset'}")


def txt2ply(root_dir, features, binary=True):
    data_path = os.path.abspath(root_dir)

    for _, entry in enumerate(tqdm(os.scandir(data_path))):
        filename = os.path.splitext(entry.name)[0]
        cloud = np.loadtxt(entry.path, delimiter=';')
        np2ply(cloud, features, filename, binary)

def ply2np(path_to_file):
    plydata = PlyData.read(path_to_file)
    data = plydata.elements[0].data
    # nm.memmap to np.ndarray
    data = np.array(list(map(list, data)))
    array = data[:,3]
    return array

def np2ply(data_array, out_dir, ply_name, features, binary):

    abs_file_path = os.path.join(out_dir, ply_name + '.ply')

    cloud = list(map(tuple, data_array))
    vertex = np.array(cloud, dtype=features)
    el = PlyElement.describe(vertex, 'vertex')
    if binary:
        PlyData([el]).write(abs_file_path)
    else:
        PlyData([el], text=True).write(abs_file_path)


if __name__ == '__main__':
    # ROOT_DIR = '/media/arvc/data/datasets/S3DIS/indoor3d_sem_seg_hdf5_data/'
    ROOT_DIR = '/home/arvc/PycharmProjects/HKPS/results/'

    # feat_XYZRGBXnYnZn = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('xn', 'f4'),
    #             ('yn', 'f4'), ('zn', 'f4'), ('label', 'i4')]

    feat_XYZRGB = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    txt2ply(ROOT_DIR, feat_XYZRGB, binary=False)




