import open3d as o3d
import numpy as np

if __name__ == '__main__':
    pcd = np.loadtxt('pcdPython.txt')
    unique, counts = np.unique(pcd[:,3], return_counts=True)
    for i in range(len(unique)):
        print(f'[{unique[i]:.1f} -- {counts[i]:.1f} ]')
