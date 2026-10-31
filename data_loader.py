import numpy as np
import scipy.io as sio
def load_mat(path):

    data = sio.loadmat(path)

    X = np.squeeze(data['X'])

    Y = np.squeeze(data['Y'])


    return X,Y

# GT = load_mat('data/data_reuters.mat')
# print(np.unique(GT, return_counts=True)[1])


