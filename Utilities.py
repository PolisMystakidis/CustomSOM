import numpy as np


# load dataset
def flatten_images(train_img,test_img):
    N1 = train_img.shape[0]
    n = train_img.shape[1]
    m = train_img.shape[2]
    if len(train_img.shape)>3:
        z = train_img.shape[3]
    else:
        z=1
    N2 = test_img.shape[0]

    return np.resize(train_img, (N1, n*m*z)) , np.resize(test_img, (N2, n*m*z))

def unflatten_images(train_img,test_img,end_dims):
    return np.resize(train_img, (train_img.shape[0], end_dims[0] ,  end_dims[1] ,  end_dims[2])), np.resize(test_img, (test_img.shape[0], end_dims[0], end_dims[1],end_dims[2]))