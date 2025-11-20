import tensorflow as tf
from Autoencoder import Autoencoder
from Utilities import flatten_images , unflatten_images
import numpy as np
from SOM import SOM
import os
from matplotlib import pyplot as plt
import time
import torch

from tensorflow.keras.datasets.mnist import load_data
print("Tensorflow GPU : ",tf.test.is_gpu_available())
class_names = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train/255
x_test = x_test/255

# Reshaping from N x 32 x 32 to N x 32 x 32 x 1
x_train = np.asarray( [np.reshape(x , (x.shape[0],x.shape[1],1)) for x in x_train] )
x_test = np.asarray( [np.reshape(x , (x.shape[0],x.shape[1],1)) for x in x_test] )

y_train = np.asarray( [[y] for y in y_train] )
y_test = np.asarray( [[y] for y in y_test] )

x_train , x_test = flatten_images(x_train,x_test)


#Create autoencoder
start_dims = x_train.shape[1]
end_dims = 12*12

autoencoder = Autoencoder(list_enc_shapes=[start_dims,12*12*3 ,12*12*2 , end_dims], list_dec_shapes=[12*12*2,12*12*3 , start_dims])
autoencoder.generate_autoencoder()

# #Train autoencoder
# t0 = time.time()
# history = autoencoder.fit_autoencoder(x_train, x_train, x_test, x_test, epochs=100, batchsize=32)
# print("Training time : ", time.time() - t0)
# plt.plot(history.history['loss'])
# plt.show()
# autoencoder.save("mnist_autoencoder")


autoencoder.load("mnist_autoencoder")

# Pass dataset to autoencoder
supresed_x_train = autoencoder.encode(x_train)
supresed_x_test = autoencoder.encode(x_test)

# Transform into shape (28,28,1)
sup_im_train = []
for x in supresed_x_train:
    sup_im_train.append(x.reshape((int(np.sqrt(end_dims)),int(np.sqrt(end_dims)),1)))
sup_im_train = torch.tensor(sup_im_train)

sup_im_test = []
for x in supresed_x_test:
    sup_im_test.append(x.reshape((int(np.sqrt(end_dims)),int(np.sqrt(end_dims)),1)))
sup_im_test = torch.tensor(sup_im_test)

# Create SOM
x_eval = sup_im_test[:500]
y_eval = y_test[:500]

n = 150
m = 150
l = int(np.sqrt(end_dims))
w = int(np.sqrt(end_dims))
d = 1
som = SOM(l, w, d, n, m ).cuda()

#Train SOM
t0 = time.time()
train_logs , eval_logs = som.learn(sup_im_train,y_train =y_train ,epochs=10,eval_data=x_eval,eval=True,eval_labels=y_eval,class_names=class_names,eval_rate=len(sup_im_train))
print("Train time : ",time.time()-t0)

som.save_weights(os.curdir+f"mnist_AE_weights{n}x{m}_model")
som.load_weights(os.curdir + f"mnist_AE_weights{n}x{m}_model")

plt.plot(train_logs)
plt.plot(eval_logs)
plt.show()

# som.print_map_matrix()
for i in range(n):
    for j in range(m):
        plt.imshow(som.get_image_on_numpy(i,j))
        plt.show()
