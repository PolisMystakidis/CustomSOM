from SOM import SOM
import os
import time
from matplotlib import pyplot as plt

import numpy as np
from tensorflow.keras.datasets.mnist import load_data

class_names = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train/255
x_test = x_test/255

# Reshaping from N x 32 x 32 to N x 32 x 32 x 1
x_train = np.asarray( [np.reshape(x , (x.shape[0],x.shape[1],1)) for x in x_train] )
x_test = np.asarray( [np.reshape(x , (x.shape[0],x.shape[1],1)) for x in x_test] )

y_train = np.asarray( [[y] for y in y_train] )
y_test = np.asarray( [[y] for y in y_test] )


x_eval = x_test[:500]
y_eval = y_test[:500]

l , w , d = x_train.shape[1:]
n = 100
m = 100
som = SOM(l, w, d, n, m )
t0 = time.time()
train_logs , eval_logs = som.learn(x_train,y_train =y_train ,epochs=10,eval_data=x_eval,eval=True,eval_labels=y_eval,class_names=class_names,eval_rate=15000)
print("Train time : ",time.time()-t0)

som.save_weights(os.curdir+f"mnist_weights{n}x{m}_model")
som.load_weights(os.curdir + f"mnist_weights{n}x{m}_model")


map = som.map_classes(x_train,y_train,class_names)
acc = som.compute_class_accuracy(x_test,y_test,class_names)
print("Acc on total test set : ",acc)

plt.plot(train_logs)
plt.plot(eval_logs)
plt.show()

# som.print_map_matrix()
for i in range(n):
    for j in range(m):
        plt.imshow(som.get_image_on_numpy(i,j))
        plt.show()


