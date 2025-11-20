from SOM import SOM
import os
from matplotlib import pyplot as plt
import time

from tensorflow.keras.datasets.cifar10 import load_data
class_names = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train/255
x_test = x_test/255

x_eval = x_test[:5000]
y_eval = y_test[:5000]

l , w , d = x_train.shape[1:]
n = 100
m = 100
som = SOM(l, w, d, n, m ).cuda()

t0 = time.time()
train_logs , eval_logs = som.learn(x_train,y_train=y_train,epochs=10,eval_data=x_eval,eval=True,eval_labels=y_eval,class_names=class_names,eval_rate=len(x_train))
print("Train time : ",time.time()-t0)
# som.save_weights(os.curdir+f"cifar10_weights{n}x{m}_model")

som.load_weights(os.curdir + f"cifar10_weights{n}x{m}_model")

plt.plot(train_logs)
plt.plot(eval_logs)
plt.show()

# map = som.map_classes(x_eval,y_eval,class_names)
# acc = som.compute_class_accuracy(x_test,y_test,class_names)
# print("Acc on total test set : ",acc)

# som.print_map_matrix()
for i in range(n):
    for j in range(m):
        plt.imshow(som.get_image_on_numpy(i,j))
        plt.show()




