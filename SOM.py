import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.decomposition import PCA
import time

def nd_distance(rgb1, rgb2):
    flat1 = torch.flatten(rgb1)
    flat2 = torch.flatten(rgb2)
    diff = flat1-flat2
    d = torch.sqrt(diff.matmul(diff.T))
    return d

class SOM(nn.Module):
    def __init__(self , length , width , depth , n , m , learning_rate = 1,lr_decay = 1.001):
        super(SOM, self).__init__()
        self.width = width
        self.depth = depth
        self.length = length
        self.n = n
        self.m = m
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.weights = torch.zeros(n,m,length , width , depth).cuda()
        self.col_matrix = torch.Tensor([[j for j in range(self.m)] for _ in range(self.n)]).cuda()
        self.row_matrix = self.col_matrix.T
        self.data_std = None
        self.data_mean = None
        self.dw = None
        self.distance_f = torch.nn.PairwiseDistance(p=2)



    def get_image_on(self,m,n):
        return self.weights[n][m]
    def get_image_on_numpy(self,n,m):
        cpu_w = torch.tensor(self.weights[n][m]).cpu()
        return cpu_w.numpy()
    def get_weights(self):
        images = []
        for i in range(self.n):
            images.append([])
            for j in range(self.m):
                images[i].append(self.weights[i][j])
        return images
    def get_weights_numpy(self):
        images = []
        for i in range(self.n):
            images.append([])
            for j in range(self.m):
                images[i].append(self.get_image_on_numpy(i,j))
        return images
    def print_map_matrix(self):
        fig = plt.figure(figsize=(self.n, self.m))
        grid = ImageGrid(fig, 111,  nrows_ncols=(self.n, self.m), axes_pad=0.1,)
        flat = [self.get_image_on_numpy(i, j) for j in range(self.m) for i in range(self.n)]
        for ax, im in zip(grid, flat):
            ax.imshow(im)

        plt.show()

    def forward(self, x):
        ret =[]
        for im in x:
            x , y = self.predict_class_index(im)
            ret.append(self.weights[x][y])
        return ret

    def backpropagate(self,x):
        index = self.predict_class_index(x)
        h_mx = self.compute_h_matrix(index[0], index[1], self.data_std)
        d = x - self.weights
        h_flat = torch.flatten(h_mx)
        h_mx_temp = h_flat.repeat(self.depth * self.length * self.width).reshape(self.depth * self.length * self.width,self.n*self.m).T
        temp_dw = self.lr * h_mx_temp * torch.reshape(d , (self.n*self.m , self.depth * self.length * self.width) )
        self.dw = torch.reshape(temp_dw,(self.n,self.m, self.length , self.width,self.depth))
        self.weights = torch.add(self.weights, self.dw)

    def init_weights(self,data):
        flat_dim=1
        for s in data.shape[1:]:
            flat_dim = flat_dim * s
        flat_data = np.reshape(data , (data.shape[0],flat_dim))
        eivalues = PCA(10).fit_transform(flat_data.T).T
        for i in range(10*eivalues.shape[0]):
            for j in range(10*eivalues.shape[1]):
                rand_i = np.random.randint(0, self.n)
                rand_j = np.random.randint(0, self.m)
                self.weights[rand_i,rand_j] =  torch.tensor( eivalues[i % eivalues.shape[0]][j % eivalues.shape[1]] )

    def learn(self,x_train,epochs,eval=False,y_train = None,eval_data = None , eval_labels=None, class_names=None , eval_rate = 100):
        N_data = x_train.shape[0]
        self.data_std = torch.std(torch.Tensor(x_train)).item()
        self.data_mean = torch.mean(torch.Tensor(x_train)).item()
        print("Initilizing Weights ...")
        self.init_weights(x_train)
        steps = 0

        eval_logs = None
        train_logs = None
        if eval:
            eval_logs = []
            train_logs = []


        print("Learning map")
        self.dw = torch.zeros((self.n, self.m, self.length, self.width, self.depth)).cuda()
        x_train = torch.tensor(x_train).cuda()
        for epoch in range(epochs):
            i_data = 0
            for x in x_train:
                self.backpropagate(x)
                self.lr = 0.9*(1-steps/( 2 * len(x_train)*epochs))
                i_data += 1
                steps+=1
                if steps % 100 == 0:
                    print("Epoch : ", epoch, "/", epochs, "Progress : ", i_data, "/", N_data, " | ",
                          (i_data / N_data * 100).__round__(2), "%")
                if eval and steps % eval_rate == 0:
                    print("Evaluation...")
                    self.map_classes(x_train, y_train, class_names)
                    train_acc = self.compute_class_accuracy(x_train,y_train,class_names)
                    print("Class prediction accuracy for train : ", train_acc)
                    train_logs.append(train_acc)
                    if not eval_data is None and not eval_labels is None:
                        eval_acc = self.compute_class_accuracy(eval_data,eval_labels,class_names)
                        print("Class prediction accuracy for evaluation : ", eval_acc)
                        eval_logs.append(eval_acc)

        return train_logs,eval_logs

    def compute_class_accuracy(self,data,labels,class_names):
        t_data = torch.tensor(data).cuda()
        pred = self.predict_class(data)
        acc = 0
        for i in range(len(data)):
            if pred[i] == class_names[labels[i][0]]:
                acc += 1
        acc = acc / len(data)
        return acc



    def compute_h_matrix(self,maxi,maxj,std):
        row_dist = (self.row_matrix-maxi)
        row_dist = torch.mul(row_dist,row_dist)
        col_dist = (self.col_matrix-maxj)
        col_dist = torch.mul(col_dist,col_dist)
        dist = row_dist+col_dist
        dist = torch.sqrt(dist)
        h = torch.exp( - torch.div(dist,2*std))
        return h

    def predict_class_index(self,x):
        x = torch.tensor(x)
        dists = torch.sqrt(self.distance_f(torch.flatten(x),torch.reshape(self.weights,(self.n,self.m,self.length*self.width*self.depth)).reshape((self.n*self.m,self.length*self.width*self.depth))))
        #dists = torch.stack([nd_distance(x, self.weights[i][j]) for i in range(self.n) for j in range(self.m)])
        min_sim = torch.argmin(dists).item()
        index = (min_sim % self.n, min_sim // self.n)
        return index
    def save_weights(self,path):
        torch.save(self.weights,path)
    def load_weights(self,path):
        self.weights = torch.load(path)

    def map_classes(self,data,labels,class_names):
        indexed = self.get_indexes_per_newron(data,labels,class_names)
        self.class_map = [[class_names[np.argmax(indexed[i][j])] for j in range(len(indexed[0]))] for i in range(len(indexed))]
        return self.class_map
    def get_indexes_per_newron(self,data,labels,class_names):
        indexed = [[[0 for _ in range(len(class_names))] for _ in range(self.m)] for _ in range(self.n)]
        i = 0
        t_data = torch.tensor(data).cuda()
        for x, y in zip(t_data, labels):
            index = self.predict_class_index(x)
            indexed[index[0]][index[1]][y[0]] += 1
            i += 1
        return indexed

    def predict_class(self,data):
        t_data = torch.tensor(data).cuda()
        indexes = [self.predict_class_index(d) for d in t_data]
        classes = [self.class_map[index[0]][index[1]] for index in indexes]
        return classes


