import os
from random import shuffle

import PIL
from PIL import Image
from numpy import *


class Datapreprocessor:


    def __init__(self, data_path, save_path):

        self.data_path = data_path
        self.listing = os.listdir(self.data_path)
        self.num_samples = size(self.listing)
        self.save_path = save_path

        print("Number of Images:"+self.num_samples)

    def preprepare_data(self):

        for file in self.listing:
            im = Image.open(self.data_path+"\\"+file)
            img = im.resize((200, 200), resample=PIL.Image.BILINEAR)
            img.save(self.save_path+file, "JPEG")

        imlist = os.listdir(self.save_path)
       # img1 = array(Image.open(self.data_path+"\\prepared\\infected\\"+imlist[0]))

        immatrix = array([array(Image.open(self.save_path+im2)).flatten() for im2 in imlist], 'f')

        label = np.ones((self.num_samples), dtype=int)
        label[0:200] = 0
        label[200:400] = 1

        data, label = shuffle(immatrix, label, random_state=2)
        train_data = [data, label]
        X, Y = (train_data[0], train_data[1])
        return X,Y