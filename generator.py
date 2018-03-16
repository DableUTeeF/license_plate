import json
from PIL import Image
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import Iterator
import cv2


class Generator(Iterator):
    def __init__(self, list, imgpath, batch_size=32, img_size=(112, 56),
                 preprocess_function=None):
        if type(list) == str:
            self.list = json.load(open(list, 'r'))
        else:
            self.list = list
        self.imgpath = imgpath
        self.total_batch_seen = 0
        self.batch_size = batch_size
        self.img_size = img_size
        self.preprocess_function = preprocess_function
        super(Generator, self).__init__(len(self.list), batch_size, shuffle=False, seed=None)

    def __getitem__(self, idx):

        x = np.array([], dtype='float32')
        f1 = np.array([], dtype='uint8')
        f2 = np.array([], dtype='uint8')
        f3 = np.array([], dtype='uint8')
        f4 = np.array([], dtype='uint8')
        for i in range(self.batch_size):

            if i + idx * self.batch_size >= len(self.list):
                break
            # print(self.list[idx+i].keys()[0])
            path = self.imgpath + self.list[idx+i].keys()[0] + '.jpg'
            x1, y1 = self.list[idx+i].values()[0][1], self.list[idx+i].values()[0][2]
            x2, y2 = self.list[idx+i].values()[0][3], self.list[idx+i].values()[0][4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            s = cv2.imread(path)
            # img = Image.open(path).convert('RGB')
            img = Image.fromarray(s)
            img = img.crop((x1, y1, x2, y2))
            img = img.resize(self.img_size)
            img = np.array(img, ndmin=3, dtype='float32')
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

            out = self.list[idx+i].values()[0][0]
            f1s = to_categorical(int(out[0]), 10)
            f1s = f1s.reshape((1, f1s.shape[0]))
            f2s = to_categorical(int(out[1]), 10)
            f2s = f2s.reshape((1, f2s.shape[0]))
            f3s = to_categorical(int(out[2]), 10)
            f3s = f3s.reshape((1, f3s.shape[0]))
            f4s = to_categorical(int(out[3]), 10)
            f4s = f4s.reshape((1, f4s.shape[0]))
            if i == 0:
                x = img
                f1, f2, f3, f4 = f1s, f2s, f3s, f4s
            else:
                x = np.concatenate((x, img))
                f1 = np.concatenate((f1, f1s))
                f2 = np.concatenate((f2, f2s))
                f3 = np.concatenate((f3, f3s))
                f4 = np.concatenate((f4, f4s))
        self.total_batch_seen += 1

        if self.preprocess_function is not None:
            x = self.preprocess_function(x)
        return x, [f1, f2, f3, f4]
