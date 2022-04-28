import os
import csv
import h5py
import socket
import visdom
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

import torch
import random
import numpy as np
from skimage import io
import cv2
import torch.nn.functional as F
import itertools
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from PIL import Image
import sys
import gc
import objgraph
 
'''LABELS = ['other','industrial land', 'urban residential', 'rural residential', 'traffic land', 'paddy field',
                      'irrigated land', 'dry cropland', 'garden plot', 'arbor woodland', 'shrub land',
                      'natural grassland', 'artificial grassland', 'river', 'lake', 'pond'] # Label names
palette = {0: (0, 0, 0),  # 其他
               1: (200, 0, 0),  # industrial land
               2: (250, 0, 150),  # urban residential
               3: (200, 150, 150),  # rural residential
               4: (250, 150, 150),  # traffic land
               5: (0, 200, 0),  # paddy field
               6: (150, 250, 0),#irrigated land
               7: (150, 200, 150), # dry cropland
               8: (200, 0, 200),#garden plot
               9: (150, 0, 250),#arbor woodland
               10: (150,150,250),#shrub land
               11: (250,200,0),#natural grassland
               12: (200,  200,   0),#artificial grassland
               13: (0,   0, 200),#river
               14:(0,     150, 200),#lake
               15:(0,     200, 250)#pond
               }       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}'''
#GID5
LABELS = ['other','built-up', 'farmland', 'forest', 'meadow', 'water'] # Label names
palette = {0: (0, 0, 0),  # 其他
               1: (255, 0, 0),  # built-up
               2: (0, 255, 0),  # farmland
               3: (0, 255, 255),  # forest
               4: (255, 255, 0),  # meadow
               5: (0, 0, 255)  # water
               }       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}
#postdam
'''LABELS = ['background','Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car'] # Label names
palette = {0: (255, 0, 0),  # 其他
               1: (255, 255, 255),  # Impervious surfaces
               2: (0, 0, 255),  # Building
               3: (0, 255, 255),  # Low vegetation
               4: (0, 255, 0),  # Tree
               5: (255, 255, 0)  # Car
               }       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}'''
def mkdir_p(path):
    """
    make dir if not exist
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def preprocessing(data,color_augmentation=True):
    """
    Image preprocess for both train set and test set
    :param data:
    :return:
    """
    assert (len(data.shape) == 3)
    assert (data.shape[0] == 3)
    '''imgs = rgb2gray(data)
    imgs = dataset_normalized(imgs)
    imgs = clahe_equalized(imgs)
    imgs = adjust_gamma(imgs, 1.2)
    imgs = imgs / 255.'''
    if color_augmentation:
        data=data.transpose((1,2,0))
        imgs = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY )
        imgs = imgs/255.
        imgs =np.reshape(imgs,(1,imgs.shape[0],imgs.shape[1]))
    else:
        imgs = data/255.
        '''data=data.transpose((1,2,0))
        imgs = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY )
        imgs = dataset_normalized(imgs)
        imgs = clahe_equalized(imgs)
        imgs = adjust_gamma(imgs, 1.2)
        imgs = imgs / 255.'''
    '''imgs=imgs.transpose((1,2,0))
    imgs = transforms.ToTensor()(imgs)
    imgs = transforms.Normalize(0.5, 0.5)(imgs)'''
    return imgs


def rgb2gray(rgb):
    assert (len(rgb.shape) == 3)  # 4D arrays
    assert (rgb.shape[0] == 3)
    bn_imgs = rgb[0, :, :] * 0.299 + rgb[1, :, :] * 0.587 + rgb[ 2, :, :] * 0.114
    return bn_imgs


def dataset_normalized(imgs):
    """
    normalize over the dataset
    :param imgs:
    :return:
    """
    assert (len(imgs.shape) ==3)  # 4D arrays
    assert (imgs.shape[0] == 1)  # check the channel is 1
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    
    imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized) - np.min(imgs_normalized))) * 255
    return imgs_normalized


def clahe_equalized(imgs):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization.
    After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
    :param imgs:
    :return:
    """
    assert (len(imgs.shape) == 3)  # 4D arrays
    assert (imgs.shape[0] == 1)  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    imgs_equalized= clahe.apply(np.array(imgs, dtype=np.uint8))
    return imgs_equalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 3)  # 4D arrays
    assert (imgs.shape[0] == 1)  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = cv2.LUT(np.array(imgs, dtype=np.uint8), table)
    return new_imgs


def histo_equalized(imgs):
    """
    histogram equalization
    :param imgs:
    :return:
    """
    assert (len(imgs.shape) == 3)  # 4D arrays
    assert (imgs.shape[0] == 1)  # check the channel is 1
    imgs_equalized = cv2.equalizeHist(np.array(imgs, dtype=np.uint8))
    return imgs_equalized

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d
def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, reduction='mean'):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, reduction='mean')
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, reduction='mean')
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, label_values=LABELS):
    label=np.arange(0,len(label_values),1)
    cm = confusion_matrix(
        gts,
        predictions,
        labels=label)

    # print("Confusion matrix :")
    # print(cm)

    # print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    # print("{} pixels processed".format(total))
    # print("Total accuracy : {}%".format(accuracy))

    # print("---")
    np.seterr(divide="ignore", invalid="ignore")
    MIoU = np.diag(cm) / (
                np.sum(cm, axis=1) + np.sum(cm, axis=0) -
                np.diag(cm))
    np.seterr(divide="warn", invalid="warn")
    miou = np.nanmean(MIoU)

    
    # Compute F1 score
    F1Score = []
    for i in range(len(label_values)):
        if (np.sum(cm[i, :]) + np.sum(cm[:, i]))==0:
            continue
        else:
            F1Score.append(2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i])))
    '''F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            if (np.sum(cm[i, :]) + np.sum(cm[:, i]))==0:
                F1Score[i] = 1
            else:    
                F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass'''
    # print("F1Score :")
    # for l_id, score in enumerate(F1Score):
        # print("{}: {}".format(label_values[l_id], score))

    # print("---")
    F1Score = np.asarray(F1Score)
    MF1 = np.nanmean(F1Score)
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe);
    return accuracy,kappa,miou,MF1

class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, DATA_FOLDER, LABEL_FOLDER,WINDOW_SIZE,patch_num):
        super(Train_dataset, self).__init__()
        self.WINDOW_SIZE = WINDOW_SIZE
        self.patch_num = patch_num
        self.data_files = []
        self.label_files = []
        dirs = os.listdir(DATA_FOLDER)
        for file in dirs:
            if file[-4:]=='.jpg':
                self.data_files.append(DATA_FOLDER+file)
                self.label_files.append(LABEL_FOLDER+file[:-4]+'_label.jpg') #other dataset
                #self.label_files.append(LABEL_FOLDER+file[:-8]+'_label.jpg') #postdam

        self.imgs,self.gts = self.get_patch()
    def get_patch(self):
        img_patch=[]
        gt_patch= []
        
        per_num = self.patch_num//len(self.data_files)
        for i in range(len(self.data_files)):
            data = np.asarray(preprocessing(io.imread(self.data_files[i]).transpose((2,0,1))), dtype='float32')
            label = np.asarray(convert_from_color(cv2.cvtColor(cv2.imread(self.label_files[i]),cv2.COLOR_BGR2RGB)), dtype='int64')
            if i==len(self.data_files)-1:
                for j in range(per_num+(self.patch_num % len(self.data_files))):
                    x1, x2, y1, y2 = get_random_pos(data, self.WINDOW_SIZE)
                    img_patch.append(data[:, x1:x2,y1:y2])
                    gt_patch.append(label[x1:x2,y1:y2])  
            else:
                for j in range(per_num):
                    x1, x2, y1, y2 = get_random_pos(data, self.WINDOW_SIZE)
                    img_patch.append(data[:, x1:x2,y1:y2])
                    gt_patch.append(label[x1:x2,y1:y2])  
            del (data,label)
            print("img patch memory:",sys.getsizeof(img_patch))
            print("gt patch memory:",sys.getsizeof(gt_patch))
            gc.collect()
        return np.asarray(img_patch),np.asarray(gt_patch)
    def __len__(self):
        return self.imgs.shape[0]

    def data_augmentation(cls, *arrays, flip=True, mirror=True): #*表示接收的参数用元组表示
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip: #水平翻转
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror: #垂直翻转
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self,idx):
    
        img = self.imgs[idx]
        gt = self.gts[idx]
        
        img,gt = self.data_augmentation(img,gt)
        return (torch.from_numpy(img),
                torch.from_numpy(gt))
                
class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, DATA_FOLDER, LABEL_FOLDER,WINDOW_SIZE,cache=False):
        super(ISPRS_dataset, self).__init__()
        self.WINDOW_SIZE = WINDOW_SIZE
        self.cache = cache
        dirs = os.listdir(DATA_FOLDER)
        temp_img= []
        self.data_files = []
        self.label_files = []
        # List of files
        for file in dirs:
            if file[-4:]=='.tif':
                temp_img.append(file)
        temp_name = random.sample(temp_img ,10)
        for f in temp_name:
            self.data_files.append(DATA_FOLDER+f)
            #self.label_files.append(LABEL_FOLDER+f[:-8]+'_label.tif')#postdam
            
            self.label_files.append(LABEL_FOLDER+f[:-4]+'_label.tif') #other dataset
        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
            
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    def data_augmentation(cls, *arrays, flip=True, mirror=True): #*表示接收的参数用元组表示
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip: #水平翻转
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror: #垂直翻转
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image

        random_idx = random.randint(0, len(self.data_files) - 1)
        # If the tile hasn't been loaded yet, put in cache
        
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = np.asarray(preprocessing(io.imread(self.data_files[random_idx]).transpose((2,0,1))), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(cv2.cvtColor(cv2.imread(self.label_files[random_idx]),cv2.COLOR_BGR2RGB)), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.WINDOW_SIZE)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p)) 


class Test_dataset(torch.utils.data.Dataset):
    def __init__(self, img,patch_size,stride):
        super(Test_dataset, self).__init__()
        self.img = img
        self.patch_size = patch_size
        self.stride = stride
        self.img_patchs,self.X,self.Y,self.X_temp,self.Y_temp = self.get_patch()
    def __len__(self):
        return self.img_patchs.shape[0]
    def get_patch(self):
        img_patch = []
        X = []
        Y = []
        X_temp = []
        Y_temp = []

        for x in range(0, self.img.shape[1], self.stride):
            if x + self.patch_size[0] > self.img.shape[1]:
                x = self.img.shape[1] - self.patch_size[0]
            for y in range(0, self.img.shape[2], self.stride):
                if y + self.patch_size[1] > self.img.shape[2]:
                    y = self.img.shape[2] - self.patch_size[1]
                img_patch.append(np.copy(self.img[:,x:x + self.patch_size[0], y:y + self.patch_size[1]]))
                X.append(x)
                Y.append(y)
                X_temp.append(self.patch_size[0])
                Y_temp.append(self.patch_size[1])
                
        img_patch = np.asarray(img_patch)
        X = np.asarray(X)
        Y = np.asarray(Y)
        X_temp = np.asarray(X_temp)
        Y_temp = np.asarray(Y_temp)
        return img_patch,X,Y,X_temp,Y_temp
    
    def __getitem__(self,index):
        img = self.img_patchs[index]
        Xs = self.X[index]    
        Ys = self.Y[index]
        Xt = self.X_temp[index]
        Yt = self.Y_temp[index]
        return torch.Tensor(img),Xs,Ys,Xt,Yt