import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools

import time

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from utils.misc import *
from model.UNet import *
from model.network import *
from model.MNet import *
import argparse
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='CNN')
# use last save model
parser.add_argument('--device', type=str, default='1', help='GPU device (default: 0)')
parser.add_argument('--stride_size', type=int, default=512, help='stride size (default: 5)')
parser.add_argument('--check_path', type=str, default='./logs/GID-5/U_Net/1/checkpoints/last.pt',
                    help='load model path')
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
parser.add_argument('--batch_size', type=int, default=10, help='input batch size for training (default: 128)')
args = vars(parser.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
torch.cuda.is_available()
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
checkpoint = torch.load(args['check_path'])

print(checkpoint['args'])
print(checkpoint['logs'] )


WINDOW_SIZE = (checkpoint['args']['patch_size'], checkpoint['args']['patch_size']) # Patch size
IN_CHANNELS = 1 # Number of input channels (e.g. RGB)
FOLDER = checkpoint['args']['data_path'] # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = args['batch_size'] # Number of samples in a mini-batch



DATASET = checkpoint['args']['dataset']   
DATA_FOLDER=''
LABEL_FOLDER=''
if DATASET == 'GID-15':
    LABELS = ['other','industrial land', 'urban residential', 'rural residential', 'traffic land', 'paddy field',
                      'irrigated land', 'dry cropland', 'garden plot', 'arbor woodland', 'shrub land',
                      'natural grassland', 'artificial grassland', 'river', 'lake', 'pond'] # Label names
    MAIN_FOLDER = FOLDER + 'GID-15/'
    DATA_FOLDER = MAIN_FOLDER + 'test_image_RGB/'
    LABEL_FOLDER = MAIN_FOLDER + 'test_label_15classes/'
    #ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
elif DATASET == 'GID-5':
    LABELS =['other','built-up', 'farmland', 'forest', 'meadow', 'water']
    MAIN_FOLDER = FOLDER + 'GID-5/'
    DATA_FOLDER = MAIN_FOLDER + 'test_image_RGB/'
    LABEL_FOLDER = MAIN_FOLDER + 'test_label_5classes/'
elif DATASET == 'postdam':
    LABELS = ['background','Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car']
    MAIN_FOLDER = FOLDER + 'postdam/'
    DATA_FOLDER = MAIN_FOLDER + 'test_1_DSM/'
    LABEL_FOLDER = MAIN_FOLDER + 'test_5_label_all/'  
    
N_CLASSES = len(LABELS) # Number of classes

   
if checkpoint['args']['model'] == 'U_Net':
    net = SegNet(in_channels=IN_CHANNELS, out_channels=N_CLASSES)
elif checkpoint['args']['model'] == 'AU_Net':
    net = AUnet(in_channel=IN_CHANNELS,num_classes=N_CLASSES)
elif checkpoint['args']['model'] == 'ATU_Net':
    net = ATUnet(in_channel=IN_CHANNELS,num_classes=N_CLASSES)
elif checkpoint['args']['model'] == 'ATUL_Net':
    net = ATULnet(in_channel=IN_CHANNELS,num_classes=N_CLASSES)    
net.cuda()

mkdir_p(args['check_path'][:-3])
save_path = args['check_path'][:-3]+'/'
def test(net, stride=args['stride_size'], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    dirs = os.listdir(DATA_FOLDER)
    all_acc = 0
    all_f1 = 0
    all_kappa = 0
    all_miou = 0
    n=1
    net.eval()
    for file in dirs:
        if file[-4:]=='.tif':
            print(n)
            image_path = DATA_FOLDER+file
            label_path = LABEL_FOLDER+file[:-4]+'_label.tif'
            #label_path= LABEL_FOLDER+file[:-8]+'_label.tif' #postdam
            img = (np.asarray(preprocessing(io.imread(image_path).transpose(2, 0, 1)), dtype='float32'))
            '''gt = (np.asarray(cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB), dtype='uint8') for
                   id in test_ids)'''
            gt_e = (convert_from_color(cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)))
            # Switch the network to inference mode
            test_set = Test_dataset(img=img,patch_size=WINDOW_SIZE,stride=stride)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
            pred = np.zeros(img.shape[1:] + (N_CLASSES,))
            y=[]
            
            progress_bar=tqdm(test_loader)
            with torch.no_grad():
                for baech_idx, (patch, X, Y, W, H) in enumerate(progress_bar):
                    patch = patch.detach().cuda()
                    outs = net(patch.contiguous())
                    #outs = net(patch.contiguous(),y)
                    outs = torch.nn.functional.softmax(outs,dim =1)
                    outs = outs.data.cpu().numpy()
                    for i in  range(outs.shape[0]):
                        out = outs[i].transpose((1, 2, 0))
                        pred[X[i]:X[i] + W[i], Y[i]:Y[i] + H[i], :] = pred[X[i]:X[i] + W[i], Y[i]:Y[i] + H[i], :]+out
                    del (outs,out,patch)
                pred = np.argmax(pred, axis=-1)
            accuracy, kappa ,miou,MF1=metrics(pred.ravel(), gt_e.ravel())
            img = convert_to_color(pred)
            io.imsave(save_path+file[:-4]+'.png', img)
            file_perf = open(save_path + 'performances.txt', 'a+')
            file_perf.write(file[:-4]
                + "    " + str(accuracy)
                + "    " + str(kappa)
                + "    " + str(miou)
                + "    " + str(MF1)
                +"\n")
            file_perf.close()
            all_acc=all_acc+accuracy
            all_kappa = all_kappa+kappa
            all_miou = all_miou+miou
            all_f1 = all_f1+MF1
            n=n+1
    file_perf = open(save_path + 'performances.txt','a+')
    file_perf.write("Total  " 
        + "    " + str(all_acc/(n-1))
        + "    " + str(all_kappa/(n-1))
        + "    " + str(all_miou/(n-1))
        + "    " + str(all_f1/(n-1))
        +"\n")
    file_perf.close()        
    
    
#net.load_state_dict(torch.load('./segnet_final'))
net.load_state_dict(checkpoint['net'])
test(net, stride=64,batch_size=10,window_size=(256,256))