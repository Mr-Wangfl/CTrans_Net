# imports and stuff
import numpy as np
from skimage import io
from glob import glob
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
# Torch imports
import os
import cv2
from tensorboardX import SummaryWriter
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

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--device', type=str, default='1', help='GPU device (default: 0)')
parser.add_argument('--dataset', default='GID-5', choices=['GID-5','GID-15','postdam'])
parser.add_argument('--data_path', type=str, default='datasets/', help='data path')
parser.add_argument('--model', default='U_Net', choices=['AU_Net','M_Net','U_Net', 'ATU_Net','ATUL_Net'])
parser.add_argument('--batch_size', type=int, default=10, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

parser.add_argument('--patch_size', type=int, default=256, help='patch size (default: 256)')

parser.add_argument('--load_last', action='store_true', default=False, help='load last model')
parser.add_argument('--load_path', type=str, default='logs/', help='load model path')
parser.add_argument('--logs_path', type=str, default='logs/', help='load model path')

args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.cuda.is_available()
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

if str(args['logs_path']).endswith('/') is False:
    args['logs_path'] += '/'
    
if args['load_path'] is not None and str(args['load_path']).endswith('/') is False:
    args['load_path'] += '/'
    
if args['load_last'] is False:
    mkdir_p(args['logs_path'] + args['dataset'] + '/' + args['model'] + '/')
    index = np.sort(np.array(os.listdir(args['logs_path'] + args['dataset'] + '/' + args['model'] + '/'), dtype=int))
    index = index.max() + 1 if len(index) > 0 else 1
    basic_path = args['logs_path'] + args['dataset'] + '/' + args['model'] + '/' + str(index) + '/'
    mkdir_p(basic_path)
    args['load_path'] = basic_path
    cur_epoch = 0
else:
    basic_path = args['load_path']
    assert os.path.exists(basic_path), '目录不存在'
    assert os.path.isfile(basic_path + 'checkpoints/last.pt'), 'Error: no checkpoint file found!'
    checkpoint = torch.load(basic_path + 'checkpoints/last.pt')
    checkpoint['args']['load_last'] = args['load_last']
    checkpoint['args']['load_path'] = args['load_path']
    args = checkpoint['args']
    cur_epoch = checkpoint['epoch'] + 1
    logs = checkpoint['logs']
    print('保存模型的最后一次训练结果： %s, 当前训练周期: %4d, ' % (str(logs[-1]), cur_epoch))
    #assert cur_epoch < args['epochs'], '已经跑完了，cur_epoch: {}，epochs: {}'.format(cur_epoch, args['epochs'])

print('当前日志目录： ' + basic_path)
mkdir_p(basic_path + 'checkpoints/periods/')
mkdir_p(basic_path + 'tensorboard/')
print(args)
with open(basic_path + 'args.txt', 'w+') as f:
    for arg in args:
        f.write(str(arg) + ': ' + str(args[arg]) + '\n')
        
WINDOW_SIZE = (args['patch_size'], args['patch_size']) # Patch size
IN_CHANNELS = 1 # Number of input channels (e.g. RGB)
FOLDER = args['data_path'] # Replace with your 
BATCH_SIZE = args['batch_size'] # Number of samples in a mini-batch

DATASET = args['dataset']   
DATA_FOLDER=''
LABEL_FOLDER=''
ts_writer = SummaryWriter(log_dir=basic_path + 'tensorboard/', comment=args['model'])
args_str = ''
for arg in args:
    args_str += str(arg) + ': ' + str(args[arg]) + '<br />'
ts_writer.add_text('args', args_str, cur_epoch)

if DATASET == 'GID-15':
    LABELS = ['other','industrial land', 'urban residential', 'rural residential', 'traffic land', 'paddy field','irrigated land', 'dry cropland', 'garden plot', 'arbor woodland', 'shrub land','natural grassland', 'artificial grassland', 'river', 'lake', 'pond'] # Label names
    MAIN_FOLDER = FOLDER + 'GID-15/'
    DATA_FOLDER = MAIN_FOLDER + 'train_image_RGB/'
    LABEL_FOLDER = MAIN_FOLDER + 'train_label_15classes/'
    #test path
    image_path = './datasets/GID-15/test_image_RGB/GF2_8.tif'
    label_path = './datasets/GID-15/test_label_15classes/GF2_8_label.tif'
elif DATASET == 'GID-5':
    LABELS =['other','built-up', 'farmland', 'forest', 'meadow', 'water']
    MAIN_FOLDER = FOLDER + 'GID-5/'
    DATA_FOLDER = MAIN_FOLDER + 'train_image_RGB/'
    LABEL_FOLDER = MAIN_FOLDER + 'train_label_5classes/'
    #test path
    image_path = './datasets/GID-5/test_image_RGB/GF2_PMS1__L1A0001015649-MSS1.tif'
    label_path = './datasets/GID-5/test_label_5classes/GF2_PMS1__L1A0001015649-MSS1_label.tif'
elif DATASET == 'postdam':
    LABELS = ['background','Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car']
    MAIN_FOLDER = FOLDER + 'postdam/'
    DATA_FOLDER = MAIN_FOLDER + 'train_1_DSM/'
    LABEL_FOLDER = MAIN_FOLDER + 'train_5_Labels_all/'
    #test path
    image_path = './datasets/postdam/test_1_DSM/top_potsdam_2_14_RGB.tif'
    label_path = './datasets/postdam/test_5_label_all/top_potsdam_2_14_label.tif'
    
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory
test_images = (np.asarray(preprocessing(io.imread(image_path).transpose(2, 0, 1)), dtype='float32'))
test_set = Test_dataset(img=test_images,patch_size=WINDOW_SIZE,stride=WINDOW_SIZE[0])

if args['model'] == 'U_Net':
    net = SegNet(in_channels=IN_CHANNELS, out_channels=N_CLASSES)
elif args['model'] == 'AU_Net':
    net = AUnet(in_channel=IN_CHANNELS,num_classes=N_CLASSES)
elif args['model'] == 'ATU_Net':
    net = ATUnet(in_channel=IN_CHANNELS,num_classes=N_CLASSES)
elif args['model'] == 'ATUL_Net':
    net = ATULnet(in_channel=IN_CHANNELS,num_classes=N_CLASSES)  
elif args['model'] == 'M_Net':
    net = MNet(inplanes=IN_CHANNELS,num_classes=N_CLASSES)    

net.cuda()

optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=0.0005)
#We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 50, 75], gamma=0.1)
if args['load_last'] is True and cur_epoch > 0:
    net.load_state_dict(checkpoint['net'], strict=False)
    print('load path: ' + basic_path + 'checkpoints/last.pt')
    
def test(net,stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE):
    # Use the network on the test set
    eroded_labels = (convert_from_color(cv2.cvtColor(cv2.imread(label_path),cv2.COLOR_BGR2RGB)))
    pred = np.zeros(test_images.shape[1:] + (N_CLASSES,),dtype=float)
    print(pred.shape)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False, num_workers=0)
    y = []
    
    # Switch the network to inference mode
    net.eval()
    with torch.no_grad():
        for baech_idx, (img, X, Y, W, H) in enumerate(test_loader):
            img = img.detach().cuda()
            #outs = net(img.contiguous())
            outs = net(img.contiguous(),y)
            outs = torch.nn.functional.softmax(outs,dim =1)
            outs = outs.data.cpu().numpy()
            for i in  range(outs.shape[0]):
                out = outs[i].transpose((1, 2, 0))
                pred[X[i]:X[i] + W[i], Y[i]:Y[i] + H[i], :] = pred[X[i]:X[i] + W[i], Y[i]:Y[i] + H[i], :]+out
        pred = np.argmax(pred, axis=-1)
        img = convert_to_color(pred) 
    io.imsave('./Evaluation.png', img)
    Acc, kappa ,miou,MF1 =metrics(pred.ravel(), np.asarray(eroded_labels).ravel())
    
    return Acc, kappa ,miou,MF1



def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch = 1):
    weights = weights.cuda()
    criterion = nn.L1Loss().cuda()
    for e in range(cur_epoch, epochs + 1):
         
        train_loss = 0
        #train_set = Train_dataset(DATA_FOLDER,LABEL_FOLDER,WINDOW_SIZE=WINDOW_SIZE,patch_num=100)
        train_set = ISPRS_dataset(DATA_FOLDER,LABEL_FOLDER,WINDOW_SIZE=WINDOW_SIZE, cache=CACHE)
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            #output = net(data)
            #loss = CrossEntropy2d(output, target)
            output,L_ouput = net(data,target)
            loss1 = CrossEntropy2d(output, target)
            loss2 = criterion(L_ouput,torch.zeros([BATCH_SIZE],dtype=torch.float).cuda())
            loss = loss1+loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            mean_loss = train_loss/(batch_idx +1)
            #losses[iter_] = loss.item()
            #mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
            temp = len(train_loader)//10
            if (batch_idx+1) % temp == 0:
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                acc=accuracy(pred, gt)
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx+1, len(train_loader),
                    100. * (batch_idx+1) / len(train_loader), mean_loss, acc))
                io.imsave('./rgb.png', rgb)
                io.imsave('./gt.png', convert_to_color(gt))
                io.imsave('./pre.png', convert_to_color(pred))
            del(data, target, loss)
        if scheduler is not None:
            scheduler.step()   
        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            if e % 5==0:
                print("Evaluation")
                Acc, kappa ,miou,MF1= test(net,stride=min(WINDOW_SIZE))
                print(str(Acc)
                + "    " + str(kappa)
                + "    " + str(miou)
                + "    " + str(MF1))
            logs=[e,mean_loss,acc]
            state = {
            'net': net.state_dict(),
            'epoch': e,
            'logs': logs,
            'args': args
            }
            torch.save(state, basic_path + 'checkpoints/periods/{}.pt'.format(e))
            torch.save(state, basic_path + 'checkpoints/last.pt')
            del(state)
    

train(net, optimizer, 300, scheduler)
ts_writer.close()
