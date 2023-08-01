import sys 
sys.path.append(sys.path[0]+"/..") 
#sys.path.append("/home/james/Desktop/james/my_test/thesis/deepmask-pytorch") 
#print(sys.path)

import argparse
import models
import numpy as np
import pandas as pd
import time
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
#from tools.InferDeepMask import Infer
from InferDeepMask import Infer
from utils.load_helper import load_pretrain
from loader import get_loader, dataset_names
from tools.train import BinaryMeter, IouMeter

#path="/home/james/Desktop/james/my_test/thesis/deepmask-pytorch/"

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch DeepMask/SharpMask evaluation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepMask', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: DeepMask)')
parser.add_argument('--resume', default='exps/deepmask/train/model_best.pth.tar',
                    type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--dataset', default='coco', choices=dataset_names(),
                            help='data set')
#parser.add_argument('--img', default='data/testImage.jpg',
#                    help='path/to/test/image')
parser.add_argument('--nps', default=2, type=int,
                    help='number of proposals to save in test')
parser.add_argument('--si', default=-2.5, type=float, help='initial scale')
parser.add_argument('--sf', default=.5, type=float, help='final scale')
parser.add_argument('--ss', default=.5, type=float, help='scale step')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                            help='number of data loading workers (default: 12)')
parser.add_argument('-b', '--batch', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--maxload', default=4000, type=int, metavar='N',
                    help='max number of training batches per epoch')
parser.add_argument('--testmaxload', default=500, type=int, metavar='N',
                    help='max number of testing batches')
parser.add_argument('--iSz', default=160, type=int, metavar='N',
                    help='input size')
parser.add_argument('--oSz', default=56, type=int, metavar='N',
                    help='output size')
parser.add_argument('--gSz', default=112, type=int, metavar='N',
                    help='ground truth size')
parser.add_argument('--shift', default=16, type=int, metavar='N',
                    help='shift jitter allowed')
parser.add_argument('--scale', default=.25, type=float,
                    help='scale jitter allowed')
parser.add_argument('--hfreq', default=.5, type=float,
                    help='mask/score head sampling frequency')

def range_end(start, stop, step=1):
    return np.arange(start, stop+step, step);


def main():
    global args
    args = parser.parse_args()
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Model
    from collections import namedtuple
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
    config = Config(iSz=160, oSz=56, gSz=112, batch=1)  # default for training
    df = pd.DataFrame()
    num=0


    model = (models.__dict__[args.arch](config))
    #model_path=path+args.resume
    #print("@@@@@@@@@@",model_path)
    model = load_pretrain(model, args.resume)
    #model = load_pretrain(model, model_path)
    model = model.eval().to(device)

    scales = [2**i for i in range_end(args.si, args.sf, args.ss)]
    meanstd = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    infer = Infer(nps=args.nps, scales=scales, meanstd=meanstd, model=model, device=device)

    # Setup data loader
    train_dataset = get_loader(args.dataset)(args, split='train')
    val_dataset = get_loader(args.dataset)(args, split='val')
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch, num_workers=args.workers,
        pin_memory=True, sampler=None)
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch, num_workers=args.workers,
        pin_memory=True, sampler=None)

    mask_meter = IouMeter(0.5, len(val_loader.dataset))
    mask_meter.reset()
    score_meter = BinaryMeter()
    score_meter.reset()

    # Setup Metrics
    criterion = nn.SoftMarginLoss().to(device)

    with torch.no_grad():
        for i, (img, target, head_status) in enumerate(train_loader):
        #for i, (img, target, head_status) in enumerate(val_loader):
            img = img.to(device)
            target = target.to(device)

            # compute output
            tic = time.time()
            output = model(img)
            toc = time.time() - tic
            print('          %05.3f s' % toc)
            loss = criterion(output[head_status[0]], target)

            # measure accuracy and record loss
            if head_status[0] == 0:
                mask_meter.add(output[head_status[0]], target)
            else:
                score_meter.add(output[head_status[0]], target)
            
            #print(output[head_status[0]])
            print(type(output[head_status[0]]))
            print(type(target))
            #cv2.imshow('output', output[head_status[0]].sigmoid().cpu().data.numpy())
            #cv2.imshow('target', target.sigmoid().cpu().data.numpy())
            #cv2.waitKey(0)

            df = df.append({'time':toc, 'Iou':mask_meter.value('mean'), 'median':mask_meter.value('median'), 'suc@.5':mask_meter.value('0.5'), 'suc@.7':mask_meter.value('0.7'), 'acc':score_meter.value()}, ignore_index = True)
            print('%d IoU: mean %05.2f median %05.2f suc@.5 %05.2f suc@.7 %05.2f | acc %05.2f' % (
                i, mask_meter.value('mean'), mask_meter.value('median'), mask_meter.value('0.5'), mask_meter.value('0.7'),
                score_meter.value()))
            #print(df)
            #df.to_csv("test.csv", index=False)
            print("###############################")
            num=num+1
            if num==1:
                break
'''
            print('| start'); tic = time.time()
            im = img[0].cpu().permute([1,2,0]).numpy()
            print("im=", np.shape(im),"image=", np.shape(img),"label=",np.shape(target))
            #im = np.array(Image.open(args.img).convert('RGB'), dtype=np.float32)
            #im = np.array(Image.open(args.img).convert('RGB').resize((640,480)), dtype=np.float32)
            cv2.imshow('test', im)
            cv2.waitKey(0)

            h, w = im.shape[:2]
            img = np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0).astype(np.float32)
            img = torch.from_numpy(img / 255.).to(device)
            infer.forward(img)
            masks, scores = infer.getTopProps(.2, h, w)
            #print("@@@@@@@@@@@@@@",np.shape(masks),np.shape(scores))
            toc = time.time() - tic
            print('| done in %05.3f s' % toc)

            for i in range(masks.shape[2]):
                res = im[:,:,::-1].copy().astype(np.uint8)
                res[:, :, 2] = masks[:, :, i] * 255 + (1 - masks[:, :, i]) * res[:, :, 2]

                mask = masks[:, :, i].astype(np.uint8)
                #_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt_area = [cv2.contourArea(cnt) for cnt in contours]
                cnt_max_id = np.argmax(cnt_area)
                contour = contours[cnt_max_id]
                polygons = contour.reshape(-1, 2)

                predict_box = cv2.boundingRect(polygons)
                predict_rbox = cv2.minAreaRect(polygons)
                rbox = cv2.boxPoints(predict_rbox)
                print('Segment Proposal Score: {:.3f}'.format(scores[i]))

                res = cv2.rectangle(res, (predict_box[0], predict_box[1]),
                              (predict_box[0]+predict_box[2], predict_box[1]+predict_box[3]), (0, 255, 0), 3)
                res = cv2.polylines(res, [np.int0(rbox)], True, (0, 255, 255), 3)
                #print("@@@@@@@@SIZE=",np.shape(mask))
                #print("",mask)
                #cv2.imshow('Mask', mask)
                cv2.imshow('Proposal', res)
                cv2.waitKey(0)
            cv2.destroyAllWindows()
'''
if __name__ == '__main__':
    main()
