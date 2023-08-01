#!/usr/bin/env python
import os, sys
sys.path.insert(1,"/home/james/Desktop/james/my_test/thesis/deepmask-pytorch") 
#print(sys.path)

import time
import argparse
import os.path as osp
import numpy as np
import pandas as pd
import tqdm
import skimage.io
from PIL import Image

import cv2
import fcn
import torch
from torch.autograd import Variable

import torchfcn
import models
from utils.load_helper import load_pretrain
from tools.InferDeepMask import Infer
import mydata

model_names = sorted(name for name in models.__dict__ 
                        if not name.startswith("__") and callable(models.__dict__[name]))
#print(model_names)

#FCN parameter
parser = argparse.ArgumentParser(description='PyTorch wkentaro/pytorch-fcn & DeepMask/SharpMask evaluation')

parser.add_argument('-model_file', default='model/coco/fcn8s.pth.tar',help='coco Model path (FCN)')
#parser.add_argument('-model_file', default='model/city/fcn8s.pth.tar',help='city Model path (FCN)')

parser.add_argument('-g', '--gpu', type=int, default=0,help='num of gpu')

#DeepMask parameter
parser.add_argument('-arch', '-a', metavar='ARCH', default='DeepMask', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: DeepMask)')
parser.add_argument('-resume', default='model/coco/DeepMask.pth.tar',help='coco Model path (deepmask)')
#parser.add_argument('-resume', default='model/city/DeepMask.pth.tar',help='city Model path (deepmask)')

parser.add_argument('-nps', default=10, type=int,
                    help='number of proposals to save in test')
parser.add_argument('-si', default=-2.5, type=float, help='initial scale')
parser.add_argument('-sf', default=.5, type=float, help='final scale')
parser.add_argument('-ss', default=.5, type=float, help='scale step')

def range_end(start, stop, step=1):
    return np.arange(start, stop+step, step)

def main():
    print("########main#######")
    num=0
    fcn_time = []
    dm_time = []
    total_time = []
    fcn_acc = []
    fcn_cacc = []
    fcn_mIU = []
    fcn_FAMV_acc = []
    args = parser.parse_args()
    #print(args.model_file,args.gpu,args.arch,args.resume,args.img,args.nps,args.si,args.sf,args.ss)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    #setup dataset
    root = osp.expanduser('~/data/datasets')
    
    val_loader = torch.utils.data.DataLoader(
        #torchfcn.datasets.VOC2011ClassSeg(
        mydata.VOC2011ClassSeg(
        #mydata.City2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    # Setup FCN Model
    n_class = len(val_loader.dataset.class_names)
    model = torchfcn.models.FCN8s(n_class=21)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.cuda()
    else:
        device = torch.device('cpu')

    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    
    # Setup Deepmask Model
    from collections import namedtuple
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
    config = Config(iSz=160, oSz=56, gSz=112, batch=1)  # default for training
    model2 = (models.__dict__[args.arch](config))
    model2 = load_pretrain(model2, args.resume)
    model2 = model2.eval().to(device)

    scales = [2**i for i in range_end(args.si, args.sf, args.ss)]
    meanstd = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    infer = Infer(nps=args.nps, scales=scales, meanstd=meanstd, model=model2, device=device)
    print('==> Evaluating with FCN&Deepmask')
    
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        tic = time.time()
        score = model(data)
        toc = time.time() - tic
        print('\n==> FCN done in %05.3f s' % toc)
        total_time.append(toc)
        fcn_time.append(toc)

        imgs = data.data.cpu()
        #print(np.shape(imgs))
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        #skimage.io.imsave('test.png', lbl_pred[0])
        #print(np.shape(imgs),np.shape(lbl_true),np.shape(lbl_pred))

        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            #FCN
            img, lt = val_loader.dataset.untransform(img, lt)
            org_img = img[:,:,::-1].copy().astype(np.uint8)
            label_trues.append(lt)
            label_preds.append(lp)
            #cv2.imshow('input', img)
            #cv2.waitKey(0)

            #viz = fcn.utils.visualize_segmentation(
            #    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
            #    label_names=val_loader.dataset.class_names)
            #visualizations.append(viz)
            #visualizations = fcn.utils.get_tile_image(viz)
            result = fcn.utils.label2rgb(lp, n_labels=n_class)
            #skimage.io.imsave('output/viz_evaluate'+str(num)+'.png', viz)
            #skimage.io.imsave('output/result'+str(num)+'.jpg', result)
            num+=1
            #np.set_printoptions(threshold=np.inf)
            #print(np.shape(result), np.shape(viz), np.shape(lbl_pred))
            #cv2.imshow("FCN_input", img)
            #cv2.imshow("FCN_output", result)
            #cv2.imshow("FCN_Compare", viz)
            #lbl_pred[lbl_pred>0]=255
            #cv2.imshow("FCN_mask", lbl_pred[0].astype(np.uint8))
            #cv2.waitKey()

            for i in range(len(img[:,0,0])):
                for j in range(len(img[0,:,0])):
                    if lbl_pred[0,i,j]==0:
                        img[i,j,:]=0

            #cv2.imshow("test", img)
            #cv2.waitKey()
           
            #DeepMask
            im = img 
            h, w = im.shape[:2]
            #print(np.shape(im),h,w)
            img2 = np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0).astype(np.float32)
            img2 = torch.from_numpy(img2 / 255.).to(device)
            tic = time.time()
            infer.forward(img2)
            masks, scores = infer.getTopProps(.2, h, w)
            #print(np.shape(masks),np.shape(scores))
            toc = time.time() - tic
            print('==>DeepMask done in %05.3f s' % toc)
            total_time[num-1] += toc
            dm_time.append(toc)
            img_pseg = np.zeros((result.shape), dtype = 'uint8')

            for i in range(masks.shape[2]):
                channel = np.random.randint(2, size=1)
                color = np.random.randint(256, size=1)
                #print(channel, color)
                res = im[:,:,::-1].copy().astype(np.uint8)
                res[:, :, 2] = masks[:, :, i] * 255 + (1 - masks[:, :, i]) * res[:, :, 2]
                
                for j in range(len(img_pseg[:,0,0])):
                    for k in range(len(img_pseg[0,:,0])):
                        if (lbl_pred[0,j,k]!=0 and masks[j, k, i]!=0):
                            img_pseg[j, k, :] = 0
                            img_pseg[j, k, channel] = masks[j, k, i] * color #+ (1 - masks[j, k, i]) * img_pseg[j, k, 2]
                
                mask = masks[:, :, i].astype(np.uint8)
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
                #print("contours = ",contours)
                #print("SIZE = ",np.shape(mask))
                mask[mask>0]=255
                #cv2.imshow('Mask', mask)
                #cv2.imshow("FCN_Compare", viz)
                '''
                cv2.imshow("input", org_img)
                cv2.imshow('Proposal', res)
                cv2.imshow("FCN_output", result)
                cv2.imshow("panoptic_seg", img_pseg)
                cv2.moveWindow("input", 0, 0)
                cv2.moveWindow("Proposal", 600, 0)
                cv2.moveWindow("FCN_output", 600, 600)
                cv2.moveWindow("panoptic_seg", 0, 600)
                kbin = cv2.waitKey(0)
                '''
                kbin = 0

                if kbin==27:
                    break
                if kbin == 99:
                    cv2.imwrite('img/output_'+str(num)+'.png', org_img)
                    cv2.imwrite('img/output1_'+str(num)+'.png', res)
                    cv2.imwrite('img/output2_'+str(num)+'.png', result)
                    cv2.imwrite('img/output3_'+str(num)+'.png', img_pseg)

        metrics = torchfcn.utils.label_accuracy_score(
            lt, lp, n_class=n_class)
        metrics = np.array(metrics)
        metrics *= 100
        fcn_acc.append(metrics[0])
        fcn_cacc.append(metrics[1])
        fcn_mIU.append(metrics[2])
        fcn_FAMV_acc.append(metrics[3])
        #print('''\
        #Accuracy: {0}
        #Accuracy Class: {1}
        #Mean IU: {2}
        #FWAV Accuracy: {3}'''.format(*metrics))
        #print("fcn_time = %lf, Deepmask_time = %lf, total_time = %lf" %(fcn_time[num-1], dm_time[num-1], total_time[num-1]))
        #print("##############################################")
            
        #if num==2:
        #    break
        cv2.destroyAllWindows()
        if kbin==27:
            break
        #print("fcn_time = %lf, Deepmask_time = %lf, total_time = %lf" %(fcn_time[num-1], dm_time[num-1], total_time[num-1]))

    metrics = torchfcn.utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    FCN:
    Accuracy: {0}
    Accuracy Class: {1}
    Mean IU: {2}
    FWAV Accuracy: {3}'''.format(*metrics))
    total_time = np.array(total_time)
    fcn_time = np.array(fcn_time)
    dm_time = np.array(dm_time)
    print("time = ", total_time)
    print("time_size = ",total_time.shape)
    print("average time = ",total_time.mean())
    print("Total have %d images" %(num))

    df = pd.DataFrame()
    #df = pd.read_csv("voc_coco1.csv")
    df.at[:, "fcn_time"] = fcn_time[:]
    df.at[:, "Deepmask_time"] = dm_time[:]
    df.at[:, "total_time"] = total_time[:]
    df.at[:, "fcn_acc"] = fcn_acc[:]
    df.at[:, "fcn_accuracy_class"] = fcn_cacc[:]
    df.at[:, "fcn_mean_IU"] = fcn_mIU[:]
    df.at[:, "fcn_FAMV_accuracy"] = fcn_FAMV_acc[:]
    #df.to_csv("test.csv", index=False)
    #df.to_csv("voc_coco3.csv", index=False)
    #df.to_csv("city.csv", index=False)

if __name__ == '__main__':
        main()
