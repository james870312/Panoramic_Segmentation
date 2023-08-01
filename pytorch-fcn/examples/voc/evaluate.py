#!/usr/bin/env python

import datetime
import pytz
import argparse
import os
import os.path as osp

import fcn
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
import torchfcn
import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    time = []
    num=0
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser('~/data/datasets')
    val_loader = torch.utils.data.DataLoader(
        #torchfcn.datasets.VOC2011ClassSeg(root, split='seg11valid', transform=True),
        torchfcn.datasets.City2011ClassSeg(root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    n_class = len(val_loader.dataset.class_names)

    if osp.basename(model_file).startswith('fcn32s'):
        model = torchfcn.models.FCN32s(n_class=21)
    elif osp.basename(model_file).startswith('fcn16s'):
        model = torchfcn.models.FCN16s(n_class=21)
    elif osp.basename(model_file).startswith('fcn8s'):
        if osp.basename(model_file).startswith('fcn8s-atonce'):
            model = torchfcn.models.FCN8sAtOnce(n_class=21)
        else:
            model = torchfcn.models.FCN8s(n_class=21)
    else:
        raise ValueError

    if torch.cuda.is_available():
        model = model.cuda()
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
    #print(len(val_loader))
    
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data, volatile=True), Variable(target)
    
        time_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        score = model(data)
        elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - time_start).total_seconds()
        time.append(elapsed_time)
        #time_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        #print(np.shape(data))

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        print(np.shape(imgs),np.shape(lbl_true),np.shape(lbl_pred))

        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)

            label_trues.append(lt)
            label_preds.append(lp)
            #viz = fcn.utils.visualize_segmentation(
            #    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
            #    label_names=val_loader.dataset.class_names)

            #print(np.shape(img))
            #print(np.shape(lp))
            #print(np.shape(viz))
            #skimage.io.imsave('test.jpg', img)
            #skimage.io.imsave('test.png', lp)
            #skimage.io.imsave('test.jpg', viz)
            #visualizations.append(viz)
            #visualizations = fcn.utils.get_tile_image(viz)
            #test = fcn.utils.label2rgb(lp, label_names=val_loader.dataset.class_names, n_labels=n_class)
            #print("num = ", num)
            result = fcn.utils.label2rgb(lp, n_labels=n_class)
            #skimage.io.imshow(lp)
            #skimage.io.imshow(viz)
            #skimage.io.imshow(result)
            #plt.show()
            '''
            test = score.data[0].cpu().numpy()[:, :, :]
            for i in range(21):
                #print(test.shape)
                skimage.io.imshow(test[i])
                plt.show()
            '''
            #skimage.io.imsave('examples/voc/output_city2/viz_evaluate'+str(num)+'.png', viz)
            skimage.io.imsave('examples/voc/output_city2/result'+str(num)+'.png', result)

            num+=1
 
            '''
            if len(visualizations) < 18:
                viz = fcn.utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=val_loader.dataset.class_names)
                visualizations.append(viz)
            '''
            metrics = torchfcn.utils.label_accuracy_score(
                lt, lp, n_class=n_class)
            metrics = np.array(metrics)
            metrics *= 100
            print('''\
            Accuracy: {0}
            Accuracy Class: {1}
            Mean IU: {2}
            FWAV Accuracy: {3}'''.format(*metrics))
            print("            time = ", time[-1],'s')
            print("##############################################")
           
        #if num==5:
        #    break
    
    #print(type(label_trues),type(label_preds),type(n_class))
    metrics = torchfcn.utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    Accuracy: {0}
    Accuracy Class: {1}
    Mean IU: {2}
    FWAV Accuracy: {3}'''.format(*metrics))
    time = np.array(time)
    print("time = ", time)
    print("time_size = ",time.shape)
    print("average time = ",time.mean())

    print("Total have %d images" %(num))
    #viz = fcn.utils.get_tile_image(visualizations)
    #skimage.io.imsave('viz_evaluate.png', viz)
    #print(np.shape(viz))
 

if __name__ == '__main__':
    main()
