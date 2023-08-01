import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import PIL.Image as pil_image

import matplotlib.pyplot as plt

# 每个类别对应的RGB值
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
        [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
        [64,128,0],[192,128,0],[64,0,128],[192,0,128],
        [64,128,128],[192,128,128],[0,64,0],[128,64,0],
        [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引

# 将图片映射成索引数组
def image_to_label(image):
    """将图片映射成类别索引的数组"""
    data=np.array(image,dtype='int32')
    # 按照上面一样的计算规则，得到对应的值
    index = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[index],dtype='int64')

class VOCDataset(Dataset):
    """自定义数据类加载规则"""
    def __init__(self,file_path=None,transform=None):
        """初始化函数"""
        
        images_labels=[]
        file=open(file_path)
        for name in file.readlines():
            
            # 移除空格和换行符
            name=name.strip()
            
            image="/home/james/data/datasets/VOC/VOCdevkit/VOC2012/JPEGImages/"+name+".jpg"
            label="/home/james/data/datasets/VOC/VOCdevkit/VOC2012/SegmentationClass/"+name+".png"
            images_labels.append((image,label))
            
        self.images_labels=images_labels
        self.transform=transform
    
    def __getitem__(self,index):
        """在DataLoader中会调用这个方法读取一个batch的数据"""
        
        image_path,label_path=self.images_labels[index]

        # 使用image.open加载目标图和特征图
        image=pil_image.open(image_path)
        label=pil_image.open(label_path).convert('RGB')
        
         # 裁剪图片，使其所有的图片输入一致   
#         x,y,width,height=transforms.RandomCrop.get_params(img=image,output_size=(224,224))
#         image=function.crop(image,x,y,width,height)
#         label=function.crop(label,x,y,width,height)

        image=transforms.Resize((512,512))(image)
        label=transforms.Resize((512,512))(label)
        print(label_path)
        #label.show()
        #print(type(image))

        
        # 转化特征图
        if self.transform is not None:
            image=self.transform(image)
            
        # 映射目标图
        label=image_to_label(label)
        #print(type(label))
        #print(sum(label))
        #print(max(label.flatten()))
        #plt.imshow(label)
        # 从numpy数组转化成张量
        label=torch.from_numpy(label)
        #print(label.numpy())        
        #print(type(label))

        
        # 返回
        return image,label
    
    def __len__(self):
        """获取整个dataset的数据大小"""
        return len(self.images_labels)

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

transform_test=transforms.Compose([
    # 将数据转化成张量，并且归一化到[0,1]
    transforms.ToTensor(),
    # 将数据标准化到[-1,1]，image=(image-mean)/std
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

#test_datasets=VOCDataset(file_path="/home/james/data/datasets/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",transform=transform_test)
test_datasets=VOCDataset(file_path="/home/james/data/datasets/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",transform=transform_test)
test_loader=DataLoader(dataset=test_datasets,batch_size=8,shuffle=False,sampler=None)

# 选择设置，优先GPU
device=torch.device("cuda" if torch.cuda.is_available else "cpu")

# 加载模型
model=FCN8s()
#model.load_state_dict(torch.load("./model/fcn8s_180.pt"))
#model.load_state_dict(torch.load("./fcn8s_from_pyfcn2.pt"))
test=torch.load("./fcn8s_from_pyfcn.pth.tar")
model.load_state_dict(test['model_state_dict'])
#print(model)
model=model.to(device)

#model=torch.load("./fcn8s.pt")
# 进入评估模式
model.eval()

cm = np.array(colormap).astype('uint8')

# 测试集梯度不更新
with torch.no_grad():
    for image,label in test_loader:
        #plt.imshow(label[0].data.numpy())
        #print(max(label.numpy()))
        test=label
        #test2=np.zeros((512,512,3))
        test2=image.permute(0,2,3,1)
        # 将数据复制到GPU中
        image=image.to(device)
        label=label.to(device)
        
        #print("image",image.shape,"label",label.shape)

        # 正向传播
        output=model(image)
        
        # 把数据从GPU复制到CPU中，plt才能调用
        output=output.max(1)[1].squeeze().cpu().data.numpy()
        '''
        #pred = cm[output]
        pred = output
        plt.subplot(1, 3, 1)
        plt.imshow(pred[0])
        plt.subplot(1, 3, 2)
        plt.imshow(test[0])
        plt.subplot(1, 3, 3)
        plt.imshow(test2[0])
        plt.show()
#        output=output.cpu().numpy()
#        print(output.shape)
        break
        '''

        for i,eval_image in enumerate(output):
            plt.subplot(1, 3, 1)
            plt.imshow(eval_image)
            plt.subplot(1, 3, 2)
            plt.imshow(test[i])
            plt.subplot(1, 3, 3)
            plt.imshow(test2[i])
            #plt.show()
            plt.pause(2)
        #break

