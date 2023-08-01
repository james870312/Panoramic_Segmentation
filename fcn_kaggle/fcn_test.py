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

class FCN8s(nn.Module):
    def __init__(self):
        super(FCN8s,self).__init__()

        # 本项目有20个类别，一个背景，一共21类
        n_class=21

        # conv1
        # 输入图像为3通道，输出64个特征图，卷积核大小为（3，3），步长为1，padding为100（避免图片不兼容，其实也可以为1的）
        # 卷积输出公式：output=(input+2*padding-kernel_size)/stride+1
        #  512=(512+2*1-3)/1+1
        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(num_features=64)
        self.relu1_1=nn.ReLU(inplace=True)

        self.conv1_2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1_2=nn.BatchNorm2d(num_features=64)
        self.relu1_2=nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        # 采样输出公式：output=(input+2*padding-kernel_size)/stride+1
        # 256=(512+2*0-2)/2+1
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)

        # conv2
        self.conv2_1=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn2_1=nn.BatchNorm2d(num_features=128)
        self.relu2_1=nn.ReLU(inplace=True)

        self.conv2_2=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn2_2=nn.BatchNorm2d(num_features=128)
        self.relu2_2=nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2)


        # conv3
        self.conv3_1=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_1=nn.BatchNorm2d(num_features=256)
        self.relu3_1=nn.ReLU(inplace=True)

        self.conv3_2=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_2=nn.BatchNorm2d(num_features=256)
        self.relu3_2=nn.ReLU(inplace=True)

        self.conv3_3=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_3=nn.BatchNorm2d(num_features=256)
        self.relu3_3=nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        self.maxpool3=nn.MaxPool2d(kernel_size=2,stride=2)


        # conv4
        self.conv4_1=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn4_1=nn.BatchNorm2d(num_features=512)
        self.relu4_1=nn.ReLU(inplace=True)

        self.conv4_2=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn4_2=nn.BatchNorm2d(num_features=512)
        self.relu4_2=nn.ReLU(inplace=True)

        self.conv4_3=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn4_3=nn.BatchNorm2d(num_features=512)
        self.relu4_3=nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        self.maxpool4=nn.MaxPool2d(kernel_size=2,stride=2)


        # conv5
        # 输入图像为3通道，输出64个特征图，卷积核大小为（3，3），步长为1，padding为100（避免图片不兼容）
        self.conv5_1=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn5_1=nn.BatchNorm2d(num_features=512)
        self.relu5_1=nn.ReLU(inplace=True)

        self.conv5_2=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn5_2=nn.BatchNorm2d(num_features=512)
        self.relu5_2=nn.ReLU(inplace=True)

        self.conv5_3=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn5_3=nn.BatchNorm2d(num_features=512)
        self.relu5_3=nn.ReLU(inplace=True)

        # 最大池化层进行下采样
        self.maxpool5=nn.MaxPool2d(kernel_size=2,stride=2)


        # cnov6
        self.conv6=nn.Conv2d(in_channels=512,out_channels=4096,kernel_size=7,stride=1,padding=1)
        self.bn6=nn.BatchNorm2d(num_features=4096)
        self.relu6=nn.ReLU(inplace=True)
        self.drop6=nn.Dropout2d(p=0.5)

        # cnov7
        self.conv7=nn.Conv2d(in_channels=4096,out_channels=4096,kernel_size=1,stride=1,padding=1)
        self.bn7=nn.BatchNorm2d(num_features=4096)
        self.relu7=nn.ReLU(inplace=True)
        self.drop7=nn.Dropout2d(p=0.5)

        # cnov8，本项目有20个类别，一个背景，一共21类
        self.conv8=nn.Conv2d(in_channels=4096,out_channels=n_class,kernel_size=1,stride=1,padding=1)

        # 上采样2倍（16，16，21）————>（32，32，21）
        self.up_conv8_2=nn.ConvTranspose2d(in_channels=n_class,out_channels=n_class,kernel_size=2,stride=2,bias=False)

        # 反卷积ConvTranspose2d操作输出宽高公式
        # output=((input-1)*stride)+outputpadding-(2*padding)+kernelsize
        # 34=(16-1)*2+0-(2*0)+4

        # 第4层maxpool值做卷积运算
        self.pool4_conv=nn.Conv2d(in_channels=512,out_channels=n_class,kernel_size=1,stride=1)

        # 利用反卷积上采样2倍
        self.up_pool4_2=nn.ConvTranspose2d(in_channels=n_class,out_channels=n_class,kernel_size=2,stride=2,bias=False)

        # 第3层maxpool值做卷积运算
        self.pool3_conv=nn.Conv2d(in_channels=256,out_channels=n_class,kernel_size=1,stride=1)

        # 利用反卷积上采样8倍
        self.up_pool3_8=nn.ConvTranspose2d(in_channels=n_class,out_channels=n_class,kernel_size=8,stride=8,bias=False)


    def forward(self,x):
        """正向传播"""

        # 记录初始图片的大小（32，21，512，512）
        h=x

        # conv1
        x=self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x=self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x=self.maxpool1(x)
        test1=x

        # conv2
        x=self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x=self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x=self.maxpool2(x)
        test2=x

        # conv3
        x=self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x=self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x=self.relu3_3(self.bn3_3(self.conv3_3(x)))
        x=self.maxpool3(x)
        pool3=x
        test3=x

        # conv4
        x=self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x=self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x=self.relu4_3(self.bn4_3(self.conv4_3(x)))
        x=self.maxpool4(x)
        pool4=x
        test4=x

        # conv5
        x=self.relu5_1(self.bn5_1(self.conv5_1(x)))
        x=self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x=self.relu5_3(self.bn5_3(self.conv5_3(x)))
        x=self.maxpool5(x)
        test5=x

        # conv6
#         print(self.conv6(x).shape)
#         print(self.bn6(self.conv6(x)).shape)
#         print(self.relu6(self.bn6(self.conv6(x))).shape)
#         print(self.drop6(self.relu6(self.bn6(self.conv6(x)))).shape)
        x=self.drop6(self.relu6(self.bn6(self.conv6(x))))
        test6=x

        # conv7
        x=self.drop7(self.relu7(self.bn7(self.conv7(x))))
        test7=x

        # conv8
        x=self.up_conv8_2(self.conv8(x))
        up_conv8=x
        test8=x

        # 计算第4层的值
        x2=self.pool4_conv(pool4)
        test9_1=x2
        #test9_2=test8
        # 相加融合
        #print(up_conv8.size())
        #print(x2.size())
        x2=up_conv8+x2
        # 反卷积上采样8倍
        test9_3=x2
        x2=self.up_pool4_2(x2)
        up_pool4=x2
        test9_4=x2

        # 计算第3层的值
        x3=self.pool3_conv(pool3)
        test10_1=x3
        #test10_2=test9_4
        # 相加融合
        print(up_pool4.size())
        print(x3.size())
        x3=up_pool4+x3
        test10_3=x3

        # 反卷积上采样8倍
        x3=self.up_pool3_8(x3)
        return x3
        #return test1


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
model.load_state_dict(torch.load("./fcn8s_from_vgg.pt"))
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

