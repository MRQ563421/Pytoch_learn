# Anaconda Prompt

## 配置

### 切换环境

```conda
conda activate pytorch
```

### 显示环境中的Package

```python
pip list
```

### 显示驱动版本

```
nvidia-smi
```

### 查看cuda是否可以在本的电脑使用

```python
torch.cuda.is_available() 
```

### conda下载

```
pip install nb_conda -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
```



# Python中的两大函数

```python
dir()  #打开，看见
help()#说明书

```



1、首先打开Anaconda Prompt，命令行输入：conda install -c conda-forge nb_conda_kernels

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230410223709482.png" alt="image-20230410223709482" style="zoom: 80%;" />





# Pytorch加载数据

## 1.  Dataset

提供一种方式去获取数据及其label

功能包含：a. 如何获取一个数据及其label

​					b.告诉我们一共有多少数据



类比打牌

dataset就是牌 dataloader是抓牌

告诉我们数据集在哪里，取出里面 的某个数据，他的标签，数据集的大小

## 2.  Dataloader 

为后面的网络提供不同的数据形式





pip install --user --ignore-installed jupyter 这里无法运行的兄弟   原因好像是:内核死亡  无法连接

显示error，no module named "torch"错误的  右上角的内核没选pytorch  如果jupyter没有那个选项只能b战搜索jupyter安装pytorch 装一下插件



## 3.案例

### #__getitem__魔术方法的作用

```python
#__getitem__魔术方法的作用

class Tag:
    def __init__(self):
        self.change = {'python': 'This is python'}

    def __getitem__(self, item):
        print('这个方法被调用')
        return self.change[item]

a = Tag()
print(a['python'])

#结果是  
#这个方法被调用
#This is python

#案例2
class Tag:
    def __init__(self,id):
        self.id=id
 
    def __getitem__(self, item):
        print('这个方法被调用')
        return self.id
 
a=Tag('This is id')
print(a.id)
print(a['python'])

#输出
#This is id
#这个方法被调用
#This is id
```

### 安装opencv

pip install opencv-python



### 蜜蜂和蚂蚁的案例--使用Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataSet(Dataset):

    def __init__(self,root_path,label_path):
        self.root_path = root_path
        self.label_path = label_path
        self.path = os.path.join(root_path,label_path)   # 将两个路径拼接起来形成真实路径
        self.img_dir = os.listdir(self.path)           # 将path路径下的信息提取出来形成list，这里的信息是文件名

    # 根据索引建立图片对象，并返回对象和label信息
    def __getitem__(self, idx):
        img_name = self.img_dir[idx]

        img_path = os.path.join(self.root_path,self.label_path,img_name) # 获取单个图片的路径
        # img_path = os.path.join(self.path, img_name) 是错误的
        img = Image.open(img_path)  # 调用Image.open()，获取文件对象

        label = self.label_path
        return img, label

    def __len__(self):
        return len(self.img_dir)

root = "DataSet//train"
label_ants = "ants"
mydataset_ants = MyDataSet(root,label_ants)

label_bees = "bees"
mydataset_bees = MyDataSet(root,label_bees)

mydata = mydataset_bees + mydataset_ants  # 数据集合之间可以直接相加合并
img,label = mydata[200] # 取出数据集合中的某一具体的数据及其label

```



# TensorBoard

没装包的同学试试conda install -c conda-forge tensorboard

## SummaryWriter的使用

### write.add_scalar()方法  

```python
    # 源码
    
    def add_scalar(
        self,
        tag,                 #代表标题
        scalar_value,        # y轴
        global_step=None,    # x轴
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
        """Add scalar data to summary.

        Args:
            tag (string): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event
            new_style (boolean): Whether to use new style (tensor field) or old
              style (simple_value field). New style could lead to faster data loading.
        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()
```



```python
from torch.utils.tensorboard import SummaryWriter

write = SummaryWriter("lags")

for i in range(1000):
    write.add_scalar("Y=2x",2*i,i)  # 添加标量

write.close()  # 关闭
```

运行之后会产生一个lags事件文件

![image-20230412142502715](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230412142502715.png)

#### 打开.add_scalar产生的事件文件方法

在终端打开，也就是交互式窗口

```
tensorboard --logdir=lags --port=6007    后面是指定端口号   
```

​	<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230412143345388.png" alt="image-20230412143345388" style="zoom: 80%;" />



```python
from torch.utils.tensorboard import SummaryWriter

write = SummaryWriter("lags")

for i in range(1000):
    write.add_scalar("Y=2x",3*i,i)  # 添加标量

write.close()  # 关闭
```

改变变量可能会包含原来的事件文件，解决方法就是  删除原来 lags下的全部事件文件，重新开始。

### 添加图片.add_image

```python
from torch.utils.tensorboard import SummaryWriter
import numpy
from PIL import Image

img_path = "DataSet/train/ants/0013035.jpg"
image = Image.open(img_path)  # 类型是PIL.JpegImagePlugin.JpegImageFile

image_array = numpy.array(image) # 转换成numpy类型
print(image_array.shape) # (512, 768, 3) 格式

write = SummaryWriter("logs")

write.add_image("test",image_array,1,dataformats='HWC') # img_tensor要求numpy 或者 tensor型
                                        # math 要求(3, H, W) 3通道 高 宽
for i in range(100):
    write.add_scalar("y=3*x",3*i,i)
```

```python
 #源代码 .add_image  tag标题 img_tensor纵坐标，这里是图片 global_step 横坐标
    
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
```

# transforms

## 将PIL类型的图片格式转换成tensor(通过实例对象的方法实现)

tensor_img = tensor_trans(img)调用了call方法，相当于tennsor_img = tensor_tras.__call__(img)



<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230415222945849.png" alt="image-20230415222945849" style="zoom: 50%;" />



```python
from torchvision import transforms
from PIL import Image
image_path = "DataSet/train/ants/5650366_e22b7e1065.jpg"
image = Image.open(image_path)  # <class 'PIL.JpegImagePlugin.JpegImageFile'>

tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(image)  # <class 'torch.Tensor'>



transforms包里面的 class ToTensor类中的__call__方法

这里使用了 魔术方法 __call__

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)
```

## 常见的Transforms

**输入   **   ** PIL**   **Image.open()

**输出**      **tensor**     ** ToTensor()

**作用**      **narrays**    **cv.imread

### 类型转换#Totensor

```python
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
image_path = "DataSet/train/ants/5650366_e22b7e1065.jpg"
image = Image.open(image_path)  # <class 'PIL.JpegImagePlugin.JpegImageFile'>

tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(image) # <class 'torch.Tensor'>

write = SummaryWriter("logs2")
write.add_image("test",img_tensor,1)

write.close()
```





### 归一化#Normalize

```python
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
image_path = "DataSet/train/ants/0013035.jpg"
image = Image.open(image_path)

# ToTensor
image_totensor = transforms.ToTensor()
image_tensor = image_totensor(image) # 转换成tensor数据类型

write = SummaryWriter("logs1")
write.add_image("Totensor",image_tensor,1)

# Normalize
print(image_tensor[1][1][1]) # tensor(0.5961)
trans_nor = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) # 用来调整信道3个
img_norm = trans_nor(image_tensor)
print(img_norm[1][1][1])  # tensor(0.0961)

write.add_image("Normalize",img_norm,0)

write.close()
```

```python
class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

```

### 调整大小#Resize

```python
# Resize
trans_Resize = transforms.Resize((512,512))  #将图片大小调整成 512*512型
img_Resize = trans_Resize(image)             # 经过Resize 将一个PIL型的图片调整大小，输出PIL型图片

# 将PIL型的图片调整为 Tensor 类型，并输出tensorBoard
img = image_totensor(img_Resize)       
write.add_image("Resize",img,1)
```

### 连续操作#Compose

```python
# Compose
trans_Resize = transforms.Resize(512)  # 等比缩放
# 构造trans工具，先将PIL类型的数据调整大小，然后转换成tensor类型，Compose里面的list接受transforms类型
trans_Compose = transforms.Compose([trans_Resize,image_totensor])  

img = trans_Compose(image)
write.add_image("Resize",img,2)
```

### 随机裁剪#Randomcrop

```python
# randomcrop
trans_randomcrop = transforms.RandomCrop(50) #设置裁剪大小50*50  自定义为（60,60）
trans_compose  = transforms.Compose([trans_randomcrop,image_totensor])

for i in range(10):
    img = trans_compose(image)
    write.add_image("Random",img,i)
```

# Tensorvision中数据集的使用

```python
import torchvision
# 下载数据集，第一个参数root是下载的位置，第二个参数train是是否是测试数据集，
# download是true说明是开始下载，本来有数据基的话不用下载
train_set = torchvision.datasets.CIFAR10("./dataset",train=True,download=True)
test_set = torchvision.datasets.CIFAR10("./dataset",train=False,download=True)
```

```python
print(train_set[2])   #(<PIL.Image.Image image mode=RGB size=32x32 at 0x2AF67C949B0>, 9)
img,taget =test_set[2]
print(img) 			#<PIL.Image.Image image mode=RGB size=32x32 at 0x2AF67C947F0>
print(taget)		#8

img.show()
```

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter
#设置转换规则
dataset_trans = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((512,512)),
     torchvision.transforms.ToTensor()]
)
# 下载数据集，第一个参数root是下载的位置，第二个参数train是是否是测试数据集，
# download是true说明是开始下载，本来有数据基的话不用下载
train_set = torchvision.datasets.CIFAR10("./dataset",train=True,transform=dataset_trans,download=True)
test_set = torchvision.datasets.CIFAR10("./dataset",train=False,transform=dataset_trans,download=True)
writer = SummaryWriter("log")

for i in range(10):
    img,taget =test_set[i]
    writer.add_image("train_set",img,i)
```

# dataloader

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# 下载并配置数据集
test_dataset = torchvision.datasets.CIFAR10("./dataset",train=False,
                                            transform=torchvision.transforms.ToTensor(),download=True)
#dataset=test_dataset 是要操作的数据集
# ,batch_size=4, 每次取出的数据的大小
# shuffle=False,true是随机采样，否则就是每次采样是相同的
# num_workers=0,采用几个线程处理数据，默认0是采用主线程
# drop_last=False 最后不能整除取出的时候，是否保留最后的余数
test_loader = DataLoader(dataset=test_dataset,batch_size=4,shuffle=False,num_workers=0,drop_last=False)

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs,tags = data   #imgs,tags 取出的是batch_size=4的四个的 图片和tag标签
    writer.add_images("dataset",imgs,step)
    step+=1

writer.close()

```



# 神经网络

## 基本骨架nn.Module

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230420205011573.png" alt="image-20230420205011573" style="zoom: 80%;" />

定义每次调用时执行的计算。
应该被所有子类覆盖。

```python
import torch
from torch import nn

class my_module(nn.Module):
    # 可以由generate 直接生成
    def __init__(self) -> None:
        super().__init__() # 这一条语句必须包含在自己的module里面

    # 前向传播，定义每次调用时执行的计算。
    # 应该被所有子类覆盖。
    #Although the recipe for forward pass needs to be defined within this function,
    # one should call the Module instance afterwards instead of this
    # since the former takes care of running the registered hooks while the latter silently ignores them.
    # 尽管前向传递的配方需要在这个函数中定义，
    # 但应该在之后调用Module实例，而不是这样，
    # 因为前者负责运行注册的钩子，而后者则默默地忽略它们。
    def forward(self,input):
        output = input+1
        return output

MyModule = my_module()
# 将x转换成tensor类型
x= torch.tensor(1.0)
output = MyModule(x)
print(output)
```

## 卷积操作

```python
import torch
import torch.nn.functional as F

input = torch.Tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.Tensor([[1,2,1],
                       [0,1,2],
                       [2,1,0]])

print(input.shape) # torch.Size([5, 5])
print(kernel.shape)# torch.Size([3, 3])

#转换数据类型
input= torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
print(input.shape) # torch.Size([1, 1, 5, 5])
print(kernel.shape) # torch.Size([1, 1, 3, 3])   #第一个是batch Size批量数据多少，第二个数据是通道数，
												#  第三数据是高，第四个是宽


# 二维卷积操作，input、kernel是tensor数据类型，同时是（1,1，*，*）
# stride 是卷积核的步长  padding是外围拓展多少格
output = F.conv2d(input,kernel,stride=1)
print(output)

```



## 卷积层

###  Conv2d

```python
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
# ./data是当前路径 ../data是父类路径
test_datasets = torchvision.datasets.CIFAR10("./data",train=False,
                                             transform=torchvision.transforms.ToTensor()
                                             ,download=True)

test_loader = DataLoader(test_datasets,batch_size=64,shuffle=False,drop_last=False)



class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # kernel_size卷积核的大小，里面的值我们不设置，只设置核的大小，训练的时候会动态调整，
        #而卷积核的数量由软件自动设置，主要是依据out_channels和in_channels的比值确定
        self.conv2 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv2(x)
        return x

mymodule = MyModule()
print(mymodule)
# MyModule(
#   (conv2): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
# )

```



最后那个，通道是2，大小是3*3，batch size是1

![image-20230420231805022](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230420231805022.png)



改变batch size    最后那个，通道是1，大小是3*3，batch size是2

![image-20230420231852625](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230420231852625.png)

## 最大池化层

### MaxPool2d()

```python
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.Tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

print(input.shape)       # torch.Size([5, 5])
input= torch.reshape(input,(-1,1,5,5))
print(input.shape) # torch.Size([1, 1, 5, 5])

class my_module(nn.Module):
    def __init__(self):
        super(my_module, self).__init__()
        # ceil_mode=True保留最后 最大池化剩余的取最大值
        self.max_pooling = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.max_pooling(input)
        return output

mymodule = my_module()
input= mymodule(input)
print(input) # tensor([[[[2., 3.],
             # [5., 1.]]]])
```





```python
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10("./data",train=False,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_dataloader  = DataLoader(dataset=datasets,batch_size=64,shuffle=False)

class my_module(nn.Module):
    def __init__(self):
        super(my_module, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool(input)

        return output

writer = SummaryWriter("../logs")

myModule = my_module()
step = 0
for data in test_dataloader:
    imgs,label = datas
    writer.add_images("input",imgs,step)

    # 因为类型是一样的，不会改变通道数所以不用进行类型转换
    output = myModule(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()

```

## 非线性激活

### .ReLU() （大于0 保留，小于0，丢弃）

```python
import torch
from torch import nn
from torch.nn import ReLU

input = torch.Tensor([[1,-1.2],
                     [-2.0,1]])
input = torch.reshape(input,shape=(-1,1,2,2))  # 如果这一句被注释掉的话，最后面print(output.shape)是torch.Size([2, 2])
# 没有被注释掉的话，最后的print(output.shape)是 torch.Size([1, 1, 2, 2])


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.Relu = ReLU(inplace=False)  # inplace=False保留原始数据，否则不保留，会被结果覆盖

    def forward(self,input):
        output = self.Relu(input)

        return output

mymodule = MyModule()
output = mymodule(input)
print(output.shape)
```

### .Sigmoid()

利用tensorboard做输出

```python
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.relu = ReLU(inplace=False)
        self.sigmoid1 = Sigmoid()
    def forward(self,input):
        output = self.sigmoid1(input)

        return output

test_datasets = torchvision.datasets.CIFAR10("./data",train=False,
                                             transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset=test_datasets,batch_size=5)

mymodule = MyModule()

writer = SummaryWriter("../logs_relu")
step =0
for data in dataloader:
    imgs,labels = data
    writer.add_images("input",imgs,global_step=step)

    output = mymodule(imgs)
    writer.add_images("output",img_tensor=output,global_step=step)
    step +=1

writer.close()

```

## 线性层

将图片展开成一排

```python
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = Linear(in_features=12288,out_features=20) #输入是12288，输出是20

    def forward(self,input):
        output = self.linear(input)

        return output
mymoule = MyModule()

test_datasets = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset=test_datasets,batch_size=4)


for data in dataloader:
    imgs,label = data
    print(imgs.shape) # torch.Size([4, 3, 32, 32])

    output = torch.reshape(imgs,(1,1,1,-1))
    print(output.shape) # torch.Size([1, 1, 1, 12288])


    #         imgs               ->           output            ->     output
    # torch.Size([4, 3, 32, 32]) ->torch.Size([1, 1, 1, 12288]) ->torch.Size([1, 1, 1, 20])
    output= mymoule(output)
    print(output.shape)
```



利用flatten直接展平

```python
for data in dataloader:
    imgs,label = data
    print(imgs.shape) # torch.Size([4, 3, 32, 32])

    output = torch.flatten(imgs)  # torch.Size([12288])
    print(output.shape)           # flatten 直接展平

    output = mymoule(output)   # torch.Size([20])
    print(output.shape)
```

## 搭建小实战-Sequential

**cifar10 model structure**



![Structure-of-CIFAR10-quick-model](C:\Users\MRQ\Desktop\深度学习\Structure-of-CIFAR10-quick-model.png)

![image-20230421220101941](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230421220101941.png)



**dilation 是表示是否是空洞卷积，不是的话，dilation是1**

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

        self.conv1 = Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=1024,out_features=64)
        self.linear2 = Linear(in_features=64,out_features=10)

    def forward(self,x):
        x = self.conv1(x)
        x= self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x= self.maxpool3(x)
        x= self.flatten(x)
        x= self.linear1(x)
        x = self.linear2(x)

        return x

mymodule  = MyModule()
print(mymodule)  
# MyModule(
#   (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear1): Linear(in_features=1024, out_features=64, bias=True)
#   (linear2): Linear(in_features=64, out_features=10, bias=True)
# )

input = torch.ones((64,3,32,32))
output = mymodule(input)
print(output.shape)  # torch.Size([64, 10])
```



### 使用Sequential简化代码

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

        self.sequential = Sequential(

            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10),
        )

    def forward(self,x):
       x = self.sequential(x)
       return x


mymodule  = MyModule()
print(mymodule)

input = torch.ones((64,3,32,32))
output = mymodule(input)
print(output.shape)  # torch.Size([64, 10])

```

## 损失函数和反向传播

**L1Loss损失函数**

```python
import torch
from torch import nn
input =  torch.Tensor([1,2,3])
target = torch.Tensor([4,5,6])
input = torch.reshape(input,(1,1,1,3))
target = torch.reshape(target,(1,1,1,3))

loss = nn.L1Loss()
output = loss(input,target)
print(output)  # tensor(3.)

```

**nn.MSELoss()**

```python
# 平方差
mseloss = nn.MSELoss()
output_mse = mseloss(input,output)
print(output_mse)  # tensor(1.6667) 
```



### 交叉熵

####  nn.CrossEntropyLoss()

![image-20230422101836563](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422101836563.png)



```python
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x= torch.reshape(x,(1,3))
# 交叉熵
loss_cross = nn.CrossEntropyLoss()
result = loss_cross(x,y)
print(result)  # tensor(1.1019)
```

#### 利用神经网络

```python
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

        self.sequential = Sequential(

            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10),
        )

    def forward(self,x):
       x = self.sequential(x)
       return x


dataset  =torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=2)

cross_loss = nn.CrossEntropyLoss()
mymodule = MyModule()
for data in dataloader:
    imgs,target = data
    output = mymodule(imgs)
    
    # 计算经过神经网路的结果与实际target目标的差别
    result_loss = cross_loss(output,target)
    print(result_loss)

```

## 优化器

### 随机梯度下降-torch.optim.SGD

```python
# lr是学习速率 params=mymodule.parameters()是参数
optim = torch.optim.SGD(params=mymodule.parameters(),lr=0.01)
for data in dataloader:
    imgs,target = data
    output = mymodule(imgs)

    # 计算经过神经网路的结果与实际target目标的差别
    result_loss = cross_loss(output,target)

    optim.zero_grad()           # 每次循环开始的时候先将梯度置零，否则每一步的梯度会一直增加
    result_loss.backward()     # 根据梯度方向传播，以减小梯度 在这一步会计算梯度
    optim.step()                # 优化器减小梯度
    print(result_loss)
```



```python
# lr是学习速率 params=mymodule.parameters()是参数
optim = torch.optim.SGD(params=mymodule.parameters(),lr=0.01)

for epoch in range(20):

    loss = 0    # 计算每一轮循环的损失函数 也就是误差之和
    for data in dataloader:
        imgs,target = data
        output = mymodule(imgs)

        # 计算经过神经网路的结果与实际target目标的差别
        result_loss = cross_loss(output,target)

        optim.zero_grad()           # 每次循环开始的时候先将梯度置零，否则每一步的梯度会一直增加
        result_loss.backward()     # 根据梯度方向传播，以减小梯度 在这一步会计算梯度
        optim.step()                # 优化器减小梯度 调整模型中的参数

        loss = loss + result_loss  # 计算每一轮loss
    print(loss)
```



**完整代码**

```python
import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

        self.sequential = Sequential(

            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10),
        )

    def forward(self,x):
       x = self.sequential(x)
       return x


dataset  =torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=2)

cross_loss = nn.CrossEntropyLoss()
mymodule = MyModule()

# lr是学习速率 params=mymodule.parameters()是参数
optim = torch.optim.SGD(params=mymodule.parameters(),lr=0.01)

for epoch in range(20):

    loss = 0    # 计算每一轮循环的损失函数 也就是误差之和
    for data in dataloader:
        imgs,target = data
        output = mymodule(imgs)

        # 计算经过神经网路的结果与实际target目标的差别
        result_loss = cross_loss(output,target)

        optim.zero_grad()           # 每次循环开始的时候先将梯度置零，否则每一步的梯度会一直增加
        result_loss.backward()     # 根据梯度方向传播，以减小梯度 在这一步会计算梯度
        optim.step()                # 优化器减小梯度 调整模型中的参数

        loss = loss + result_loss  # 计算每一轮loss
    print(loss)
```

# 现有的网络模型的使用和修改

## vgg16网络模型

```python
# 下载网络模型
vgg16_true = torchvision.models.vgg16(pretrained=True)      # 预先已经训练好了
vgg16_false = torchvision.models.vgg16(pretrained=False)  # pretrained = False预先不训练

print(vgg16_true)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
print(vgg16_false)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
```

### (classifier): Sequential( 外面增加模型

**vgg16_true.add_module( "add_module1",  nn.Linear(in_features=1000,out_features=10  ))**

```python
# 在(classifier): Sequential( 外面增加模型
vgg16_true.add_module("add_module1",nn.Linear(in_features=1000,out_features=10))
print(vgg16_true)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
#   (add_module1): Linear(in_features=1000, out_features=10, bias=True)
# )

# 在(classifier): Sequential( 里面增加模型
```



###  里面增加模型

**vgg16_true.classifier.add_module(**

```python
# 在(classifier): Sequential( 里面增加模型
vgg16_true.classifier.add_module("add_module2",nn.Linear(in_features=1000,out_features=10))
print(vgg16_true)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#     (add_module2): Linear(in_features=1000, out_features=10, bias=True)
#   )
#   (add_module1): Linear(in_features=1000, out_features=10, bias=True)
# )


```

### 直接修改模型

**vgg16_false.classifier[6] = nn.Linear(**

```python
# 直接修改(classifier): Sequential(  里面的模型
vgg16_false.classifier[6] = nn.Linear(in_features=4096,out_features=10)
print(vgg16_false)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=10, bias=True)
#   )
# )
```

# 网络模型的保存与读取

## 方式1 

### torch.save

使用官方模型Vgg16

将模型导入

```python
import torch
import torchvision.models

vgg16_false = torchvision.models.vgg16(pretrained=False)
```

```python
# 方式一：模型结构 + 模型参数
torch.save(vgg16_false,"vgg16_methon1.pth") # 第一个参数是要保存的网络模型，第二个参数是保存的地址/路径
```

### torch.load

```python
# 方式一加载模型
import torchvision.models

module = torch.load("vgg16_methon1.pth")
```





## 使用自己的模型

### 方式一模型保存

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=2)
    
    def forward(self,input):
        output = self.conv1(input)
        
        return output

mymodule = MyModule()
torch.save(mymodule,"mymodule.pth")
```

### 模型加载



```python
from torch import nn
# 要先声明或者先导入才能使用 mymodule = torch.load("mymodule.pth")
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=2)

    def forward(self,input):
        output = self.conv1(input)

        return output
# 这里相对于直接建立网络，少了mymodule = Mymodule()语句
mymodule = torch.load("mymodule.pth")
```

```python
# 也可以这样直接全部导入

from nn_save import *
mymodule = torch.load("mymodule.pth")
```

## 方式二

### 模型保存

```python
import torch
import torchvision.models

vgg16_false = torchvision.models.vgg16(pretrained=False)


# 方式二:只保存模型参数  保存格式是 字典型
torch.save(vgg16_false.state_dict(),"vgg16_methon2.pth")

```

### 模型加载

```python
# 方式二加载模型
# 第一步先新建网络模型
vgg16 = torchvision.models.vgg16(pretrained=False)

# 第二步加载参数
parameter = torch.load("vgg16_methon2.pth")

# 网络模型加载参数
vgg16.load_state_dict(parameter)
```

# 完整的网络训练套路

## 1-准备数据

```python
import torchvision.datasets

train_data = torchvision.datasets.CIFAR10("./data",train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./data",train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
```

## 2-加载数据

```python
# 2- 加载数据
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)
```

### 数据的长度

```python
# 数据的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("测试数据集的长度是：{}".format(test_data_size))
print("训练数据集的长度是：{}".format(train_data_size))
```



## 3-创建网络



![image-20230422170651740](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422170651740.png)

**放在MyModule文件中，使用前要先导包**

```python
import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(in_features=64,out_features=10)
        )

    def forward(self, x):
        x = self.module1(x)

        return x

# 网络测试
if __name__ == '__main__':
    input = torch.ones((64,3,32,32)) # 第一个64 batch_size表示的是图片的数目，第二个是通道，最后两个是H*W，宽×高
    mymodule = MyModule()

    output = mymodule(input)
    print(output.shape) # torch.Size([64, 10])
                        # 返回64代表64行数据，每行10个数据，代表每一张图片在10个类别中的概率
```



 

```python
# 要保证自己建立的MYMODULE和训练的文件在同一个文件夹，引用
# 3- 创建网络模型
from MyModule import *
mymodule = MyModule()
```

## 4-损失函数（交叉熵）

```python
# 4- 损失函数-交叉熵
loss_fn = nn.CrossEntropyLoss()
```



## 5-优化器(设置模型参数和学习速率)

```python
# 优化器
learning_rate = 1e-1# 训练速率
optim = torch.optim.SGD(mymodule.parameters(),learning_rate)
```



## 6- 设置训练网络的一些参数

```python
#记录训练次数
total_train_step =0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
```

## 7-开始训练

```python
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i))

    # 训练步骤开始
    for data in train_dataloader:
        imgs,target = data
        output = mymodule(imgs) # 经过神经网络获得输出

        # 计算损失函数-交叉熵
        loss = loss_fn(output,target)	

        # 进行优化
        optim.zero_grad()  # 优化前要将梯度置零
        loss.backward()     # 计算梯度，反向传播
        optim.step()        # 根据梯度进行优化参数

        # 总训练次数加一
        total_train_step +=1
        print("训练次数{}，损失{}".format(total_train_step,loss))
        
        # 损失函数计算偏差，优化器根据梯度优化模型参数
```

```python
-----第0轮训练开始-----
训练次数1，损失2.300030469894409
训练次数2，损失2.2911341190338135
训练次数3，损失2.3091936111450195
训练次数4，损失2.3224315643310547
训练次数5，损失2.28849458694458
```

## 数据类型

```python
a= torch.tensor(5.)
print(a)   # tensor(5.)
print(a.item()) # 5.0
```



## 测试集

```python
# 测试步骤开始， torch.no_grad()将梯度置零
with torch.no_grad():
    total_test_loss =0
    for data in test_dataloader:
        imgs,target = data
        output = mymodule(imgs)
        loss = loss_fn(output,target)

        total_test_loss += loss
    print("测试集的LOSS：{}".format(total_test_loss))
```

```python

#总代码（训练过程）

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i))

    # 训练步骤开始
    for data in train_dataloader:
        imgs,target = data
        output = mymodule(imgs)

        # 计算损失函数-交叉熵
        loss = loss_fn(output,target)

        # 进行优化
        optim.zero_grad()  # 优化前要将梯度置零
        loss.backward()     # 计算梯度，反向传播
        optim.step()        # 根据梯度进行优化参数

        # 总训练次数加一
        total_train_step +=1
        if total_train_step % 200 == 0:
            print("训练次数{}，损失{}".format(total_train_step,loss))

    # 测试步骤开始， torch.no_grad()将梯度置零
    with torch.no_grad():
        total_test_loss =0
        for data in test_dataloader:
            imgs,target = data
            output = mymodule(imgs)
            loss = loss_fn(output,target)

            total_test_loss += loss
        print("测试集的LOSS：{}".format(total_test_loss))
```

## 准确率

![image-20230422201502181](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422201502181.png)

```python
import torch

# 如[0.1,0.3]表示第一个输入，其是分类0的概率是0.1，在分类1的概率是0.3
                    # 在0、1分类中的概率
input = torch.tensor([[0.1,0.3],
                    [0.2,0.7]])


print(input.argmax(1)) # tensor([1, 1])
# 1表示横向看，0表示纵向比较 也就是[0.1,0.2]表示是第一个输入，其是分类0的概率是0.1，在分类1的概率是0.2

targets = torch.tensor([0,1]) # 两个输入的目标分类分别是0 和1，也就是两个输入分别属于0、1
preds = input.argmax(1)  # 根据各个输入在各个分类的概率，找到最大的概率是在哪个分类，输出给preds 各个输入是属于哪个分类
print((preds == targets).sum()) # 各个分类求和，预测对了为1，没有预测对就是0，将各个分类正确的加和，表示预测正确的个数
# tensor(1)
```

## 9-完整的训练套路

```python
import torch
import torchvision.datasets


# 1- 准备数据
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 2- 加载数据
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)


# 数据的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("测试数据集的长度是：{}".format(test_data_size))
print("训练数据集的长度是：{}".format(train_data_size))

# 要保证自己建立的MYMODULE和训练的文件在同一个文件夹，引用
# 3- 创建网络模型
from MyModule import *
mymodule = MyModule()

# 4- 损失函数-交叉熵
loss_fn = nn.CrossEntropyLoss()

#5- 优化器
learning_rate = 1e-1# 训练速率
optim = torch.optim.SGD(mymodule.parameters(),learning_rate)


#6- 设置训练网络的一些参数

#记录训练次数
total_train_step =0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 7-开始训练
writer =SummaryWriter("../logs_train")
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i))

    # 训练步骤开始
    mymodule.train() # 这一步只对部分训练集合有效
    for data in train_dataloader:
        imgs,target = data
        output = mymodule(imgs)

        # 计算损失函数-交叉熵
        loss = loss_fn(output,target)

        # 进行优化
        optim.zero_grad()  # 优化前要将梯度置零
        loss.backward()     # 计算梯度，反向传播
        optim.step()        # 根据梯度进行优化参数

        # 总训练次数加一
        total_train_step +=1
        if total_train_step % 100 == 0:
            print("训练次数{}，损失{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)  # 写到tensorboard中


    # 测试步骤开始， torch.no_grad()将梯度置零
    mymodule.eval() # 开启评估
    with torch.no_grad():
        total_test_loss =0

        # 正确率
        total_accuary =0
        for data in test_dataloader:
            imgs,target = data
            output = mymodule(imgs)
            loss = loss_fn(output,target)

            # 一次测试正确率
            accuary = (output.argmax(1) == target).sum()
            total_accuary += accuary

            total_test_loss += loss
        print("测试集的LOSS：{}".format(total_test_loss))

        print("整体测试集上的正确率{}".format(total_accuary/test_data_size))
        # 将测试集写入tensorboard中
        writer.add_scalar("test_loss" ,total_test_loss,total_test_step)
        writer.add_scalar("test_accuary",total_accuary,total_test_step)
        total_test_step +=1

        # 保存每次循环的网络模型
        torch.save(mymodule,"mymodule{}.pth".format(i))
        print("本次模型已经保存")

writer.close()
```

# 利用GPU加速

![image-20230422205405507](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422205405507.png)



主要是图中三个部分进行GPU加速 ，就是在对应的实例对象后面调用conda（）

具体如下：

## 网络模型

```python
# 3- 创建网络模型
from MyModule import *
mymodule = MyModule()
mymodule = mymodule.cuda()
```

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422205955175.png" alt="image-20230422205955175" style="zoom:80%;" />



**优化一下：**

```python
# 3- 创建网络模型
from MyModule import *
mymodule = MyModule()
if torch.cuda.is_available():
    mymodule = mymodule.cuda()
```



<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422210754088.png" alt="image-20230422210754088" style="zoom:80%;" />



## 损失函数

```python
# 4- 损失函数-交叉熵
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
```

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422210042583.png" alt="image-20230422210042583" style="zoom:80%;" />

```python
# 4- 损失函数-交叉熵
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
```





## 数据（输入imgs，标注target）

### 训练数据集

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422210249365.png" alt="image-20230422210249365" style="zoom:80%;" />

```python
# 训练步骤开始
mymodule.train() # 这一步只对部分训练集合有效
for data in train_dataloader:
    imgs,target = data
    if torch.cuda.is_available():
        imgs = imgs.coda()
        target = target.conda()
```

### 评估数据集

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422211047869.png" alt="image-20230422211047869" style="zoom:80%;" />



```python
# 测试步骤开始， torch.no_grad()将梯度置零
mymodule.eval() # 开启评估
with torch.no_grad():
    total_test_loss =0

    # 正确率
    total_accuary =0
    for data in test_dataloader:
        imgs,target = data
        if torch.cuda.is_available():
            imgs = imgs.coda()
            target = target.conda()

        output = mymodule(imgs)
        loss = loss_fn(output,target)
```



# 利用GPU加速（二）

数据、网络模型、损失函数调用 .to(device)         device = torch.device("cpu") 或者 device = torch.device("cuda")  或者 device = torch.device("cuda"   if  torch.cuda.is_available()  else   "cpu"  ) 或者 device = torch.device("cuda : 0")

## 放在最前面

```python
device = torch.device("cpu")
```

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422212248204.png" alt="image-20230422212248204" style="zoom:80%;" />

## 网络模型

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422212403474.png" alt="image-20230422212403474" style="zoom:80%;" />

```python
# 3- 创建网络模型
from MyModule import *
mymodule = MyModule()
# if torch.cuda.is_available():
#     mymodule = mymodule.cuda()
mymodule = mymodule.to(device=device)
```



## 损失函数

```python
# 4- 损失函数-交叉熵
loss_fn = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()
loss_fn = loss_fn.to(device=device)
```

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422212510415.png" alt="image-20230422212510415" style="zoom:80%;" />



## 数据

```python
mymodule.train() # 这一步只对部分训练集合有效
for data in train_dataloader:
    imgs,target = data
    # if torch.cuda.is_available():
    #     imgs = imgs.coda()
    #     target = target.conda()
    imgs = imgs.to(device)  # 这个必需传递给原来的imgs
    target = target.to(device) # 这个必需传递给原来的 target
```

<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422212659788.png" alt="image-20230422212659788" style="zoom:80%;" />





<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422212756408.png" alt="image-20230422212756408" style="zoom:80%;" />

```python
mymodule.eval() # 开启评估
with torch.no_grad():
    total_test_loss =0

    # 正确率
    total_accuary =0
    for data in test_dataloader:
        imgs,target = data
        # if torch.cuda.is_available():
        #     imgs = imgs.coda()
        #     target = target.conda()
        imgs = imgs.to(device)  # 这个必需传递给原来的imgs
        target = target.to(device)  # 这个必需传递给原来的 target
```





# 完整的模型验证套路

![image-20230422221247548](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422221247548.png)



```python
import torch
import torchvision
from PIL import Image

from MyModule import *

# 加载网络模型
mymodule = torch.load("mymodule0.pth")

# 用来验证的图片加载
image_path = "D:/pythonProject/module/data/img.png"
img = Image.open(image_path)
print(img) # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=902x626 at 0x191A5F58710>

# 这里注意  torchvision.transforms.Resize((32,32)), 有 两 个括号
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])

# 将待验证的图片转换成 神经网络所能够识别的格式tensor数据类型
img = trans(img)
print(img.shape)  # tensor数据类型应该用 .shape来查看数据类型，大小
# torch.Size([3, 32, 32])

img = torch.reshape(img,(-1,3,32,32))
mymodule.eval()
with torch.no_grad():
    # 这里需要一个四维的格式
    output = mymodule(img)  #  RuntimeError: Expected 4-dimensional input for 4-dimensional weight [32, 3, 5, 5],
                            # but got 3-dimensional input of size [3, 32, 32] instead
    print(output)
# tensor([[-2.0615,  0.3733, -0.2153, -0.3034, -0.0281, -0.1069,  2.3884,  0.3806,
#          -0.7318,  0.3272]])

# 将上面的每一种概率找到最大值，并返回索引，也就是告诉我们，这个测试的图片是属于哪一类的概率最大
print(output.argmax(1)) # tensor([6])
```

## 也就是说，先打开图片，调整一下格式，用神经网络测试获得输出，根据输出找到属于哪一类







<img src="C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230422223351462.png" alt="image-20230422223351462" style="zoom:80%;" />
