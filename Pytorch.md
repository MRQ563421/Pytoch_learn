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

s