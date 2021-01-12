import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,input = 3, output = 128,conv_kernel = (3,3),conv_padding=(1,1),con_strides = (1,1),
                 pool_kernel= (2,2),pool_stride=(1,1),pool_padding= (0, 0,),dropout_rate = 0.2,is_pool = True):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=conv_kernel, stride=con_strides,padding=conv_padding)
        self.bn = nn.BatchNorm2d(output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_rate)
        self.AveragePooling = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
        self.is_pool = is_pool
    def forward(self,x):
        x = self.dropout(self.relu(self.bn(self.conv(x))))
        if self.is_pool:
            x= self.AveragePooling(x)
        return x

class cnn_3layer_fc_model(nn.Module):
    def __init__(self,n_classes,n1 = 128, n2=192, n3=256, dropout_rate = 0.2,fc=100):
        super(cnn_3layer_fc_model, self).__init__()
        self.Block1 = Block(3, n1,pool_padding= (1, 1),dropout_rate = dropout_rate)
        self.Block2 = Block(n1, n2,conv_kernel = (2,2),con_strides=(2,2), conv_padding=(0,0), pool_kernel = (2,2), pool_stride = (2,2),dropout_rate = dropout_rate)
        self.Block3 = Block(n2, n3,conv_kernel = (3,3),conv_padding=(2,2),dropout_rate = dropout_rate,is_pool = False)
        self.fc = nn.Linear(fc*n3,n_classes,bias=False)

    def forward(self, x):
        x= self.Block1(x)
        x= self.Block2(x)
        x= self.Block3(x)
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.fc(x)
        return x

class cnn_2layer_fc_model(nn.Module):
    def __init__(self,n_classes,n1 = 128, n2=256, dropout_rate = 0.2, fc=100):
        super(cnn_2layer_fc_model, self).__init__()
        self.Block1 = Block(3, n1, pool_padding=(1, 1), dropout_rate=dropout_rate)
        self.Block2 = Block(n1, n2, conv_kernel=(3, 3), con_strides=(2, 2), pool_kernel=(2, 2), pool_stride=(2, 2),
                            dropout_rate=dropout_rate,is_pool=False)
        self.fc = nn.Linear(fc * n2, n_classes,bias=False)

    def forward(self, x):
        x= self.Block1(x)
        x= self.Block2(x)
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.fc(x)
        return x
