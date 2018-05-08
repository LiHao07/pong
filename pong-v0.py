import torch
import tensorflow as tf
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.nn import init
import numpy as np
import gym
from PIL import Image

#hyper parameters
EPOCH=3000
HIDDEN_SIZE=200
LR=0.0001
GAMMA=0.99
RENDER=False
TRAIN_TYPE=1
TRANS=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

tot=0
def prepare(i):
    i=i[35:195,:,:]
    i=i[::2,::2,0:1]
    i[i == 144] = 0
    i[i == 109] = 0
    i[i>0]=1
    i=TRANS(i)
    return i

def image_plt(fig):
    x=fig.shape[0]
    y=fig.shape[1]
    image=Image.new('RGB',(x,y))
    for i in range (0,x):
        for j in range (0,y):
            image.putpixel([i,j],(fig[i][j],fig[i][j],fig[i][j]))
    image.show()
    
class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,8,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(8,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.nn1=nn.Linear(16*20*20,HIDDEN_SIZE)
        self.out=nn.Linear(HIDDEN_SIZE,1)
#         self._layer_init(self.nn1)
#         self._layer_init(self.out)
        
    def _layer_init(self, layer):
        init.xavier_uniform(layer.weight)
        init.constant(layer.bias, 0.1)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(-1,6400)
        x=self.nn1(x)
        x=F.relu(x)
        x=self.out(x)
        x=F.sigmoid(x)
        return x

class PG(object):
    def __init__(self):
        self.net=NET()
        self.opt=torch.optim.Adam(self.net.parameters(),lr=LR)
        self.s=[]
        self.a=[]
        self.r=[]
        self.vt=[]
    
    def choose(self,x):
        x=x.view(-1,1,80,80)
        x=Variable(x)
        output=self.net(x)
        a=0 if (np.random.uniform() <output.data.numpy()[0]) else 1
        return a
    
    def store(self,s,a,r):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        if(TRAIN_TYPE==2 and r<0):
            tmp=len(self.r)-1
            tot=0
            while(tmp>=0 and self.r[tmp]<=0):
                tmp-=1
                tot+=1
            if tot==len(self.s):self.s=[]
            else:self.s=self.s[0:len(self.s)-tot]
            if tot==len(self.a):self.a=[]
            else:self.a=self.a[0:len(self.a)-tot]
            if tot==len(self.r):self.r=[]
            else:self.r=self.r[0:len(self.r)-tot]
        if(r!=0):
            tmp=self.discount_r(self.r)
            self.vt+=self.discount_r(self.r).tolist()
            
            if(len(self.s)>512):
                #self.s=np.asarray(self.s)
                self.learn()
                return 1
        return 0
    def learn(self,):
        batch_size=len(self.s)
        self.s=torch.stack(self.s,dim=0)
        self.s=Variable(self.s)
        p=self.net(self.s)
        loss=0
        for i in range(batch_size):
#             if(self.a[i]==1):
#                 loss+=-(1-p[i])*self.vt[i]
#             else:
#                 loss+=-p[i]*self.vt[i]
            if(self.a[i]==1):
                loss+=-(1-torch.log(p[i]))*self.vt[i]
            else:
                loss+=-torch.log(p[i])*self.vt[i]
        loss/=batch_size
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        self.s=[]
        self.a=[]
        self.r=[]
        self.vt=[]
        
    def discount_r(self,r):
        batch_size=len(r)
        r_=np.zeros(batch_size)    
        tot = 0
        for i in reversed(range(0, batch_size)):
            if(r[i]!=0):
                tot=0
            tot=tot*GAMMA+r[i]
            r_[i]=tot
#         r_-=np.mean(r_)
#         r_/=np.std(r_)
        return r_
    
env = gym.make("Pong-v0") 
env = env.unwrapped
pg=PG()
Max=-22
epoch=0
while(True):
    s=env.reset()
    for i in range(20):
        s,r,done,info=env.step(0)
    s=prepare(s)
    tot=0
    pre_s=TRANS(np.zeros(shape=(80,80,1)))
    done=False
    reward=0
    while(not done):
        x=s-pre_s
        pre_s=s
        a=pg.choose(x)
        a_=a+2
        s_,r,done,info=env.step(a_)
        reward+=r
        tot+=pg.store(s,a,r)
        s=prepare(s_)
        if(RENDER and epoch%10==0):
            env.render()
    env.close()
    if(reward>Max):
        torch.save(pg.net.state_dict(), 'pg_net3.pkl')  # 保存整个网络
    #if(epoch%10==0):
    print('epoch %d | reward %d %d'%(epoch,reward,tot))
    epoch+=1
    
    
    