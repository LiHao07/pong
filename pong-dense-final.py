import random
import math
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import gym.spaces

import torch.nn as nn
from torch.nn import init
import numpy as np
import gym
#hyper  parameters
HIDDEN_SIZE=200
LR=0.00001
GAMMA=0.99
RENDER=False

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
    i=i*255
    return i
    
class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()
        self.nn1=nn.Linear(6400,HIDDEN_SIZE)
        self.out=nn.Linear(HIDDEN_SIZE,1)
        self._layer_init(self.nn1)
        self._layer_init(self.out)
        
    def _layer_init(self, layer):
        init.xavier_uniform_(layer.weight, gain=1)
        init.constant_(layer.bias, val=0)

    def forward(self,x):
        x=x.view(-1,6400)
        x=self.nn1(x)
        x=F.relu(x)
        x=self.out(x)
       # x=F.sigmoid(x)
        return x

class PG(object):
    def __init__(self):
        self.net=NET()
        self.opt=torch.optim.Adam(self.net.parameters(),lr=LR)
        self.s=[]
        self.a=[]
        self.r=[]
        self.vt=[]
    
    def choose(self,x,sta):
        x=x.view(-1,1,80,80)
        x=Variable(x,requires_grad=False)
        output=F.sigmoid(self.net(x))
        p_=output.data.numpy()[0]
        sta[int(p_*10)]+=1
        if (np.random.uniform()<p_):
            a=1
        else:
            a=0
        return a
    
    def store(self,s,a,r):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
    def learn(self,):
        self.vt=self.discount_r(self.r)
        tot=len(self.vt)
        s=torch.stack(self.s[0:tot],dim=0)
        s=Variable(s)
        p=self.net(s)
        p_=F.relu(p)
        loss=0
        for i in range(tot):
            z=self.a[i]
            x=p[i]
            xx=p_[i]
            loss+=(xx - x * z + torch.log(1 + torch.exp(-torch.abs(x))))*self.vt[i]
        
#         p=F.sigmoid(p)
#         loss=0
#         for i in range(tot):
#            # print(self.a[i],torch.log(p[i]).data.numpy(),1-self.a[i],torch.log(1-p[i]).data.numpy(),self.vt[i])
#             loss+=(self.a[i]*(-torch.log(p[i]))+(1-self.a[i])*(-torch.log(1-p[i])))*self.vt[i]
          
        print(loss.data.numpy())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        pg.s=[]
        pg.a=[]
        pg.r=[]
        pg.vt=[]
    def discount_r(self,r):
        batch_size=len(r)
        r_=np.zeros(batch_size)    
        tot = 0
        for i in reversed(range(0, batch_size)):
            if(r[i]!=0):
                tot=0
            tot=tot*GAMMA+r[i]
            r_[i]=tot
        r_-=np.mean(r_)
        r_/=np.std(r_)
        return r_

env = gym.make("Pong-v0") 
#env = env.unwrapped
pg=PG()
pg.net.load_state_dict(torch.load('pg_net_parameters_Max.pkl'))
#Max=-22
epoch=0
ep_r=2.6
while(True):
    s=env.reset()
   # for i in range(20):
    #    s,r,done,info=env.step(0)
    s=prepare(s)
    pre_s=TRANS(np.zeros(shape=(80,80,1))).type('torch.FloatTensor')
    done=False
    reward=0
    tot=0
    sta=[0 for i in range(11)]
    while(not done):
        tot+=1
        x=s-pre_s
        a=pg.choose(x,sta)
        a_=a+2
        s_,r,done,info=env.step(a_)        
        reward+=r

 #       pg.store(x,a,r)
        pre_s=s
        s=prepare(s_)
        if(RENDER):env.render()
    if(RENDER):env.close()    
  #  ep_r=ep_r*0.99+reward*0.01
    
    print('epoch %d | reward %d '%(epoch,reward))
    epoch+=1
