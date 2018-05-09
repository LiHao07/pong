import sys
sys.path.append('/Users/lihao/gym')
sys.path.append('/Users/lihao/miniconda3/lib/python3.6/site-packages')

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

#hyper parameters
HIDDEN_SIZE=200
LR=0.0001
GAMMA=0.99
RENDER=True

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
        self._layer_init(self.nn1,math.sqrt(6/6600))
        self._layer_init(self.out,math.sqrt(6/201))
        
    def _layer_init(self, layer,x):
        init.uniform(layer.weight, a=-x, b=x)
        init.constant(layer.bias, 0)

    def forward(self,x):
        x=x.view(-1,6400)
        x=self.nn1(x)
        x=F.relu(x)
        x=self.out(x)
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
        x=Variable(x)
        output=self.net(x)
        output=F.sigmoid(output)
        p_=output.data.numpy()[0]
        sta[int(p_*10)]+=1
        if(random.random()<p_):
            a=0
        else:
            a=1
        return a
    
    def store(self,s,a,r):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        if(r!=0):
            tmp=self.discount_r(self.r)
            self.vt+=self.discount_r(self.r).tolist()
            self.r=[]
    def learn(self,):
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
            loss+=xx - x * z + torch.log(1 + torch.exp(-torch.abs(x)))
                
        loss/=tot
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
#         r_-=np.mean(r_)
#         r_/=np.std(r_)
        return r_

env = gym.make("Pong-v0") 
env = env.unwrapped
pg=PG()
Max=-22
epoch=0
ep_r=-21
while(True):
    s=env.reset()
    for i in range(20):
        s,r,done,info=env.step(0)
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

        pg.store(s,a,r)
        pre_s=s
        s=prepare(s_)
        if(RENDER):env.render()
    if(RENDER):env.close()    
    ep_r=ep_r*0.99+reward*0.01
    
    print('epoch %d | reward %d | ep_r %.4f'%(epoch,reward,ep_r))
    pg.learn()
    print(sta)
    print('')
    
    if(epoch%20==0):
        torch.save(pg.net.state_dict(), 'pg_net_parameters.pkl')  
    epoch+=1
    
    
    