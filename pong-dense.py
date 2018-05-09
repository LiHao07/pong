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
EPOCH=3000
HIDDEN_SIZE=200
LR=0.0001
GAMMA=0.99
RENDER=True
TRAIN_TYPE=1
CHANNEL1=8
CHANNEL2=16
CAP=3000
BATCH_SIZE=1000
LIM=1000
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
        self.out=nn.Linear(HIDDEN_SIZE,3)
#         self._layer_init(self.nn1)
#         self._layer_init(self.out)
        
    def _layer_init(self, layer):
        init.xavier_uniform(layer.weight)
        init.constant(layer.bias, 0.1)

    def forward(self,x):
        x=x.view(-1,6400)
        x=self.nn1(x)
        x=F.relu(x)
        x=self.out(x)
        x=F.softmax(x,dim=1)
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
        p_=output.data.numpy().ravel()
        for i in range(3):
            sta[int(p_[i]*10)]+=1
        a= np.random.choice(range(3), p=p_)
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
        BATCH=1
        BATCH_SIZE=tot//BATCH
        while(BATCH_SIZE>LIM):
            BATCH+=1
            BATCH_SIZE=tot//BATCH
        print('tot : %d | BATCH %d | BATCH_SIZE %d'%(tot,BATCH,BATCH_SIZE))
        for j in range(BATCH):
            index=np.random.choice(tot,BATCH_SIZE)
            s=[]
            for i in range(BATCH_SIZE):
                s.append(self.s[index[i]])
            s=torch.stack(s,dim=0)
            s=Variable(s)
            p=self.net(s)
            loss=0
            for i in range(BATCH_SIZE):
                loss+=torch.log(1-p[i][self.a[index[i]]])*self.vt[index[i]]
                
            loss/=BATCH_SIZE
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
        a_=0 if(a==0) else a+2
        s_,r,done,info=env.step(a_)        
        reward+=r

        pg.store(s,a,r)
        pre_s=s
        s=prepare(s_)
        if(RENDER):env.render()
    if(RENDER):env.close()    
    ep_r=ep_r*0.99+reward*0.01
    
    print('epoch %d | reward %d | ep_r %.4f'%(epoch,reward,ep_r))
    if(len(pg.vt)>LIM):
        pg.learn()
    print(sta)
    print('')
    
    if(epoch%20==0):
        torch.save(pg.net.state_dict(), 'pg_net4.pkl')  
    epoch+=1
    
    
    