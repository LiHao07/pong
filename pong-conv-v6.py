import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

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
CAP=1500
BATCH_SIZE=1000
LIM=375
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
        self.conv1=nn.Sequential(
            nn.Conv2d(1,CHANNEL1,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(CHANNEL1,CHANNEL2,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.nn1=nn.Linear(CHANNEL2*20*20,HIDDEN_SIZE)
        self.out=nn.Linear(HIDDEN_SIZE,3)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(-1,CHANNEL2*20*20)
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
        self.vt.append(r)
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
        pg.s,pg.a,pg.r,pg.vt=[],[],[],[]
        
def calc_w(s):
    Min1=80
    Max1=-1
    for i in range(80):
        if(int(s[0][i][70])==1 and int(s[0][i][71])==1):
            if(i>Max1):Max1=i
            if(i<Min1):Min1=i
    for i in range(80):
        for j in range(80):
            if(int(s[0][i][j])==1):
                if(j!=70 and j!=71 and j!=8 and j!=9):return abs((Min1+Max1)/2-(i+0.5)),Min1,i+0.5
                else:
                    if( (j==0 or int(s[0][i][j-1])==0) and (j==79 or int(s[0][i][j+1])==0)):
                        return abs((Min1+Max1)/2-(i+0.5)),Min1,i+0.5
    return abs((Min1+Max1)/2-40),Min1,40

def abs(x):
    if(x<0):x=-x
    return x
    
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
    reward2=0
    tot=0
    sta=[0 for i in range(11)]
    pre_delta,_,__=calc_w(s)
    while(not done):
        tot+=1
        x=s-pre_s
        a=pg.choose(x,sta)
        
        a_=0 if(a==0) else a+2
        s_,r,done,info=env.step(a_)
        
        reward+=r
        
        s_=prepare(s_)
        delta,_,__=calc_w(s_)
        
        r=pre_delta-delta
        if(r<-0.01):r*=10
    
        reward2+=r
        
        pg.store(s,a,r)
        
        pre_delta=delta
        pre_s=s
        s=s_
        
        if(RENDER):env.render()
    if(len(pg.vt)>LIM):
        pg.learn()
    
    if(RENDER):env.close()    
    reward2/=tot
    ep_r=ep_r*0.99+reward*0.01
    if(epoch==0):ep_r2=reward2
    else:ep_r2=ep_r2*0.99+reward2*0.01
    
    if(epoch%20==0):
        torch.save(pg.net.state_dict(), 'pg_net4.pkl')  
    
    print('epoch %d | reward %d | ep_r %.4f | ep_r2 %.4f'%(epoch,reward,ep_r,ep_r2))
    print(sta)
    epoch+=1
    
    
    