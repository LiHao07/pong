import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import torch.nn as nn
from torch.nn import init
import numpy as np
import gym

#hyper parameters
EPOCH=3000
HIDDEN_SIZE=200
LR=0.0001
GAMMA=0.99
RENDER=False
TRAIN_TYPE=1
CHANNEL1=16
CHANNEL2=32
CAP=2000
BATCH_SIZE=500
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
        
    def _layer_init(self, layer):
        init.xavier_uniform(layer.weight)
        init.constant(layer.bias, 0.1)

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
    
    def choose(self,x):
        x=x.view(-1,1,80,80)
        x=Variable(x)
        output=self.net(x)
        a= np.random.choice(range(3), p=output.data.numpy().ravel())
        return a
    
    def store(self,s,a,r,flag):
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
        if(flag):
            tmp=self.discount_r(self.r)
            self.vt+=self.discount_r(self.r).tolist()
            self.r=[]
            batch_size=len(self.vt)
            if(batch_size>CAP):
                self.s=self.s[-CAP:]
                self.a=self.a[-CAP:]
                self.vt=self.vt[-CAP:]

    def learn(self,):
        tot=len(self.vt)
        s=[]
        s=torch.stack(self.s[0:tot-1],dim=0)
        s=Variable(s)
        p=self.net(s)
        loss=0
        for i in range(BATCH_SIZE):
            loss+=-torch.log(p[i][self.a[i]])*self.vt[i]
        loss/=BATCH_SIZE
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
    def discount_r(self,r):
        batch_size=len(r)
        r_=np.zeros(batch_size)    
        tot = 0
        for i in reversed(range(0, batch_size)):
            if(r[i]!=0):
                tot=0
            tot=tot*GAMMA+r[i]
            r_[i]=tot
        return r_
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
                if(j!=70 and j!=71 and j!=8 and j!=9):return (Min1+Max1)/2,i+0.5,j
                else:
                    if( (j==0 or int(s[0][i][j-1])==0) and (j==79 or int(s[0][i][j+1])==0)):
                        return (Min1+Max1)/2,i+0.5,j
    return (Min1+Max1)/2,40,40
    
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
    while(not done):
        tot+=1
        x=s-pre_s
        pre_s=s
        a=pg.choose(x)
        a_=0 if(a==0) else a+2
        s_,r,done,info=env.step(a_)
        reward+=r
        flag=(r!=0)
        s_=prepare(s_)
        x1,x0,y0=calc_w(s_)
        tmp=x1-x0
        if(tmp<0):tmp=-tmp
        r=r*100-tmp
        reward2+=r
        pg.store(s,a,r,flag)
        s=s_
        env.render()
        if(RENDER and epoch%10==0):
            env.render()
    if(len(pg.vt)>BATCH_SIZE):
        pg.learn()
        pg.s=[]
        pg.a=[]
        pg.r=[]
        pg.vt=[]
        
    reward2/=tot
    ep_r=ep_r*0.99+reward*0.01
    if(epoch==0):ep_r2=reward2
    else:ep_r2=ep_r2*0.99+reward2*0.01
        
    env.close()
    
    if(epoch%20==0):
        torch.save(pg.net.state_dict(), 'pg_net4.pkl')
    
    print('epoch %d | reward %d | ep_r %.4f | ep_r2 %.4f'%(epoch,reward,ep_r,ep_r2))
    epoch+=1
    
    
    