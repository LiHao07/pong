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
LR=0.00001
GAMMA=0.99
RENDER=False
TRAIN_TYPE=2
CHANNEL1=16
CHANNEL2=32
CAP=1000
BATCH_SIZE=70
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
        self.out=nn.Linear(HIDDEN_SIZE,1)
        
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
        if(r>0):
            tmp=self.discount_r(self.r)
            self.vt+=self.discount_r(self.r).tolist()
            batch_size=len(self.s)
            if(batch_size>CAP):
                self.s=self.s[-CAP:]
                self.a=self.a[-CAP:]
                self.vt=self.vt[-CAP:]
        if(r!=0 and len(self.vt)>BATCH_SIZE):
            self.learn()
    def learn(self,):
        tot=len(self.vt)
        index=np.random.choice(tot,BATCH_SIZE)
#         print(index)
#         print(len(self.s),len(self.a),len(self.vt))
        s,a,vt=[],[],[]
        for i in range(BATCH_SIZE):
            s.append(self.s[int(index[i])])
            a.append(self.a[int(index[i])])
            vt.append(self.vt[int(index[i])])
        s=self.s[-BATCH_SIZE]
        s=self.s[-BATCH_SIZE]
        s=self.s[-BATCH_SIZE]
        
        s=torch.stack(s,dim=0)
        s=Variable(s)
        p=self.net(s)
        loss=0
        for i in range(BATCH_SIZE):
            if(a[i]==1):
                loss+=-(1-torch.log(p[i]))*vt[i]
            else:
                loss+=-torch.log(p[i])*vt[i]
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
    
env = gym.make("Pong-v0") 
env = env.unwrapped
pg=PG()
#pg.net.load_state_dict(torch.load('pg_net3.pkl'))
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
    while(not done):
        x=s-pre_s
        pre_s=s
        a=pg.choose(x)
        a_=a+2
        s_,r,done,info=env.step(a_)
        reward+=r
        #pg.store(s,a,r)
        s=prepare(s_)
        if(RENDER and epoch%10==0):
            env.render()
    ep_r=ep_r*0.99+reward*0.01
    env.close()
    if(epoch%100==0):
        torch.save(pg.net.state_dict(), 'pg_net3.pkl')
    print('epoch %d | reward %d | ep_r %.4f'%(epoch,reward,ep_r))
    epoch+=1
    
    
    