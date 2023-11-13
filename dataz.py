#alignment/vis
from mtl import make_env

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import pickle

from teaming import logger
from teaming.learnmtl import Net

idx=0
q,i,AGENTS=[[1,10,4],[1,2,6],[1,3,8]][idx]
Q=q
GEN=1
RESOLUTION,SAMPLES=100,10

ROBOTS=AGENTS

fname="save/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl"

log = logger.logger()
log.load(fname)
env=make_env(ROBOTS)
pos=log.pull("position")
print(np.array(pos).shape)


teams=np.array(log.pull("types")[0])
INDEX=1





#env.reset()
def eval(x,y,t,q):
    
    test=pos[-1][0].copy()

    test[q]=[x,y]

    env.data["Agent Orientations"][q,:]=[np.sin(t),np.cos(t)]
    env.data["Agent Positions"]=test
    env.data["Agent Position History"]=np.array([env.data["Agent Positions"]])
    env.data["Steps"]=0
    env.data["Observation Function"](env.data)
    z=env.data["Agent Observations"]
    s=z[q]

    #reward.assignGlobalReward(env.data)
    #reward.assignDifferenceReward(env.data)
    
    env.data["Reward Function"](env.data)
    g=env.data["Global Reward"]
    #print(g)
    #g=0.0
    r=0.0
    #r=net[q].feed(s)[0]

    #print(env.data["Agent Orientations"])
    return s,g

def eval2(steps,N,q):
    x=np.linspace(-5,35,steps)
    y=x.copy()
    S=[]
    G=[]
    for j in range(steps):
        for i in range(steps):
            for k in range(N):
                t=np.random.random()*2*np.pi
                s,g=eval(x[i],y[j],t,q)
                S.append(s)
                G.append(g)
    print("network")
    S=np.array(S)
    G=np.array(G)
    print(S.shape,G.shape)
    with open("save/testz"+str(AGENTS)+str(Q)+".data", 'wb') as f:
        pickle.dump([S,G], f)

RES=100
N=10
if 0:
    eval2(RES,N,0)

with open("save/testz"+str(AGENTS)+str(Q)+".data", 'rb') as f:
    S,G=pickle.load( f)

print(S[0],S[1])

L=[]
lr=0.001
hidden=80
opti=0
acti=1
net=Net(hidden,lr,2,opti,acti)
BATCH=32
G=np.array([G]).T
print(S.shape,G.shape)
for i in range(100):
    print(i)
    ll=[]
    for j in range(len(S)//BATCH):
        idx=np.random.randint(0,len(S),20)
        s=S[idx,:]
        g=G[idx]
        
        l=net.train(s,g)
        ll.append(l)
    L.append(np.mean(ll))
print(max(G),min(G),np.mean(G))
zz=net.feed(S)
zz=np.reshape(zz,(RES,RES,N))
zz=np.mean(zz,axis=2)
gg=np.reshape(G,(RES,RES,N))
gg=np.mean(gg,axis=2)
plt.subplot(131)
plt.plot(L)
plt.subplot(132)
plt.imshow(zz)
plt.subplot(133)
plt.imshow(gg)
plt.show()
    


    