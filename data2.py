#alignment/vis
from mtl import make_env

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import pickle

from teaming import logger
from teaming.learnmtl import Net

idx=2
#q,i,AGENTS=[[1,10,4],[1,2,6],[1,3,8]][idx]
q,i,AGENTS=[[0,10,4],[0,4,6],[0,2,8]][idx]
Q=q
GEN=1
RESOLUTION,SAMPLES=100,50

ROBOTS=AGENTS

fname="save/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl"

log = logger.logger()
log.load(fname)
env=make_env(ROBOTS)
pos=log.pull("position")
print(np.array(pos).shape)


teams=np.array(log.pull("types")[0])
INDEX=1


net=[Net(80,1,1) for i in range(AGENTS)]
for i in range(AGENTS):
    net[i].model.load_state_dict(torch.load(fname+"a.mdl")[i])

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
    #g=0.0
    r=0.0
    #r=net[q].feed(s)[0]

    #print(env.data["Agent Orientations"])
    return s,g

def eval2(steps,N,q):
    x=np.linspace(-5,35,steps)
    y=x.copy()
    S=[]
    for j in range(steps):
        for i in range(steps):
            for k in range(N):
                t=np.random.random()*2*np.pi
                s=eval(x[i],y[j],t,q)[0]
                S.append(s)
    print("network")
    S=np.array(S)
    zz=net[q].feed(S)

    zz=np.reshape(zz,(steps,steps,N))
    zz=np.mean(zz,axis=2)
    return zz

As=[i for i in range(AGENTS)]
print(As)
VALS=[0.8,1.0,0.6,0.3,0.2,0.1]
poi=log.pull("poi")[0]
txt=[str(i) for i in VALS]

CMAP=matplotlib.colormaps["plasma"]
CMAP=matplotlib.colormaps["Greys"]
CMAP=matplotlib.colormaps["seismic"]
if GEN:
    data=[]
    for i in range(len(As)):
        print(i)
        zz=eval2(RESOLUTION,SAMPLES,As[i])
        data.append(zz)
    with open("save/test"+str(AGENTS)+str(Q)+".data", 'wb') as f:
        pickle.dump(data, f)
else:   
    with open("save/test"+str(AGENTS)+str(Q)+".data", 'rb') as f:
        data=pickle.load(f)

for i in range(len(As)):
    zz=data[i]

    ext=[-5,35,-5,35]
    plt.subplot(2,AGENTS//2+1,i+1)
    plt.title("Agent "+str(As[i]))#+", Agent "+str(INDEX))
    ax=plt.gca()
    
    #zz[zz<0.3]=0.3
    im=ax.imshow(np.flipud(zz)/20,extent=ext,cmap=CMAP)
    lbl="Critic Value"
    

    for t in range(len(txt)):
        plt.text(poi[t,0]-3,poi[t,1]+2,txt[t],c='#ff0000')
    plt.scatter(poi[:,0],poi[:,1],s=50,c='#ff0000',marker="v",zorder=10000)
    plt.xlim([-5,35])
    plt.ylim([-5,35])
    plt.axis('off')

cax = plt.gcf().add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, cax=cax)

plt.subplot(2,AGENTS//2+1,i+3)


VALS=[0.8,1.0,0.6,0.3,0.2,0.1]
 
txt=[str(i) for i in VALS]
vals=np.array(VALS)*0+1000


for i in range(AGENTS):
    data=[]
    for j in range(len(pos)):
        #print(np.array(pos).shape)
        p=pos[j][0][i]
        data.append(p)
    
    mkr='*'
    mkr=[".",",","*","v","^","<",">","1","2","3","4","8"][i]
    mkr="$"+str(i)+"$"
    #clr="k"
    x,y=np.array(data).T
    plt.plot(x,y,color='k',marker=mkr,linewidth=1.0,linestyle=":")

lgnd=["Agent "+str(i) for i in range(AGENTS)]
    
#print(lgnd)

#plt.legend(lgnd)
for i in range(len(txt)):
    plt.text(poi[i,0]+1,poi[i,1]+1,txt[i])

plt.scatter(poi[:,0],poi[:,1],s=100,c='#0000ff',marker="v",zorder=10000)
plt.xlim([-5,35])
plt.ylim([-5,35])


plt.subplots_adjust(left=0.05, bottom=0.05, right=.87, top=.95, wspace=0.05, hspace=0.05)


plt.show()