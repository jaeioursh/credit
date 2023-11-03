from urllib.request import CacheFTPHandler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from math import comb
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('tableau-colorblind10')
#matplotlib.rcParams['text.usetex'] = True
from teaming import logger
DEBUG=0

#schedule = ["evo"+num,"base"+num,"EVO"+num]
#schedule = ["base"+num+"_"+str(q) for q in [0.0,0.25,0.5,0.75,1.0]]
AGENTS=10
ROBOTS=AGENTS
vals=sorted([0.8,1.0,0.6,0.3,0.2,0.1],reverse=True)
lbls={0:"Al w/ shaping",1:"Alignment",2:"Ctrfctl. Aprx.",3:"FitCritic",4:"$D$",5:"$G$",6:"al shape traj",7:"al traj",8:"al shape traj max",9:"al traj max"}
if DEBUG:
    plt.subplot(1,2,1)
mint=1e9

for q in [0,1,3,4,5,6,7,8,9]:
    T=[]
    R=[]
    print(q)

    mint=10000
    for i in range(12):
        log = logger.logger()
        
        try:
            log.load("save/r"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl")
        except:
            continue
    
        r=log.pull("reward")
        #L=log.pull("loss")
        t=log.pull("test")
        n_teams=len(t[0])
        aprx=log.pull("aprx")
        
        #print(t)
        r=np.array(r)

        t=np.array(t)
        
        mint=min(len(t),mint)
        
        print(np.round(t[-1,:],2))
        t=np.sum(t,axis=1)
        print(len(t),t.shape)
        if DEBUG:
            plt.plot(t)
        R.append(r)
        print(q,i,t[-1])
        T.append(t)
    if DEBUG:
        plt.subplot(1,2,2)

    print("broken1")
    print(T,mint)
    T=[t[:mint] for t in T]
    print("broken2")
    print(T)
    std=np.std(T,axis=0)/np.sqrt(12)
    Tm=np.mean(T,axis=0)
    print("broken3")
    print(T,std,Tm)
    X=[i*1 for i in range(len(Tm))]

    plt.plot(X,Tm,label=lbls[q])
    #plt.fill_between(X,Tm-std,Tm+std,alpha=0.35, label='_nolegend_')

    #plt.ylim([0,1.15])
    plt.grid(True)
plt.legend([str(i) for i in range(8)])
max_val=sum(vals[:ROBOTS//2])*n_teams

plt.xlabel("Generation")
plt.title("Performance of " + str(AGENTS)+" Agents, Teams of "+str(ROBOTS))
leg=plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)

plt.ylabel("G")
print(len(T))
plt.plot([0,X[-1]],[max_val,max_val],"--",label="Max Score")

plt.tight_layout()
#plt.savefig("figsv3/vis8_"+str(ROBOTS)+"_"+str(AGENTS)+".png")
plt.show()