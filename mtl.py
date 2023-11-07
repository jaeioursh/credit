"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
from os import killpg
import numpy as np
import sys
import multiprocessing as mp


from rover_domain_core_gym import RoverDomainGym
import code.ccea_2 as ccea
import code.agent_domain_2 as domain

#import mods
from teaming.learnmtl import learner
from sys import argv
import pickle
#import tensorflow as tf
import time

def rand_loc(n):
    x,y=np.random.random(2)
    pos=[[x,y]]
    while len(pos)<6:
        X,Y=np.random.random(2)
        for x,y in pos:
            dist=((X-x)**2.0+(Y-y)**2.0 )**0.5
            if dist<0.2:
                X=None 
                break
        if not X is None: 
            pos.append([X,Y])
    
    return np.array(pos)


#pri alignment multiagent tumernt(vals)
def make_env(nagents,rand=0):
    vals =np.array([0.8,1.0,0.6,0.3,0.2,0.1])
    
    if rand:
        pos=np.array([
            [0.0, 0.2],
            [0.7, 0.1],
            [1.0, 0.3],
            [0.4, 0.6],
            [0.3, 0.3],
            [0.4, 0.9]
        ])
    else:
        pos=np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [1.0, 0.0]
        ])
    
    #pos=rand_loc(6)#np.random.random((6,2))
    #vals=np.random.random(6)/2.0
    print(vals)

    sim = RoverDomainGym(nagents,30,pos,vals)
 


    sim.data["Coupling"]=2
    sim.data['Number of Agents']=nagents

    obs=sim.reset()
    return sim


def round_env(nagents,rand=0):
    vals =np.array([1.0]*nagents)
    t=np.linspace(0,2*np.pi,nagents,endpoint=False)
    
    pos = np.array([np.cos(t),np.sin(t)]).T
    pos=0.9*pos+0.5
    #print(pos)
    sim = RoverDomainGym(nagents,30,pos,vals)
 
    sim.data["Coupling"]=1
    sim.data['Number of Agents']=nagents
    sim.data["Minimum Distance"]=1.2
    sim.data["Observation Radius"]=20.0
    obs=sim.reset()
    return sim



def test1(trial,k,n,train_flag,n_teams,save=1,params=None):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    env=make_env(n)
    if params is None:
        params=[5e-3, 80, 32,1000]
    OBS=env.reset()

    controller = learner(n,k,env,params)
    #controller.set_teams(n_teams)
    R=[]
    for i in range(2001):

        
        #controller.randomize()
        if i%100000==0:
            controller.set_teams(n_teams)

        if i%1==0:
            controller.test(env)

        r=controller.run(env,train_flag)
        if save:
            print(i,r,len(controller.team),train_flag)
        R.append(r)
            
        if i%50==0:
            #controller.save("tests/q"+str(frq)+"-"+str(trial)+".pkl")
            #controller.save("logs/"+str(trial)+"r"+str(16)+".pkl")
            #controller.save("tests/jj"+str(121)+"-"+str(trial)+".pkl")
            #controller.log.clear("hist")
            #controller.put("hist",controller.hist)
            if save:
                controller.save("save/"+str(k)+"-"+str(n)+"-"+str(trial)+"-"+str(train_flag)+".pkl")
    return -max(R[-20:])




def test2(trial,k,n,train_flag,n_teams,save=1,params=None):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    env=round_env(n)
    if params is None:
        params=[5e-3, 80, 32,1000]
    OBS=env.reset()
    controller = learner(n,k,env,params)
    #controller.set_teams(n_teams)
    R=[]
    for i in range(1201):
        if i%100000==0:
            controller.set_teams(n_teams)

        controller.test(env)

        r=controller.run(env,train_flag)
        if save:
            print(i,r,len(controller.team),train_flag)
        R.append(r)
            
        if i%50==0:
            if save:
                controller.save("save/r"+str(k)+"-"+str(n)+"-"+str(trial)+"-"+str(train_flag)+".pkl")
    return -max(R[-20:])

    #train_flag=0 - align w/ shape
    #train_flag=1 - alignment network
    #train_flag=2 - counterfactual-aprx
    #train_flag=3 - fitness critic
    #train_flag=4 - D*
    #train_flag=5 - G*
    #train_flag=6 - a shape train traj
    #train_flag=7 - align train traj
    #train_flag=8 - a shape train traj max
    #train_flag=9 - align train traj max
if __name__=="__main__":
    if 0:
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()
        # ... do something ...
        test1(42,5,4,1)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        
    else:
        for train in [1,7,9]:
            procs=[]
            for k in [4,6,8]:
                n=k
                teams=100
                params = [5e-3, 64, 32  ,1000,0,0]
                for i in range(12):
                    p=mp.Process(target=test1,args=(i,k,n,train,teams,1,params))
                    p.start()
                    time.sleep(0.05)
                    procs.append(p)
                    #p.join()
                for p in procs:
                    p.join()

# 100 - static
# 200 - minimax single
# 300 random
# 400 most similar