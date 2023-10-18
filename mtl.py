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


#print(vals)
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


import time

def test1(trial,k,n,train_flag,n_teams):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    env=make_env(n)
 
    OBS=env.reset()

    controller = learner(n,k,env)
    #controller.set_teams(n_teams)

    for i in range(4001):

        
        #controller.randomize()
        if i%100000==0:
            controller.set_teams(n_teams)

        if i%1==0:
            controller.test(env)

        r=controller.run(env,train_flag)# i%100 == -10)
        print(i,r,len(controller.team),train_flag)
        
            
        if i%50==0:
            #controller.save("tests/q"+str(frq)+"-"+str(trial)+".pkl")
            #controller.save("logs/"+str(trial)+"r"+str(16)+".pkl")
            #controller.save("tests/jj"+str(121)+"-"+str(trial)+".pkl")
            #controller.log.clear("hist")
            #controller.put("hist",controller.hist)
            controller.save("tests/very/"+str(k)+"-"+str(n)+"-"+str(trial)+"-"+str(train_flag)+".pkl")

    #train_flag=0 - D
    #train_flag=1 - Neural Net Approx of D
    #train_flag=2 - counterfactual-aprx
    #train_flag=3 - fitness critic
    #train_flag=4 - D*
    #train_flag=5 - G*
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
        for train in [3,4,5]:
            procs=[]
            k=5
            n=4
            for k,n in [[7,4]]:
                teams=100
                for i in range(12,18):
                    if train==1 or train==3:
                        i-=12
                    p=mp.Process(target=test1,args=(i,k,n,train,teams))
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