from mtl import test1
from skopt import gp_minimize
import multiprocessing as mp
import pickle as pkl

test4=lambda x: test1(0,4,4,1,100,save=0,params=x)
test6=lambda x: test1(0,6,6,1,100,save=0,params=x)
test8=lambda x: test1(0,8,8,1,100,save=0,params=x)

def opt(test,num,idx):
    C=[(0.001, 0.1),(3.0,120.0),(4.0,64.0),(100.0,10000.0)]
    res = gp_minimize(test, C, n_calls=41)#,acq_func="PI")
    print(res.x)
    print(res.fun)
    with open("save/"+str(num)+"-"+str(idx)+".pkl","wb") as f:
        data=[res.x_iters,res.func_vals]
        print(data)
        pkl.dump(data,f)

procs=[]
for test,num in [[test4,4],[test6,6],[test8,8]]:
    for idx in range(3):
        p=mp.Process(target=opt,args=(test,num,idx))
        p.start()
        procs.append(p)
for p in procs:
    p.join()