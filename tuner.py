from mtl import test1
from skopt import gp_minimize
import multiprocessing as mp
import pickle as pkl

test4=lambda x: sum([test1(0,4,4,1,100,save=0,params=x) for i in range(3)])
test6=lambda x: sum([test1(0,6,6,1,100,save=0,params=x) for i in range(3)])
test8=lambda x: sum([test1(0,8,8,1,100,save=0,params=x) for i in range(3)])
test4a=lambda x: sum([test1(0,4,4,0,100,save=0,params=x) for i in range(3)])
test6a=lambda x: sum([test1(0,6,6,0,100,save=0,params=x) for i in range(3)])
test8a=lambda x: sum([test1(0,8,8,0,100,save=0,params=x) for i in range(3)])

def opt(test,num,typ,idx):
    C=[(0.0001, 0.001),(3.0,120.0),(4.0,64.0),(100.0,100000.0)]
    res = gp_minimize(test, C, n_calls=50)#,acq_func="PI")
    print(res.x)
    print(res.fun)
    with open("save/b"+str(num)+"-"+str(idx)+"-"+str(typ)+".pkl","wb") as f:
        data=[res.x_iters,res.func_vals]
        print(data)
        pkl.dump(data,f)

procs=[]
for test,num,typ in [[test4,4,1],[test6,6,1],[test8,8,1],[test4,4,0],[test6,6,0],[test8,8,0]]:
    for idx in range(2):
        p=mp.Process(target=opt,args=(test,num,typ,idx))
        p.start()
        procs.append(p)
for p in procs:
    p.join()