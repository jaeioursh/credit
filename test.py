import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cpu") 
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

#output,target

class Net():
    def __init__(self,hidden=20):
        learning_rate=5e-3
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,1),
            torch.nn.Tanh()
        )
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.loss_fn = self.alignment_loss
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.sig = torch.nn.Sigmoid()
    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.model(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y,n=5,verb=0):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.model(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        #print([s.grad.shape for s in self.model.parameters()])
        self.optimizer.step()
        return loss.detach().item()
    
    def alignment_loss(self,o, t):
        o=o+t
        ot=torch.transpose(o,0,1)
        tt=torch.transpose(t,0,1)

        O=o-ot
        T=t-tt

        align = torch.mul(O,T)
        #print(align)
        align = self.sig(align)
        loss = -torch.mean(align)
        return loss
    
if __name__ == "__main__":
    if 0:
        net=Net()
        x=np.array([[0,0],[1,0],[0,1],[1,1]])
        y=np.array([[0],[1],[1],[0]])

        x=np.vstack([x for _ in range (1)])
        y=np.vstack([y for _ in range (1)])
        print(x.shape,y.shape)
        for i in range(1000):
            print(i,net.train(x,y))
        print(net.feed(x))
    if 0:
        nagents=4
        t=np.linspace(0,2*np.pi,nagents,endpoint=False)
        pos = np.array([np.sin(t),np.cos(t)]).T
        pos=20*pos+15
        print(pos)
        plt.plot(pos[:,0],pos[:,1],"o")
        plt.show()
    if 1:
        from mtl import round_env
        
        T=np.linspace(0,50,1000)
        G=[]
        i=0
        for t in T:
            i+=1
            #t=0.0
            env=round_env(1)
            #print(env.data["Agent Positions"])
            env.data["Agent Positions"]=np.array([[t,15.0]])
            #print(env.data["Agent Positions"])
            env.data["Agent Position History"]=np.array([env.data["Agent Positions"]])
            env.data["Steps"]=0
            env.data["Observation Function"](env.data)
            z=env.data["Agent Observations"]
    
            
            env.data["Reward Function"](env.data)
            g=env.data["Global Reward"]
            if g>1e3 or g<-1e3:
                print("err")
                print(i,g)
            G.append(g)
        plt.plot(T,G)
        plt.show()