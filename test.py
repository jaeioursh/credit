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
        self.optimizer.step()
        return loss.detach().item()
    
    def alignment_loss(self,o, t):
        ot=torch.transpose(o,0,1)
        tt=torch.transpose(t,0,1)

        O=o-ot
        T=t-tt

        align = torch.mul(O,T)
        #print(align)
        align = self.sig(align)
        loss = -torch.mean(align)
        return loss
    
net=Net()
x=np.array([[0,0],[1,0],[0,1],[1,1]])
y=np.array([[0],[1],[1],[0]])

x=np.vstack([x for _ in range (2)])
y=np.vstack([y for _ in range (2)])
print(x.shape,y.shape)
for i in range(10000):
    print(i,net.train(x,y))
print(net.feed(x))