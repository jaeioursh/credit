import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
device = torch.device("cpu") 
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
#attention for weighting rewards
#output,target

#attend to 

class Net():
    def __init__(self,input_dim=2,hidden=20,out_dim=1):
        learning_rate=5e-3
        
        self.dim = [input_dim,hidden,out_dim]
        self.query = nn.Linear(input_dim, hidden)
        self.key = nn.Linear(input_dim, hidden)
        self.value = nn.Linear(input_dim, hidden)
        self.softmax = nn.Softmax(dim=2)

        p=[self.query,self.key,self.value]

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        #self.loss_fn = self.alignment_loss
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.Adam(self.p, lr=learning_rate)
        
    def internal_feed(self,x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.dim[1] ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
    
    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.internal_feed(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y,n=5,verb=0):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.internal_feed(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        #print([s.grad.shape for s in self.model.parameters()])
        self.optimizer.step()
        return loss.detach().item()
    
    def alignment_loss(self,g, G):
        gt=torch.transpose(g,0,1)
        Gt=torch.transpose(G,0,1)
        align = torch.mul(g-gt,G-Gt)
        align = self.sig(align)
        loss = -torch.mean(align)
        return loss
    
if __name__ == "__main__":
    
    net=Net()
    x=np.array([[[0],[0]],[[1],[0]],[[0],[1]],[[1],[1]]])
    y=np.array([[[0]],[[1]],[[1]],[[0]]])

    #x=np.vstack([x for _ in range (1)])
    #y=np.vstack([y for _ in range (1)])
    print(x.shape,y.shape)
    print(net.feed(x))
    for i in range(1000):
        print(i,net.train(x,y))
    print(net.feed(x))

    