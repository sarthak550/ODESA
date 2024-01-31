import numpy as np
import math

class FullyConnected:
    def __init__(self, context_rows, context_cols,n_neurons,eta, threshold_open,tau,trace_tau,thresh_eta,cumulative_ts=False):

        self.rows = context_rows
        self.cols = context_cols
        self.n_neurons = n_neurons
        self.eta = eta
        self.threshold_open = threshold_open
        self.tau = tau
        self.cumulative_ts = cumulative_ts
        self.trace_tau = trace_tau
        self.thresh_eta = thresh_eta
        self.w = np.random.rand(self.n_neurons,self.rows*self.cols)
        self.w = np.divide(self.w,np.linalg.norm(self.w,axis=1,keepdims=True))
        self.traceRecencyThreshold = 0.1

        self.delta = np.zeros_like(self.w)
        self.thresh = np.random.rand(self.n_neurons)
        self.delta_thresh = np.zeros(self.n_neurons)
        self.timestamps = np.zeros((self.rows,self.cols),dtype=np.int64)
        self.winnerTrace = np.zeros(self.n_neurons,dtype=np.int64)
        self.winnerMV = np.zeros(self.n_neurons)
        self.polarity = np.zeros((self.rows,self.cols))
        self.noWinnerTrace = -np.inf

    def initialize(self):
        dim_per_neuron = math.floor((self.rows * self.cols)/self.n_neurons)
        if dim_per_neuron == 0:
            dim_per_neuron = 1
        self.w = self.w*0
        for neuron in range(self.n_neurons):
            random_dims = np.random.choice(np.arange(0,self.rows*self.cols),size=dim_per_neuron)
            self.w[neuron,random_dims] = 1
            self.w[neuron] /= np.linalg.norm(self.w[neuron])



    
    def reset_time(self):
        # self.timestamps
        self.timestamps = self.timestamps - np.inf
        self.winnerTrace = self.winnerTrace - np.inf
        self.noWinnerTrace = -np.inf
        self.polarity = self.polarity*0 # = torch.zeros(self.rows,self.cols,device=self.device)
        self.delta = self.delta*0 # = torch.zeros_like(self.w,device=self.device)
        self.winnerMV = self.winnerMV*0
        self.delta_thresh = self.delta_thresh*0

    def ingest(self,x,y,ts,p):
        self.timestamps[x,y] = ts
        self.polarity[x,y] = p

    def get_timesurface(self,ts):
        return np.multiply(self.polarity,np.exp((self.timestamps-ts)/self.tau))

    def add_event(self,x,y,ts,p):
        if not  (x <0 or y < 0):
            # return -1, -1
            if self.cumulative_ts:
                self.polarity[x,y] = self.polarity[x,y]*np.exp((self.timestamps[x,y]-ts)/self.tau) + p
                self.timestamps[x,y] = ts
            else:
                self.timestamps[x,y] = ts
                self.polarity[x,y] = p
        

    def add_trace_event(self,winner,ts):

        if self.cumulative_ts:
            self.winnerMV[winner] = self.winnerMV[winner]*np.exp((self.winnerTrace[winner]-ts)/self.trace_tau) + 1
            self.winnerTrace[winner] = ts
        else:
            self.winnerMV[winner] =  1
            self.winnerTrace[winner] = ts
        

    def record(self,ts):
        # trace_ts = ts - self.winnerTrace <=self.trace_tau
        trace_ts = np.multiply(self.winnerMV,np.exp((self.winnerTrace-ts)/self.trace_tau))
        # trace_ts = torch.exp((self.winnerTrace-ts)/self.trace_tau)
        neuronsToReward = trace_ts >= self.traceRecencyThreshold
        self.w[neuronsToReward] = self.w[neuronsToReward] + self.eta*(self.delta[neuronsToReward]-self.w[neuronsToReward])
        self.w[neuronsToReward] = np.divide(self.w[neuronsToReward],np.linalg.norm(self.w[neuronsToReward],axis=1,keepdims=True))
        updated_thresh = (1-self.thresh_eta)*self.thresh[neuronsToReward] + self.thresh_eta*self.delta_thresh[neuronsToReward]
        updated_thresh[updated_thresh<0] = 0
        self.thresh[neuronsToReward] = updated_thresh
        if np.exp((self.noWinnerTrace-ts)/self.trace_tau) >= self.traceRecencyThreshold:
            neuronsToPunish = np.logical_not(neuronsToReward)
            self.thresh[neuronsToPunish] -= self.threshold_open


    def reward(self,neuron):
        self.w[neuron] = self.w[neuron] + self.eta*(self.delta[neuron]-self.w[neuron])
        self.w[neuron] /= np.linalg.norm(self.w[neuron])
        self.thresh[neuron] = (1-self.thresh_eta)*self.thresh[neuron] + self.thresh_eta*self.delta_thresh[neuron]  


    def forward(self,x,y,ts,p):
        '''
        Returns (winnerNeuron, winningDotDistance)
        '''
        if x <0 or y < 0:
            return -1
        

        self.add_event(x,y,ts,p)

        event_context = np.multiply(self.polarity,np.exp((self.timestamps-ts)/self.tau))
        context_norm = np.linalg.norm(event_context)
        if context_norm == 0:
            
            return -1
        event_context_unit = event_context/context_norm

        dotProduct = np.matmul(self.w,event_context_unit.reshape(-1,1))

        

        if np.all(np.less(dotProduct,self.thresh.reshape(dotProduct.shape))):
            winnerNeuron = -1
            self.noWinnerTrace = ts
        else:

            dotProduct[dotProduct < self.thresh.reshape(dotProduct.shape)] = - np.inf
            winnerNeuron = np.argmax(dotProduct)
            self.delta[winnerNeuron] = event_context_unit.reshape(1,-1)
            self.delta_thresh[winnerNeuron] = dotProduct[winnerNeuron]
            self.add_trace_event(winnerNeuron,ts)



        return winnerNeuron

          

    def punish_single(self,neuron):
        self.thresh[neuron] = self.thresh[neuron] - self.threshold_open

    def punish(self,ts):
        self.thresh = self.thresh - self.threshold_open


  

    def save_weights(self,filename):
        np.save(filename,self.w)

    def save_thresh(self,filename):
        np.save(filename,self.thresh)

    def load_weights(self,filename):
        self.w = np.load(filename)

    def load_thresh(self,filename):
        self.thresh = np.load(filename)
