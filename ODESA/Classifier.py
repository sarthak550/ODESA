import math
import numpy as np



class Classifier:
    def __init__(self, context_rows, context_cols,n_neurons_per_class, n_classes, eta,tau, threshold_open, thresh_eta=0.01,cumulative_ts=False):
        
        self.rows = context_rows
        self.cols = context_cols

#         self.n_neurons_per_class = n_neurons_per_class
        self.n_classes = n_classes
        if isinstance(n_neurons_per_class, list):
            self.n_neurons_per_class = n_neurons_per_class
        else:
            self.n_neurons_per_class = [n_neurons_per_class for i in range(n_classes)]
        self.neuron_to_class = dict()
        self.class_to_neuron = dict()
        neuron_id = 0
        for class_id,n in enumerate(self.n_neurons_per_class):
            self.class_to_neuron[class_id] = list(range(neuron_id, neuron_id+n))
            for idx in range(neuron_id,neuron_id+n):
                self.neuron_to_class[idx] = class_id
            neuron_id = neuron_id+n 
        self.n_neurons = len(self.neuron_to_class.keys())
        self.eta = eta
        self.threshold_open = threshold_open
        self.thresh_eta = thresh_eta
        self.cumulative_ts = cumulative_ts
        self.tau = tau
        self.thresh_eta = thresh_eta

        self.w = np.random.rand(self.n_neurons,self.rows*self.cols)
        self.w = np.divide(self.w,np.linalg.norm(self.w,axis=1,keepdims=True))

        self.delta = np.zeros_like(self.w)
        self.delta_thresh = np.zeros(self.n_neurons)


        self.thresh = np.random.rand(self.n_neurons)
        

        self.latest_context = None
        self.timestamps = np.zeros((self.rows,self.cols),dtype=np.int64)
        self.winnerTrace = np.zeros(self.n_neurons,dtype=np.int64)
        self.winnerMV = np.zeros(self.n_neurons)
        self.polarity = np.zeros((self.rows,self.cols))

    def initialize(self):
        dim_per_neuron = math.floor((self.rows * self.cols)/self.n_neurons)
        if dim_per_neuron == 0:
            dim_per_neuron = 1
        self.w = self.w*0
        for neuron in range(self.n_neurons):
            random_dims = np.random.choice(np.arange(0,self.rows*self.cols),size=dim_per_neuron)
            self.w[neuron,random_dims] = 1
            self.w[neuron] /= np.linalg.norm(self.w[neuron])

    def recently_active(self,ts):
        recent_trace = np.exp((self.winnerTrace-ts)/self.tau)
        recent_neuron = np.argmax(recent_trace)

        return recent_trace[recent_neuron] > 0.1

    
    def reset_time(self):
        self.timestamps = self.timestamps - np.inf
        self.winnerTrace = self.winnerTrace - np.inf
        self.polarity = self.polarity*0 # = torch.zeros(self.rows,self.cols,device=self.device)
        self.delta = self.delta*0 # = torch.zeros_like(self.w,device=self.device)
        self.winnerMV = self.winnerMV*0
        self.delta_thresh = self.delta_thresh*0


    def add_event(self,x,y,ts,p):
        self.polarity[x,y] = self.polarity[x,y]*np.exp((self.timestamps[x,y]-ts)/self.tau) + p
        self.timestamps[x,y] = ts

    def forward_context(self,event_context):
        '''
        TODO: This should be deprecated
        Returns (winnerNeuron,winnerClass)
        '''

        # event_context = torch.mul(self.polarity,torch.exp((self.timestamps-ts)/self.tau))
        event_context_norm = 1
        event_context_unit = event_context/event_context_norm
        dotProduct = np.matmul(self.w,event_context_unit.reshape(-1,1))
        if np.all(np.less(dotProduct,self.thresh.reshape(dotProduct.shape))):
            winnerNeuron = -1
            winnerClass = -1
        else:
            dotProduct[dotProduct < self.thresh.reshape(dotProduct.shape)] = - np.inf
            winnerNeuron = np.argmax(dotProduct)
            self.delta[winnerNeuron] = event_context.reshape(1,-1) - self.w[winnerNeuron]
            self.delta_thresh[winnerNeuron] = dotProduct[winnerNeuron]
            # self.winnerTrace[winnerNeuron] = ts
            winnerClass = self.neuron_to_class[winnerNeuron]

        return winnerNeuron,winnerClass

    def get_timesurface(self,ts):
        return np.multiply(self.polarity,np.exp((self.timestamps-ts)/self.tau))

    def forward(self,x,y,ts,p):
        '''
        Returns (winnerNeuron,winnerClass)
        '''
        if y < 0 or x < 0:
            return -1, -1
        else:
            if self.cumulative_ts:
                self.add_event(x,y,ts,p)
            else:
                self.timestamps[x,y] = ts
                self.polarity[x,y] = p
        event_context = np.multiply(self.polarity,np.exp((self.timestamps-ts)/self.tau))
        event_context_norm = np.linalg.norm(event_context)
        event_context_unit = event_context/event_context_norm

        dotProduct = np.matmul(self.w,event_context_unit.reshape(-1,1))
        self.latest_context = event_context_unit.reshape(1,-1)
        if np.all(np.less(dotProduct,self.thresh.reshape(dotProduct.shape))):
            winnerNeuron = -1
            winnerClass = -1
        else:
            dotProduct[dotProduct < self.thresh.reshape(dotProduct.shape)] = - np.inf
            winnerNeuron = np.argmax(dotProduct)
            
            self.delta[winnerNeuron] = event_context_unit.reshape(1,-1) - self.w[winnerNeuron]
            self.delta_thresh[winnerNeuron] = dotProduct[winnerNeuron]
            self.winnerTrace[winnerNeuron] = ts
            winnerClass = self.neuron_to_class[winnerNeuron]

        return winnerNeuron,winnerClass



            
    def reward(self,winnerNeuron):

        self.w[winnerNeuron] = self.w[winnerNeuron] + self.eta*self.delta[winnerNeuron]
        self.w[winnerNeuron] /= np.linalg.norm(self.w[winnerNeuron])
        updated_thresh = (1-self.thresh_eta)*self.thresh[winnerNeuron] + self.thresh_eta*self.delta_thresh[winnerNeuron]
        if updated_thresh < 0:
            updated_thresh = 0
        self.thresh[winnerNeuron] = updated_thresh


    def punish(self,winnerNeuron,label):
        # If wrong class fired. Punish by doing opposite of weight update. (Anti-STDP)
        if winnerNeuron > -1:
            self.w[winnerNeuron] = self.w[winnerNeuron] - self.eta*self.delta[winnerNeuron]
            self.w[winnerNeuron] /= np.linalg.norm(self.w[winnerNeuron])
        
        # Open the thresholds for correct class
        neurons = self.class_to_neuron[label]
        self.thresh[neurons] -= self.threshold_open

        # Find the closest winner in correct class group and reward it.
        dotProduct = np.matmul(self.w[neurons],self.latest_context.T)
        argmaxID = np.argmax(dotProduct)
        closestNeuron = neurons[argmaxID]
        self.delta[closestNeuron] = self.latest_context - self.w[closestNeuron]
        self.delta_thresh[closestNeuron] = dotProduct[argmaxID]
        self.reward(closestNeuron)


    def save_weights(self,filename):
        np.save(filename,self.w)

    def save_thresh(self,filename):
        np.save(filename,self.thresh)

    def load_weights(self,filename):
        self.w = np.load(filename)

    def load_thresh(self,filename):
        self.thresh = np.load(filename)
