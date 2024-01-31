
class FeastConvModel:
    def __init__(self):
        self.hidden_layers = []
        self.output_layer = None
        
    def add_hidden_layer(self, hidden_layer):
        '''
        Add a hidden layer to the model. It should be of class ConvHidden
        '''
        # assert(self.hidden_layers[-1])
        self.hidden_layers.append(hidden_layer)

    def add_output_layer(self, output_layer):
        '''
        Add an output layer to the model. It should be of class ConvClassifier. This has to be called before doing any other work on the model. 
        '''
        self.output_layer = output_layer

    def reset(self):
        '''
        Reset the timesurfaces of the layers in the model.  
        '''
        for layer in self.hidden_layers:
            layer.reset_time()
        self.output_layer.reset_time()

    def forward(self,event,label):
        '''
        Forward an event and it's label to the model. It does a forward pass and then update weights and thresholds accordingly based on the label. 
        attr: event : (x,y,z,ts)
        attr: label: should be an integer >= -1. -1 stands for no label. Anything above -1 considered as a class label. 

        returns: list of hidden layer winners and output winner, predicted class. Values are -1 whenver there is no output/winner from a layer. 
        '''
        x, y, z,ts = event
        # List of winners from each layer
        winners = [(-1,-1,-1,ts) for i in self.hidden_layers]

        # Forward pass of the event and collect the winners for each hidden layer. The winner is 
        for layer_idx, layer in enumerate(self.hidden_layers):
            if layer_idx == 0:
                winners[layer_idx] = layer.forward(event)
            else:
                winners[layer_idx] = layer.forward(winners[layer_idx-1])

            if winners[layer_idx][2] > -1:
                self.hidden_layers[layer_idx].reward(winners[layer_idx])
            else:
                self.hidden_layers[layer_idx].punish(ts)

            if winners[layer_idx][2] < 0:
                break
                # winners.append(layer.forward(0,winners[layer_idx-1],ts,1))
                
        # Forward pass to the output layer
        output_winner, output_class = self.output_layer.forward(winners[-1])

        # Start rewarding or punishing the previous layers from the activity of their next layers. 
        # if output_winner > -1:
        #     self.hidden_layers[-1].record(winners[-1])
        
        # Loop from the end to first layer and record based on activation of the next layer. 
        # for layer_idx in range(-1,-len(winners),-1):
        #     if winners[layer_idx][2] > -1:
        #         #REcord next layer activity by giving the out_x, out_y of the winner from that layer itself
        #         #there is no access to the next layer's neuron identity. It is still local information only.
        #         self.hidden_layers[layer_idx-1].record(winners[layer_idx-1])
            
        # For a labelled event punish the output layer, or punish the hidden layer which failed to activate
        if label > -1:
            if output_winner > -1:
                if label == output_class:
                    self.output_layer.reward(output_winner)
                else:
                    self.output_layer.punish(output_winner,label)
            else:
                if winners[-1][2] > -1:
                    self.output_layer.punish(output_winner,label)


        return winners, output_winner, output_class



    def infer(self,event):
        '''
        Infer an event by the model. It does a forward pass and without updating the weights and thresholds accordingly based on the label. 
        attr: event : (x,y,z,ts)
        attr: label: should be an integer >= -1. -1 stands for no label. Anything above -1 considered as a class label. 

        returns: list of hidden layer winners and output winner, predicted class. Values are -1 whenver there is no output/winner from a layer. 
        '''
        x, y, z,ts = event
        # List of winners from each layer
        winners = [(-1,-1,-1) for i in self.hidden_layers]

        # Forward pass of the event and collect the winners for each hidden layer. The winner is 
        for layer_idx, layer in enumerate(self.hidden_layers):
            if layer_idx == 0:
                winners[layer_idx] = layer.forward(event)
                # winners.append(layer.forward(x, y, ts, p))
            else:
                winners[layer_idx] = layer.forward(winners[layer_idx-1])
                # winners.append(layer.forward(0,winners[layer_idx-1],ts,1))
                
        # Forward pass to the output layer
        output_winner, output_class = self.output_layer.forward(winners[-1])

        return winners, output_winner, output_class

    def save(self,fileprefix):
        '''
        Saves the weights and thresh of all layers in .npy files with filenames starting with fileprefix 
        '''
        self.save_weights(fileprefix)
        self.save_thresh(fileprefix)

    def load(self,fileprefix):
        '''
        Loads the weights and thresh of all layers in .npy files with filenames starting with fileprefix 
        '''
        self.load_weights(fileprefix)
        self.load_thresh(fileprefix)

    def save_weights(self,fileprefix):
        '''
        Saves the weights of all layers in .npy files with filenames starting with fileprefix 
        '''
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer.save_weights(fileprefix+"_hidden_"+str(layer_idx)+"_weights.npy")
        self.output_layer.save_weights(fileprefix+"_output_weights.npy")

    def save_thresh(self,fileprefix):
        '''
        Saves the thresholds of all layers in .npy files with filenames starting with fileprefix 
        '''
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer.save_thresh(fileprefix+"_hidden_"+str(layer_idx)+"_thresh.npy")
        self.output_layer.save_thresh(fileprefix+"_output_thresh.npy")
       
    def load_weights(self,fileprefix):
        '''
        Loads the weights of all layers in .npy files with filenames starting with fileprefix
        '''
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer.load_weights(fileprefix+"_hidden_"+str(layer_idx)+"_weights.npy")
        self.output_layer.load_weights(fileprefix+"_output_weights.npy")

    def load_thresh(self,fileprefix):
        '''
        Loads the thresholds of all layers in .npy files with filenames starting with fileprefix 
        '''
        for layer_idx, layer in enumerate(self.hidden_layers):
            layer.load_thresh(fileprefix+"_hidden_"+str(layer_idx)+"_thresh.npy")
        self.output_layer.load_thresh(fileprefix+"_output_thresh.npy")
