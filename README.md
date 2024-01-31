# Python Implementation of ODESA - Optimized Deep Event-driven SNN Architecture.

Official Implementation of the paper - "An Optimized Deep Spiking Neural Network Architecture Without Gradients"
![STTP](https://user-images.githubusercontent.com/6342039/192964814-66465016-e92c-4477-9cd0-bf8c25c046e7.jpg)

## Usage

A typical code usage of ODESA would look like: 

```python
from ODESA.FullyConnected import FullyConnected as HiddenLayer
from ODESA.Classifier import Classifier as OutputLayer
from ODESA.FCModel import FCModel as Model

# Instantiate the hidden layers
input_layer = HiddenLayer(input_layer_context_rows, 
                                input_layer_context_cols,
                                input_layer_n_neurons,
                                input_layer_eta,
                                input_layer_threshold_open,
                                input_layer_tau,
                                input_layer_trace_tau,
                                input_layer_thresh_eta,
                                cumulative_ts=True)

# Instantiate the output layer
output_layer = OutputLayer(output_layer_context_rows,
                                output_layer_context_cols,
                                output_layer_n_neurons_per_class,
                                output_layer_n_classes,
                                output_layer_eta,
                                output_layer_tau,
                                output_layer_threshold_open,
                                thresh_eta = output_layer_thresh_eta,
                                cumulative_ts=True)

# Initalize a Feast Model to have all the layers. 
model = Model()
# Add hidden layers. Copy the statement to add more layers to the network.  
model.add_hidden_layer(input_layer)
# Finally add the output layer. Every model has to have an output layer. 
model.add_output_layer(output_layer)


# Each labelled event can be passed to the model using the forward function. If the event has no label assigned to it,
# 'label' should be equal to '-1'.
hidden_layer_winners, output_layer_winner, output_class = model.forward((x,y,ts,1),label)

```

## Future work:
A Julia implementation of ODESA which is much faster than the python version is in the works. Contact me @Y[dot]Bethi[at]westernsydney[dot]edu[dot]au if you are interested in that or need any help using ODESA for a spatio-temporal problem. 

## Citation for reference
Bethi, Yeshwanth, Ying Xu, Gregory Cohen, André Van Schaik, and Saeed Afshar. "An optimised deep spiking neural network architecture without gradients." IEEE Access (2022).
```
@article{bethi2022optimised,
title={An optimised deep spiking neural network architecture without gradients},
author={Bethi, Yeshwanth and Xu, Ying and Cohen, Gregory and Van Schaik, Andr{\'e} and Afshar, Saeed},
journal={IEEE Access},
year={2022},
volume={10},
number={},
pages={97912-97929},
doi={10.1109/ACCESS.2022.3200699},
publisher={IEEE}}
```
