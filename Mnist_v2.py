import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import nengo
import numpy as np
from numpy import random
#from src.Models.Neuron.STDPLIF import STDPLIF
#from DataLog import DataLog
from InputData import PresentInputWithPause
# from Heatmap import AllHeatMapSave,HeatMapSave
from nengo.dists import Choice
from datetime import datetime
from nengo_extras.data import load_mnist
from utilis import *
import pickle
import tensorflow as tf

#############################
# load the data
#############################

img_rows, img_cols = 28, 28
input_nbr = 10000
Dataset = "Mnist"
# (image_train, label_train), (image_test, label_test) = load_mnist()
(image_train, label_train), (image_test, label_test) = (tf.keras.datasets.mnist.load_data())


#select the 0s and 1s as the two classes from MNIST data
image_train_filtered = []
label_train_filtered = []

for i in range(0,input_nbr):
#  if (label_train[i] == 1 or label_train[i] == 0):
        image_train_filtered.append(image_train[i])
        label_train_filtered.append(label_train[i])

print("actual input",len(label_train_filtered))
print(np.bincount(label_train_filtered))

image_train_filtered = np.array(image_train_filtered)
label_train_filtered = np.array(label_train_filtered)

#############################

model = nengo.Network(label="My network",)

#############################
# Helpfull methodes
#############################

def sparsity_measure(vector):  # Gini index
    # Max sparsity = 1 (single 1 in the vector)
    v = np.sort(np.abs(vector))
    n = v.shape[0]
    k = np.arange(n) + 1
    l1norm = np.sum(v)
    summation = np.sum((v / l1norm) * ((n - k + 0.5) / n))
    return 1 - 2 * summation

#############################
# Model construction
#############################

presentation_time = 0.20 #0.35
pause_time = 0.15 #0.15
#input layer
n_in = 784
n_neurons = 20

# Learning params

learning_rate=0.00001
alf_p=0.08
alf_n=0.01
beta_p=1.5
beta_n=2.5

learning_args = {
            "lr": learning_rate,
            "winit_min":0,
            "winit_max":0.1,
    #         "tpw":50,
    #         "prev_flag":True,
            "sample_distance": int((presentation_time+pause_time)*1000),
    }

# Neuron Params 

spiking_threshold=70
tau_ref=0.02
inc_n=0.2

# Log
#full_log = False

#if(not full_log):
#    log = DataLog()
with model:
    # input layer 
    picture = nengo.Node(PresentInputWithPause(image_train_filtered, presentation_time,pause_time,0),label="Mnist")
    input_layer = nengo.Ensemble(
        n_in,
        1,
        label="Input",
        neuron_type=MyLIF_in(tau_rc=0.3,min_voltage=-1,amplitude=0.2),#nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2)),#nengo.LIF(amplitude=0.2),# nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2))
        gain=nengo.dists.Choice([2]),
        encoders=nengo.dists.Choice([[1]]),
        bias=nengo.dists.Choice([0]))

    input_conn = nengo.Connection(picture,input_layer.neurons,)

    # weights randomly initiated 
    #layer1_weights = np.round(random.random((n_neurons, 784)),5)
    # define first layer
    layer1 = nengo.Ensemble(
         n_neurons,
         1,
         label="layer1",
         neuron_type=MyLIF_out(tau_rc=0.1, min_voltage=-1),
         intercepts=nengo.dists.Choice([0]),
         max_rates=nengo.dists.Choice([20,20]),
         encoders=nengo.dists.Choice([[1]]))

    w = nengo.Node(CustomRule_post(**learning_args), size_in=784, size_out=n_neurons)
    
    nengo.Connection(input_layer.neurons, w)
    nengo.Connection(w, layer1.neurons)

    #conn1 = nengo.Connection(
    #    input_layer.neurons,
    #    layer1.neurons,
    #    transform=layer1_weights)

    # create inhibitory layer 
    inhib_wegihts = (np.full((n_neurons, n_neurons), 1) - np.eye(n_neurons)) * (- 2)

    inhib = nengo.Connection(
        layer1.neurons,
        layer1.neurons,
        synapse=0.005,
        transform=inhib_wegihts)

    weights = w.output.history
    #############################
    # setup the probes
    #############################

    #layer1_synapses_probe = nengo.Probe(conn1,"weights",label="layer1_synapses") # ('output', 'input', 'weights')
    #layer1_spikes_probe = nengo.Probe(layer1.neurons, "spikes", label="layer1_spikes") # ('output', 'input', 'spikes', 'voltage', 'refractory_time', 'adaptation', 'inhib')
    #layer1_voltage_probe = nengo.Probe(layer1.neurons, "voltage", label="layer1_voltage") # ('output', 'input', 'spikes', 'voltage', 'refractory_time', 'adaptation', 'inhib')
    
    #if(not full_log):
    #    nengo.Node(log)

    #############################

step_time = (presentation_time + pause_time) 
Args = {"Dataset":Dataset,"Labels":label_train_filtered,"step_time":step_time,"input_nbr":input_nbr}

with nengo.Simulator(model,dt=0.005) as sim:
    
    #if(not full_log):
    #    log.set(sim,Args,False,False)

    w.output.set_signal_vmem(sim.signals[sim.model.sig[input_layer.neurons]["voltage"]])
    w.output.set_signal_out(sim.signals[sim.model.sig[layer1.neurons]["out"]])
    
    sim.run(step_time * label_train_filtered.shape[0])


weights = weights[-1]

#if(not full_log):
#    log.closeLog()

#print("Prune : ",np.round(100 - (((n_in * n_neurons) - np.sum(np.array(w.output.pruned)))* 100 / (n_in * n_neurons)),2)," %")
now = str(datetime.now().time())
folder = "My_Sim_"+"learning_rate="+str(learning_rate)+"_alf_p="+str(alf_p)+"_alf_n="+str(alf_n)+"_beta_p="+str(beta_p)+"_beta_n="+str(beta_n)+"_"+now

if not os.path.exists(folder):
    os.makedirs(folder)
    

i = 0
#np.putmask(weights,w.output.pruned == 1,np.NAN)

#save the pruned model
pickle.dump(weights, open( "mnist_params_STDP_Pruned", "wb" ))
for n in weights:
    plt.matshow(np.reshape(n,(28,28)),interpolation='none')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(folder+"/"+str(i)+".png")
    plt.cla()
    i = i + 1


#Ratio = Ratio + (alpha * (CRmaining / CTotal))
