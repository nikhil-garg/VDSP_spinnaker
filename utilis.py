import nengo
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import os
from nengo.dists import Choice
from datetime import datetime
from nengo_extras.data import load_mnist
import pickle
from nengo.utils.matplotlib import rasterplot

plt.rcParams.update({'figure.max_open_warning': 0})
import time

from InputData import PresentInputWithPause
# from custom_rule import CustomRule
# from custom_rule import CustomRule_prev

from nengo_extras.graphviz import net_diagram
import nengo_ocl

from nengo.neurons import LIFRate
# from custom_rule import CustomRule
# from custom_rule import CustomRule_prev
from nengo.params import Parameter, NumberParam, FrozenObject
from nengo.dists import Choice, Distribution, get_samples, Uniform

from nengo.utils.numpy import clip, is_array_like

from nengo.connection import LearningRule
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import LearningRuleType
from nengo.builder.learning_rules import get_pre_ens,get_post_ens
from nengo.neurons import AdaptiveLIF
from nengo.synapses import Lowpass, SynapseParam
from nengo.params import (NumberParam,Default)
from nengo.dists import Choice
#from nengo_extras.neurons import spikes2events
from nengo.utils.numpy import clip
import numpy as np
import random
import math


class MyLIF_in(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage

    def step(self, dt, J, output, voltage, refractory_time):
        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = -1
        refractory_time[spiked_mask] = self.tau_ref + t_spike
class MyLIF_out(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None, inhib=[]
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage
        self.inhib = inhib

    def step(self, dt, J, output, voltage, refractory_time):
        # look these up once to avoid repeated parameter accesses

        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[:] = spiked_mask * (self.amplitude / dt)
        
        if(np.sum(output)!=0):
            voltage[voltage != np.max(voltage)] = -1
            
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = -1
        refractory_time[spiked_mask] = self.tau_ref + t_spike

        
def fun_post(X,
       a1=0,a2=1,a3=0,
       b1=1,b2=1,b3=1,b4=1,b5=1,b6=1,b7=1,
       c1=0,c2=1,c3=0,
       d1=1,d2=1,d3=1,d4=1,d5=1,d6=1,d7=1,
       e1=0, e2=0, e3=0, e4=0,e5=0,e6=0,
       alpha1=1,alpha2=0    
       ): 
    w, vmem = X
    vthp=0.25
    vthn=0.25
    vprog=0
    w_pos = (e1*w) + e3
    w_neg = e2*(1-w) + e4
    v_ov_p =  vmem - (vprog+vthp) + e5
    v_ov_n = (vprog-vthn) - vmem  + e6
    cond_1 = vmem<(vprog-vthn)
    cond_2 = vmem>(vprog+vthp)
    f1 = a1 + a2*(w_pos**1) + a3*(w_pos**2)
    g1 = b1 + b2*np.sin(b3*v_ov_n + b4) + b5*np.cos(b6*v_ov_n + b7)
    f2 = c1 + c2*(w_neg**1) + c3*(w_neg**2)
    g2 = d1 + d2*np.sin(d3*v_ov_p + d4) + d5*np.cos(d6*v_ov_p + d7)
    dW = (abs(cond_1*(alpha1*f1*g1)))  + (-1*cond_2*(alpha2*cond_2*f2*g2))    
    return dW

popt = np.array((-5.44634746e+00, -8.42848816e-01,  1.49956029e+00,  1.98056395e+00,
        4.33050527e+00,  5.28219321e-01,  7.24397333e-02,  3.37358302e+00,
        9.98808901e-01,  2.87121896e+00, -5.57633406e-01, -2.75112832e-01,
        1.60193659e+00,  4.09073550e-01, -9.26010737e-01,  4.91251299e-01,
        6.61539169e-03, -1.05305318e-01,  1.93590366e+00,  3.55720979e-01,
        3.61854190e-03, -3.54039473e-01, -1.64873794e+00, -1.93935931e-01,
        1.14033130e+00,  4.57240635e-01,  5.57668985e+00,  2.64857548e+00))

class CustomRule_post(nengo.Process):
   
    def __init__(self, vthp=0.25, vthn=0.25, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1):
       
        self.vthp = vthp
        self.vthn = vthn
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None
        
        self.sample_distance = sample_distance
        self.lr = lr
        
        self.history = []
        self.current_weight = []
        self.update_history = []
        
        self.vmem_prev = 0
        
        self.winit_min = winit_min
        self.winit_max = winit_max
        
        self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))
        dw = np.zeros((shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            vmem = self.signal_vmem_pre
            vmem = np.clip(vmem, -1, 1)
            post_out = self.signal_out_post
            
            
            vmem = np.reshape(vmem, (1, shape_in[0]))         
            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))
          
            vmem = np.vstack([vmem]*shape_out[0])            
            post_out_matrix = np.hstack([post_out_matrix]*shape_in[0])
            
            dw = post_out_matrix*dt*fun_post((self.w,vmem),*popt)   
            self.w += dw*self.lr  
            self.w = np.clip(self.w, 0,1)

            
            
            if (self.tstep%self.sample_distance ==0):
                self.history.append(self.w.copy())
                self.update_history.append(dw.copy())
            
            self.tstep +=1
            
            
            self.vmem_prev = vmem.copy()
            return np.dot(self.w, x)
        
        return step   

        self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal




#create new neuron type STDPLIF 
def build_or_passthrough(model, obj, signal):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build(obj, signal)

#---------------------------------------------------------------------
# Neuron Model declaration 
#---------------------------------------------------------------------

#create new neuron type STDPLIF 

class STDPLIF(AdaptiveLIF):
    probeable = ('spikes', 'voltage', 'refractory_time','adaptation','inhib') #,'inhib'
    
    def __init__(self, spiking_threshold = 1, inhib=[], **lif_args): # inhib=[],T = 0.0
        super(STDPLIF, self).__init__(**lif_args)
        # neuron args (if you have any new parameters other than gain
        # an bais )
        self.inhib = inhib
        #self.T = T
        self.spiking_threshold=spiking_threshold
    @property
    def _argreprs(self):
        args = super(STDPLIF, self)._argreprs
        print("argreprs")
        return args

    # dt : timestamps 
    # J : Input currents associated with each neuron.
    # output : Output activities associated with each neuron.
    def step(self, dt, J, output, voltage, refractory_time, adaptation,inhib):#inhib
        #self.T = round(self.T + 0.005,3)

        #self.TT = self.T % 0.35

        #if(self.TT >= 0.20):
             #print("reseting")
             #voltage[...] = 0
             #inhib[...] = 0
             #J[...] = 0
             #refractory_time[...] = 0
             #output[...] = 0
        #if((self.T % 0.5) == 0):
        #    #print("\n","Next input","\n")
        #    self.T = 0
        #    voltage[...] = 0
        #    inhib[...] = 0
        #J[J > 1000] = 1.2
        #J[J <= 20] = 1.5
        #if(np.sum(output) > 1):
        #    inhib[((output != 0) & (inhib == 0))] = 10
        #    inhib[J < 0] = 20
        #    J[inhib != 0] = 0
        #    voltage[inhib != 0] = 0
        #J[:] = np.round(J - J.mean(),5) / np.round(J.std(),5)
        #J = J - J.min()
        #if(J.max() >0):
        #J = (J / J.max()) + 1

        # tInhibit = 10 MilliSecond
        # AdaptiveThresholdAdd = 0.05  millivolts
        # MembraneRefractoryDuration = = 1 MilliSecond
        #print("J",J,"output",output,"voltage",voltage,refractory_time) 
        #print("J",J,"voltage",voltage,output)

        n = adaptation
        
        J = J - n
        # ----------------------------

        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = (voltage > self.spiking_threshold)
        output[:] = spiked_mask * (self.amplitude / dt)
        output[voltage != np.max(voltage)] = 0  
        if(np.sum(output) != 0):
            voltage[voltage != np.max(voltage)] = 0 
            inhib[(voltage != np.max(voltage)) & (inhib == 0)] = 20
        #print("voltage : ",voltage)
        voltage[inhib != 0] = 0 
        J[inhib != 0] = 0
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )
        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = 0
        voltage[refractory_time > 0] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike
        # ----------------------------

        n += (dt / self.tau_n) * (self.inc_n * output - n)

        #AdaptiveLIF.step(self, dt, J, output, voltage, refractory_time, adaptation)
        inhib[inhib != 0] += - 1
        #J[...] = 0
        #output[...] = 0
        

#---------------------------------------------------------------------
#add builder for STDPLIF
#---------------------------------------------------------------------

@Builder.register(STDPLIF)
def build_STDPLIF(model, STDPlif, neurons):
    
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['pre_filtered'] = Signal(
        np.zeros(neurons.size_in), name="%s.pre_filtered" % neurons)
    model.sig[neurons]['post_filtered'] = Signal(
        np.zeros(neurons.size_in), name="%s.post_filtered" % neurons)
    model.sig[neurons]['inhib'] = Signal(
        np.zeros(neurons.size_in), name="%s.inhib" % neurons)
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in),name= "%s.adaptation" % neurons
    )
    # set neuron output for a given input
    model.add_op(SimNeurons(neurons=STDPlif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            state={"voltage": model.sig[neurons]['voltage'],
                                    "refractory_time": model.sig[neurons]['refractory_time'],
                                    "adaptation": model.sig[neurons]['adaptation'],
                                    "inhib": model.sig[neurons]['inhib']
                                     }))

import os
import re
import cv2


def gen_video(directory, f_prename):
    
    assert os.path.exists(directory)

    img_array = []
    for filename in os.listdir(directory):
        if f_prename in filename:
            nb = re.findall(r"(\d+).png", filename)
            if len(nb) == 1:
                img = cv2.imread(os.path.join(directory, filename))
                img_array.append((int(nb[0]), img))

    height, width, layers = img.shape
    size = (width, height)

    img_array = sorted(img_array, key=lambda x: x[0])
    video_path = os.path.join(directory, f"{f_prename}.avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 2, size)

    for _, img in img_array:
        out.write(img)
    out.release()

    print(f"{video_path} generated successfully.")