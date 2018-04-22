# -*- coding: utf-8 -*-

import datetime
from display_digits import Display
import math
import mnist_loader as loader
import pickle
from random import randint
import time


handwritten_numbers, actual_numbers = loader.load_data()[0]


class axon():
    def __init__(self, id, reciver_id, network):             
            self.id = id
            self.reciver_id = reciver_id
            self.weight = 0#randint(-10,10)
            self.network = network
            self.reciver = None
            self.donor = None
            
            
    def release(self, donor_value, donor):
            
            if self.donor.id != donor.id:
                print 'ALARM ALARM ALARM, FUCK, axon donor instance is not the right one, crap'
                print 'self.donor.id: ', self.donor.id, "  |   donor.id: ", donor.id
            else:
                print 'YEAHY donor instance works correctly'
        
            if self.reciver == None:
                self.reciver = network.get_neuron(self.reciver_id)
            
            
            self.reciver.stimulate(donor_value * self.weight)
            

class receptor():
    def __init__(self, id, f_axons):
        self.id = id
        self.f_axons = f_axons
        self.output = 0
        
    def stimulate(self, value):    
        self.output = value
        
    def fire(self):
        for axon in self.f_axons:
            axon.release(self.output, self)


class neuron():
    def __init__(self, id, f_axons, r_axons): #f_axons = fire axons, r_axons = recive_axons
        
        self.id = id #[1=neuron, 0 = axon -- self.layer -- row -- reciver_list_axon]
        self.stimulus_sum = 0
        self.output_value = -1.
        self.f_axons = f_axons
        self.r_axons = r_axons
        self.bias = 0#randint(-10.,10.)
        self.network = network
        self.errors = [] #contains for each batchrun its error
        self.batch_output_values = []
        self.cost = -1.
        
        if self.r_axons != []:
            for r in self.r_axons:
                r.donor = self
            
        
    def stimulate(self, input_value):
        
        self.stimulus_sum += input_value
        
    def stimulation_complete(self):
        total_output_value = float(self.stimulus_sum) + self.bias
        self.output_value = sigmoid(total_output_value)
        self.batch_output_values.append(self.output_value)

    def fire(self):
        for axon in self.f_axons:
            axon.release(self.output_value)
            
    def calculate_output_error(self, training_index, new_cost):
        #δx,L=∇aCx⊙σ′(zx,L)
        print new_cost
        delta_cost = self.cost - new_cost[self.id[2]]#not sure weather correct index is putten into new_cost bitch
        error = delta_cost * deriv_sigmoid(self.stimulus_sum)
        
        if len(self.errors) != training_index:
            print "Warning: invalid trainingindex, errorplace already in use"
             
        self.errors.append(error)

    def calculate_backprop_error(self, training_index):
        #δx,l=((wl+1)Tδx,l+1)⊙σ′(zx,l). 
        
        
        error_sum = 0
        #i = 0
        for axon in self.f_axons:
            error_sum += (axon.weight * axon.reciver.errors[training_index]) 
            error_sum *= deriv_sigmoid(self.stimulus_sum)
            #i += 1

        #averange_error = error_sum/i don't know where i picked this
        
        if len(self.errors) != training_index:
            print "Warning: invalid training_index, errorplace already in use"
             
        self.errors.append(error_sum)

	def clear(self):
		self.stimulus_sum = 0
		self.output_value = 0
     

class network():
    
    
    receptors = []
    receptor_axons = []
    
    layer1_neurons = []
    layer1_axons = []
    layer1_size = 16
    
    
    layer2_neurons = []
    layer2_axons = []
    layer2_size = 16
    
    indicator_neurons = []
    indicator_layer_size = 10

    
    def __init__(self): #maybe add later self.layer_neuron_count: self.layer_neuron_count is a tuple with integers, each int represents a self.layer and its value says how many neurons it contains 
                
        #init self.receptors + axons
        for i in range (0, 784):
            current_neuron_id = [1, 0, i, -1]   #[1=neuron, 0 = axon -- self.layer -- row], reciver_list_axon
            #create_axons
            current_axons = []
            for axon_i in range(1, self.layer1_size):
                current_id = [0, 0, i - 1, axon_i - 1]
                current_revicer_id = [1,1, axon_i -1, -1]
                current_axons.append(axon(current_id + list(), current_revicer_id + list(), self))
            
            self.receptors.append(receptor(current_neuron_id + list(), current_axons + list()))
            self.receptor_axons.append(current_axons + list())
            
            
            
        #init l1 neurons
        for i in range (0, self.layer1_size):
            current_neuron_id = [1, 1, i, -1]   #[1=neuron, 0 = axon -- self.layer -- row], reciver_list_axon
            #create_axons
            current_axons = []
            for axon_i in range(1, self.layer2_size):
                current_id = [0, 1, i - 1, axon_i - 1]
                current_revicer_id = [1,2, axon_i -1, -1]
                current_axons.append(axon(current_id + list(), current_revicer_id + list(), self))
            
            self.layer1_neurons.append(neuron(current_neuron_id + list(), current_axons + list(), self.receptor_axons[i] + list()))
            #ALARM ALARM irgendetwas mit den receptor axons ist faul!!!!!!!!!!!!!!!!!!!
            self.layer1_axons.append(current_axons + list())
        
        #init l2 neurons
        for i in range (0, self.layer2_size):
            current_neuron_id = [1, 2, i, -1]   #[1=neuron, 0 = axon -- self.layer -- row], reciver_list_axon
            #create_axons
            current_axons = []
            for axon_i in range(1, self.indicator_layer_size):
                current_id = [0, 2, i - 1, axon_i - 1]
                current_revicer_id = [1,3, axon_i -1, -1]
                current_axons.append(axon(current_id + list(), current_revicer_id + list(), self))
            
            
            self.layer2_neurons.append(neuron(current_neuron_id + list(), current_axons + list(), self.layer1_axons[i] + list()))
            self.layer2_axons.append(current_axons + list())
        
        
        
        #init self.indicator neurons
        for i in range (0, self.indicator_layer_size):
            current_neuron_id = [1, 3, i, -1]   #[1=neuron, 0 = axon -- self.layer -- row -- reciver_list_axon]
            self.indicator_neurons.append(neuron(current_neuron_id + list(), [], self.layer2_axons[i] + list()))
        
    
    
    
    def get_neuron(self, neuron_id):
        if neuron_id[1] == 0:
            neuron = self.receptors[neuron_id[2]] 
        elif neuron_id[1] == 1:
            neuron = self.layer1_neurons[neuron_id[2]]
        elif neuron_id[1] == 2:
            neuron = self.layer2_neurons[neuron_id[2]]
        elif neuron_id[1] == 3:         
            neuron = self.indicator_neurons[neuron_id[2]]
        else: 
            raise RuntimeError("Neuron not found!")
            
        if neuron_id != neuron.id:
            raise RuntimeError("get_neuron doesnt find correct neuron, searched id: ", neuron_id, "  , found id: ", neuron.id)

        
        return neuron

    def get_highest_ineuron(self):
        h_in = -1
        
        for neuron in self.indicator_neurons:
            o_v = neuron.output_value
            if h_in < o_v:
                h_in = o_v
        
        print o_v
        
    

    def test_network(self, picture):
        
        #load receptors
        for i in range(0, len(picture) - 1):
            receptor_activation = picture[i]
            self.receptors[i].stimulate(receptor_activation)
            
        
        #fire receptors
        for r in self.receptors:
            r.fire()
        
        
        #fire layer1 neurons
        for n1 in self.layer1_neurons:
            n1.stimulation_complete()
        
        
        for n1 in self.layer1_neurons:
            n1.fire()
        
        
        #fire layer2 neurons
        for n2 in self.layer2_neurons:
            n2.stimulation_complete()
        
        for n2 in self.layer2_neurons:
            n2.fire()
        
        
        
        #print indicator neurons
        for ineuron in self.indicator_neurons:
            ineuron.stimulation_complete()
            print ineuron.output_value
        
        
    def clear_network_run(self):
        for neuron in self.indicator_neurons:
           neuron.clear()
        for neuron in self.layer2_neurons:
           neuron.clear()
        for neuron in self.layer1_neurons:
           neuron.clear()
        
        
        
    def change_values(self , training_speed):
        '''Gradient descent: For each l=L,L−1,…,2 update the
        weights according to the rule 
        wl→wl−ηm∑xδx,l(ax,l−1)T, 
        and the biases according to the rule 
        bl→bl−ηm∑xδx,l.
        '''
        
        for neuron in self.indicator_neurons:
           neuron.bias = self.calc_new_bias(neuron, training_speed)
        for neuron in self.layer2_neurons:
           neuron.bias = self.calc_new_bias(neuron, training_speed)
        for neuron in self.layer1_neurons:
           neuron.bias = self.calc_new_bias(neuron, training_speed)
        
        
        
        for axon in self.layer2_axons:
            axon.weight = self.calc_new_wheight(axon, training_speed)
        for axon in self.layer1_axons:
            axon.weight = self.calc_new_wheight(axon, training_speed)
        for axon in self.indicator_axons:
            axon.weight = self.calc_new_wheight(axon, training_speed)
            

    def calc_new_bias(self, neuron, q):    
        #bl→bl−ηm∑xδx,l
        b = neuron.bias
        error_sum = 0
        
        trainings = len(neuron.errors)
        
        for error in neuron.errors:
            error_sum += error
        
        new_b = b - (q/trainings) * error_sum
 
        return new_b
 
    def calc_new_wheight(self, axon, q):    
        #wl→wl−ηm∑xδx,l(ax,l−1)T
        w = axon.weight
        
        donor_activation_sum = 0
        reciver_error_sum = 0
        
        donor = axon.donor
        reciver = axon.reciver
        
        for donor_activation in donor.batch_output_activations:
            donor_activation_sum += donor_activation
        for reciver_error in reciver.errors:
            reciver_error_sum += reciver_error
        
        new_w = w - (q/trainings) * donor_activation_sum * reciver_error_sum
        
        return new_w
    
    def clear_batchrun(self):
        
        for neuron in self.indicator_neurons:
           neuron.errors = []
           neuron.batch_output_values = []
        for neuron in self.layer2_neurons:
           neuron.errors = []
           neuron.batch_output_values = []
        for neuron in self.layer1_neurons:
           neuron.errors = []
           neuron.batch_output_values = []
        
    
def sigmoid(x):
    return 1. / (1. + math.exp(-float(x)))

def deriv_sigmoid(x):     
    return sigmoid(x)*(1-sigmoid(x))

def calculate_cost(network, actual_number):
    on_index = actual_number - 1
    cost = []
    i = 0
    for neuron in network.indicator_neurons:
        if i == on_index:
            #confirmed print "Master, please check weather on_index works correct;  on_index: ", on_index, "   |  actual_number: ", actual_number
            cost.append(1 - neuron.output_value)
        else:
            cost.append(neuron.output_value)
        
        
        i += 1
    
    return cost
        
def claculate_output_error(network, training_index, current_cost):
    
    for i_neuron in network.indicator_neurons:
        i_neuron.calculate_output_error(training_index, current_cost)

        
def backprop(network, training_index):
    
    for l2neuron in network.layer2_neurons:
        l2neuron.calculate_backprop_error(training_index)
    
    for l1neuron in network.layer1_neurons:
        l1neuron.calculate_backprop_error(training_index)



def clear_testrun(network):
    pass

    
def train_network():
    global handwritten_numbers
    global actual_numbers
    
    
    batchsize = 10
    eta = 1
    #network = network()
    tranings = 1000
    
    total_cost = -1
    current_cost = []
    
    
    print 'start training'
    
    for batch_index in range(0, tranings/batchsize -1):
        print '    start batchrun'
        for training_index in range(0, batchsize - 1):
            total_index = (batch_index*batchsize) + training_index
            print '        test network'
            network.test_network(handwritten_numbers[total_index])
            print '        clauculate cost'
            current_cost.append(calculate_cost(network, actual_numbers[total_index]))
            print '        claculate output error'
            claculate_output_error(network, training_index, current_cost[training_index])
            print '        backprop'
            backprop(network, training_index)
            print '        clear network_run'
            network.clear_network_run()
        
        print '        change values'
        network.change_values(1)
        print '        clear batchrun'
        network.clear_batchrun()
    
    
        
        total_cost = 0.0
    
        for c in current_cost[batchsize -2]:
            total_cost += c**2
        
        
        total_cost /= len(current_cost[batchsize -2]) *2   
        
        print '    end Batchrun'
        print "    total_cost: ", total_cost
        current_cost = []
        exit()
     
    
    print 'training end'    
    path_name = ''.join(("network_states/network_cost: ", str(total_cost), " | ", datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"), ".pkl"))
    print 'save network as: ', path_name

    with open(path_name, 'wb') as f:
        pickle.dump('network', f, pickle.HIGHEST_PROTOCOL)

    
    
    
    
    
    
    



        
#display = Display()        
network = network()  

train_network()      


























'''class indicator_neuron():
    def __init__(self, id, axons):
        
        self.id = id
        self.stimulus_sum = 0
        self.bias = randint(-10,10)
        self.output_value = -1
        self.axons = axons
        
    def stimulate(self, input_value):
        self.stimulus_sum += input_value

    def stimulation_complete(self):
        total_output_value = float(self.stimulus_sum) + self.bias
        self.output_value = 1. / (1. + math.exp(-float(total_output_value)))
    
        

    def clear(self):
        self.stimulus_sum = 0
        self.output_value = 0
        
'''
        