import datetime
from display_digits import Display
import math
import mnist_loader as loader
from numpy import array
import numpy as np
from numpy import random as np_random
import pickle
import pprint
#from random import uniform, randint
import thread
import time


#import plotter
#import smooth_plotter
#import input_manager
#from statsmodels.tsa.interp.denton import indicator
#from conda.core import index


eta = .5


n2, n3 = 16, 16 #neurons hidden layer1, neurons hidden layer2  


handwritten_numbers, actual_numbers = loader.load_data()[0]
validate_handwritten_numbers, validate_actual_numbers = loader.load_data()[1]


#indicator_layer = array([np_random(len(handwritten_numbers[0])), [0]*len(handwritten_numbers))
hidden_layer2 = np_random.normal(0, 3, size=n2)
hidden_layer3 = np_random.normal(0, 3, size=n3)
output_layer =  np_random.normal(0, 3, size=10)
   
weights_hl2 = np_random.normal(0, 3, size=(len(hidden_layer2), len(handwritten_numbers[0])))
weights_hl3 = np_random.normal(0, 3, size=(len(hidden_layer3), len(hidden_layer2)))
weights_ol  = np_random.normal(0, 3, size=(len(output_layer), len(hidden_layer3)))

   
def test_network(index):
    
    picture = handwritten_numbers[index]
    y = actual_numbers[index]
    
    
    current_a = np.apply_along_axis(sigmoid, 0, (np.sum(picture * weights_hl2, axis=1) + hidden_layer2))
    current_a = np.apply_along_axis(sigmoid, 0, (np.sum(current_a * weights_hl3, axis = 1) + hidden_layer3))
    current_a = np.apply_along_axis(sigmoid, 0, (np.sum(current_a * weights_ol, axis = 1) + output_layer))
    
    
    print 'test network: ', current_a


def sigmoid(x):
            
    return 1. / (1. + np.exp(-x))

test_network(0)









def run_training():
      
    for batch_index in range(0, int(batchruns/batchsize)):
        #print '    start batchrun'
        for training_index in range(0, int(batchsize)):
            
            total_index = randint(0, len(handwritten_numbers)-1)
            
            network.test_network(handwritten_numbers[total_index])
            
            current_delta_cost.append(calculate_cost(network, actual_numbers[total_index]))
            
            claculate_output_error(network, training_index, current_delta_cost[training_index])
            
            backprop(network, training_index)
            
            network.clear_network_run()
            
            
            
        b_ch_sum = 0.
        iter = 0.
        for change in batch_bias_changes:  
            if change < 0:
                change *= -1
            b_ch_sum += change
            iter += 1
        if iter != 0:
            print "averange bias change: ",b_ch_sum/iter
        batch_bias_changes = []
        network.change_values(eta, batchsize)
        list = []
        for l in test_neuron.errors:
            list.append('          ' + str(l))
        network.clear_batchrun()
        total_cost = 0.0
        for c in current_delta_cost[int(batchsize) -2]:
            total_cost += c**2
        total_cost /= len(current_delta_cost[int(batchsize) -2]) *2   
        

        
        print "                        eta: ", eta
        correct_guessed_numbers_batch = 0
        print "    total_cost: ", total_cost, "batch_index: ", batch_index
        current_delta_cost = []
      
        
      
      
    

    
def validate_network():
    correct_numbers = 0
    false_numbers = 0
    
    validation_runs = 5000
    
    
    print ''
    print 'start validation'
    print 'validation_runs :', validation_runs
    
    for i in range(1, validation_runs):
        network.test_network(validate_handwritten_numbers[i])
        
        highest_neuron_value = -1
        highest_neuron_index = -1
        neuron_index = 0
        for neuron in network.indicator_neurons:
                 
            #indi.append(neuron.output_value)
       
            if neuron.output_value > highest_neuron_value:
                highest_neuron_index = neuron_index + 0
                highest_neuron_value = neuron.output_value
            
            neuron_index += 1
           
        correct_number = validate_actual_numbers[i]
        
        if correct_number == highest_neuron_index:
            print "CORRECT: a_n: ", correct_number, "  |  c_n: ", highest_neuron_index
            correct_numbers += 1
        else:   
            print "FALSE:   a_n: ", correct_number, "  |  c_n: ", highest_neuron_index
            false_numbers += 1
            
            
        network.clear_network_run()
        
    print "validation_complete"
    print "correct_numbers: ", correct_numbers, "  false_numbers: ", false_numbers
    print "correct_false_ratio: ", float(correct_numbers)/false_numbers
         
    
    



        
#display = Display()        
#network = network()  
#train_network()
#validate_network()      
#plotter.plt.show(block=True)
#smooth_plotter.plt.show(block=True)

