# -*- coding: utf-8 -*-

import datetime
#from display_digits import Display
import math
import mnist_loader as loader
#from numba import jit
import pickle
#import pprint
from random import uniform, randint
#import time


#import plotter
import smooth_plotter
#import input_manager

handwritten_numbers, actual_numbers = loader.load_data()[0]
validate_handwritten_numbers, validate_actual_numbers = loader.load_data()[1]

#display = Display()


class axon():
    def __init__(self, id, reciver_id, network):             
            self.id = id
            self.reciver_id = reciver_id
            self.weight = uniform(-3.,3.) #0
            self.network = network
            self.reciver = None
            self.donor = None
            
            
    def release(self, donor_value, donor):
            
            if self.donor.id != donor.id:
                print 'ALARM ALARM ALARM, FUCK, axon donor instance is not the right one, crap'
                print 'self.id: ', self.id, '  |  self.donor.id: ', self.donor.id, "  |   donor.id: ", donor.id
            else:
               pass
               # print 'YEAHY donor instance works correctly'
        
            if self.reciver == None:
                self.reciver = network.get_neuron(self.reciver_id)
            
            
            self.reciver.stimulate(donor_value * self.weight)
            

class receptor():
    def __init__(self, id, f_axons):
        self.id = id
        self.f_axons = f_axons
        self.output = 0
        self.batch_output_values = []
        
        if self.f_axons != []:
            for f in self.f_axons: 
                #print 'neuron mariging axon, neuron_id: ', self.id, '   |   fire_axon_id: ', f.id
                f.donor = self
        
    def stimulate(self, value):    
        self.output = value
        self.batch_output_values.append(self.output)
        
        
    def fire(self):
        for axon in self.f_axons:
            axon.release(self.output, self)

    
    def clear_batchrun(self):
        sum = 0
        for ou in self.batch_output_values:
            sum += ou
        
        
        self.batch_output_values = []

class neuron():
    def __init__(self, id, f_axons, r_axons): #f_axons = fire axons, r_axons = recive_axons
        
        self.id = id #[1=neuron, 0 = axon -- self.layer -- row -- reciver_list_axon]
        self.stimulus_sum = 0
        self.output_value = -1.
        self.f_axons = f_axons # axon this neuron puts information in
        self.r_axons = r_axons # axon this neuron recives information from
        self.bias = uniform(-3.,3.)#0
        self.old_bias = None #for debugging
        self.network = network
        self.errors = [] #contains for each batchrun its error
        self.batch_output_values = []
        self.cost = -1.
        
        if self.f_axons != []:
            for f in self.f_axons: 
                #print 'neuron mariging axon, neuron_id: ', self.id, '   |   fire_axon_id: ', f.id
                f.donor = self
            
        
    def stimulate(self, input_value):
        
        self.stimulus_sum += input_value
        
    def stimulation_complete(self):
        total_output_value = float(self.stimulus_sum) + self.bias
        self.output_value = sigmoid(total_output_value)
        self.batch_output_values.append(self.output_value)

    def fire(self):
        for axon in self.f_axons:
            axon.release(self.output_value, self)
            
    def calculate_output_error(self, training_index, new_delta_cost):
        #δx,L=∇aCx⊙σ′(zx,L)
        #print new_cost
        #delta_cost = self.cost - new_cost[self.id[2]]#(very likly correct)not sure weather correct index is putten into new_cost bitch
        
        #using the quadratic cost funktion
        #delta_cost = new_delta_cost[self.id[2]]
        #error = delta_cost * deriv_sigmoid(self.stimulus_sum)
        
        #using the cross entropy funktion
        error = new_delta_cost[self.id[2]]
        
        if len(self.errors) != training_index:
            print "Warning: invalid trainingindex, errorplace already in use"
             
        self.errors.append(error)

    def clear_network_run(self):
        self.stimulus_sum = 0
        self.output_value = 0
        
    def clear_batchrun(self):
        self.stimulus_sum = 0
        self.output_value = 0
        
        self.batch_output_values = []
        self.errors = []

    def calculate_backprop_error(self, training_index):
        #δx,l=((wl+1)Tδx,l+1)⊙σ′(zx,l). 
        
        
        error_sum = 0
        #i = 0
        for axon in self.f_axons:
            error_sum += (axon.weight * axon.reciver.errors[training_index]) 
            #error_sum *= deriv_sigmoid(self.stimulus_sum) #just for quadratic cost funktion
            #i += 1

            

        #averange_error = error_sum/i don't know where i picked this
        
        if len(self.errors) != training_index:
            print "Warning: invalid training_index, errorplace already in use"
             
        self.errors.append(error_sum)


 

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
    
    all_axons = []
    all_neurons = []
    
    def __init__(self): #maybe add later self.layer_neuron_count: self.layer_neuron_count is a tuple with integers, each int represents a self.layer and its value says how many neurons it contains 
                
        #init self.receptors + axons
        for i in range (0, 784):
            current_neuron_id = [1, 0, i, -1]   #[1=neuron, 0 = axon -- self.layer -- row], reciver_row(for axon)
            #create_axons
            current_axons = []
            for axon_i in range(0, self.layer1_size):
                current_id = [0, 0, i, axon_i]
                current_revicer_id = [1,1, axon_i, -1]
                current_axons.append(axon(current_id + list(), current_revicer_id + list(), self))
            
            self.receptors.append(receptor(current_neuron_id + list(), current_axons + list()))
            self.receptor_axons.append(current_axons + list())
            
            
            
            
        #init l1 neurons
        for i in range (0, self.layer1_size):
            current_neuron_id = [1, 1, i, -1]   #[1=neuron, 0 = axon -- self.layer -- row], reciver_row(for axon)
            #create_axons
            current_axons = []
            for axon_i in range(0, self.layer2_size):
                current_id = [0, 1, i, axon_i]
                current_revicer_id = [1,2, axon_i, -1]
                current_axons.append(axon(current_id + list(), current_revicer_id + list(), self))
            
            
            self.layer1_neurons.append(neuron(current_neuron_id + list(), current_axons + list(), self.receptor_axons[i] + list()))
            self.layer1_axons.append(current_axons + list())
        
        #init l2 neurons
        for i in range (0, self.layer2_size):
            current_neuron_id = [1, 2, i, -1]   #[1=neuron, 0 = axon -- self.layer -- row], reciver_row(for axon)
            #create_axons
            current_axons = []
            for axon_i in range(0, self.indicator_layer_size):
                current_id = [0, 2, i, axon_i]
                current_revicer_id = [1,3, axon_i, -1]
                current_axons.append(axon(current_id + list(), current_revicer_id + list(), self))
            
            
            self.layer2_neurons.append(neuron(current_neuron_id + list(), current_axons + list(), self.layer1_axons[i] + list()))
            self.layer2_axons.append(current_axons + list())
        
        
        
        #init self.indicator neurons
        for i in range (0, self.indicator_layer_size):
            current_neuron_id = [1, 3, i, -1]   #[1=neuron, 0 = axon -- self.layer -- row -- reciver_list_axon]
            self.indicator_neurons.append(neuron(current_neuron_id + list(), [], self.layer2_axons[i] + list()))
        
        
        self.all_neurons.extend(self.layer1_neurons)
        self.all_neurons.extend(self.layer2_neurons)
        self.all_neurons.extend(self.indicator_neurons)
        
        all_axons_shitty = []
        all_axons_shitty.extend(self.receptor_axons)
        all_axons_shitty.extend(self.layer1_axons)
        all_axons_shitty.extend(self.layer2_axons)
    
        for shit in all_axons_shitty: #all axons shitty consists off al lot of little lists and it is big shit to work with something like that
            self.all_axons.extend(shit)
    
    
    
    def get_neuron(self, neuron_id):
        #print neuron_id
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
        sum
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
            #print ineuron.output_value
        
        
    def clear_network_run(self):
        
        for neuron in self.all_neurons:
            neuron.clear_network_run()
        
        
        '''
        for neuron in self.indicator_neurons:
           neuron.clear()
        for neuron in self.layer2_neurons:
           neuron.clear()
        for neuron in self.layer1_neurons:
           neuron.clear()
        ''' 
           
        
        
        
    def change_values(self , training_speed, batchsize):
        '''Gradient descent: Forprint actual_numbers[1]
        each l=L,L−1,…,2 update the
        weights according to the rule 
        wl→wl−ηm∑xδx,l(ax,l−1)T, 
        and the biases according to the rule 
        bl→bl−ηm∑xδx,l.
        '''
        
        for neuron in self.all_neurons:
            neuron.old_bias = neuron.bias
            neuron.bias = self.calc_new_bias(neuron, training_speed, batchsize)
        
        for axon in self.all_axons:
            axon.weight = self.calc_new_wheight(axon, training_speed, batchsize)
        
        '''for neuron in self.indicator_neurons:
           neuron.bias = self.calc_new_bias(neuron, training_speed)
        for neuron in self.layer2_neurons:
           neuron.bias = self.calc_new_bias(neuerror = delta_cost * (1/deriv_sigmoid(self.stimulus_sum))
        ron, training_speed)
        for neuron in self.layer1_neurons:
           neuron.bias = self.calc_new_bias(neuron, training_speed)
        
        
        
        for axon in self.layer2_axons:
            axon.weight = self.calc_new_wheight(axon, training_speed)
        for axon in self.layer1_axons:
            axon.weight = self.calc_new_wheight(axon, training_speed)
        for axon in self.indicator_axons:
            axon.weight = self.calc_new_wheight(axon, training_speed)
        '''    

    def calc_new_bias(self, neuron, q, batchsize):    
        #bl→bl−ηm∑xδx,l
        
        global batch_bias_changes
        
        b = neuron.bias
        error_sum = 0
        
        
        for error in neuron.errors:
            error_sum += error
        
        new_b = b - ((q/batchsize) * error_sum)
        
        batch_bias_changes.append((q/batchsize) * error_sum)
        #print "new bias: ", new_b, "  |  error sum: ", error_sum
        return new_b
    
    #@jit
    def calc_new_wheight(self, axon, q, batchsize):    
        #wl→wl−ηm∑xδx,l(ax,l−1)T
        w = axon.weight
        
        donor_activation_sum = 0
        reciver_error_sum = 0
        
        donor = axon.donor
        reciver = axon.reciver
        
        for donor_activation in donor.batch_output_values: #mabe instead of batch_output values, use batch_output activations
            donor_activation_sum += donor_activation
        for reciver_error in reciver.errors:
            reciver_error_sum += reciver_error
        
        new_w = w - (q/batchsize) * donor_activation_sum * reciver_error_sum
        
        
        #print donor_activation_sum, reciver_error_sum
        #print "new weight: ", new_w, "  |  w: ", w
        
        return new_w
    
    def clear_batchrun(self):
        
        for receptor in self.receptors:
            receptor.clear_batchrun()
        
        for neuron in self.all_neurons:
            neuron.clear_batchrun()
    
        '''
        for axon in self.all_axons:
            axon.clear_batchrun()
        '''
#@jit
def sigmoid(x):
    if x > 100:
        x = 100
            
    return 1. / (1. + math.exp(-float(x)))
#@jit
def deriv_sigmoid(x):     
    return sigmoid(x)*(1.-sigmoid(x))

print "deriv sigmoid(10)", deriv_sigmoid(10)
print ""


def calculate_cost(network, actual_number):
    global correct_guessed_numbers
    global false_guessed_numbers
    global correct_guessed_numbers_batch
    
    on_index = actual_number
    delta_cost = []
    i = 0
    
    indi = []
    
    highest_neuron_index = -1
    highest_neuron_value = -1
    
    for neuron in network.indicator_neurons:
        #C(w,b)≡1/(2n)∑x∥y(x)−a∥2
        if on_index != actual_number:
            raise ValueError('on_index != actual_number')
        
        if i == on_index:
            #print "Master, please check weather on_index works correct;  on_index: ", on_index, "   |  actual_number: ", actual_number
            delta_cost.append(neuron.output_value - 1)
            #cost.append(1./2. * (1. - neuron.output_value)**2)
        else:
            #cost.append(1./2. * (0. - neuron.output_value)**2)
            delta_cost.append(neuron.output_value)
        
        #indi.append(neuron.output_value)
   
        if neuron.output_value > highest_neuron_value:
            highest_neuron_index = i
            highest_neuron_value = neuron.output_value
        
        i += 1
    
    #print "indi: ", indi
    #print "cost: ", cost
        
    print "            a_n: ", on_index, "  |   c n: ", highest_neuron_index
   
    if on_index == highest_neuron_index:
        correct_guessed_numbers += 1
        correct_guessed_numbers_batch += 1
    else:
        false_guessed_numbers += 1
    
    #pprint.pprint(delta_cost)
    return delta_cost

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


batch_bias_changes = []
correct_guessed_numbers = 0.  
false_guessed_numbers = 0.

correct_guessed_numbers_batch = 0.  


def train_network():
    global handwritten_numbers
    global actual_numbers
    global batch_bias_changes
    
    global correct_guessed_numbers_batch
    
    #plotter.new_variable('cost', 'b')
    #plotter.new_variable('hit_ratio', 'r')
    #plotter.new_variable('eta', 'g')
    
    smooth_plotter.new_variable('cost', 'b')
    smooth_plotter.new_variable('hit_ratio', 'r')
    #smooth_plotter.new_variable('eta', 'g')
    
    
    
    batchsize = 1.0
    
    eta = .02
    #input_manager.eta = eta
    #network = network()
    batchruns = 0
    
    total_cost = -1.
    current_delta_cost = []
    print 'DEBUG: running algorythm once, showing al relevant data of a layer 2 neuron'
    print 'neuron is layer2_neurons[3]'
    test_neuron = network.layer2_neurons[3]
    print 'start training'
    
    for batch_index in range(0, int(batchruns/batchsize)):
        #print '    start batchrun'
        for training_index in range(0, int(batchsize)):
            
            #total_index = int(batch_index*batchsize) + training_index
            total_index = randint(0, len(handwritten_numbers)-1)
            
            #display.draw_image(handwritten_numbers[total_index])
            
            
            #print '            correct number: ', actual_numbers[total_index], '  |  neuron bias: ', test_neuron.bias
            
            
            #print '\n        test network'
            network.test_network(handwritten_numbers[total_index])
            '''
            print ''
            print '            test_neuron-activation: ', test_neuron.batch_output_values[0]
            print ''UserWarning: Attempting to set identical bottom==top results
            in singular transformations; automatically expanding.
            bottom=0, top=0
            print '            layer1-activations: |  layer2-axon_weights: '
            for i in range(0, len(network.layer1_neurons)):
                l1_neuron = network.layer1_neurons[i]
                
                r_axon = test_neuron.r_axons[i]
                
                print '            %-23f%-19f' % (l1_neuron.batch_output_values[0],  r_axon.weight)
                #print '            ', l1_neuron.id, "  |  ", r_axon.id
            
            
            
            print ''
            print '\n        clauculate cost'
            print ''
            '''
            
            current_delta_cost.append(calculate_cost(network, actual_numbers[total_index]))
            
            
            #print ''
            #print '\n        claculate output error (errors of indiator neurons)'
            claculate_output_error(network, training_index, current_delta_cost[training_index])
            '''
            print ''
            print '            I-Neuron  Activation:  deltaCost: deriv_sigmoid:  z:'
            for i in range(0, len(network.indicator_neurons)):
                i_neuron = network.indicator_neurons[i]
                print'           ', i  , '        %-13f%-11f%-16f%-11f' % (i_neuron.batch_output_values[training_index], current_cost[training_index][i], deriv_sigmoid(i_neuron.stimulus_sum), i_neuron.stimulus_sum)
            
            
            
            
            
            print '\n        backprop'
            '''
            backprop(network, training_index)
            '''
            print ''
            print "            z: ", test_neuron.stimulus_sum, "  |  deriv_sigmoid: ", deriv_sigmoid(test_neuron.stimulus_sum), "  |  error neuron: ", test_neuron.errors[training_index]
            print ''
            print '                      weights f-axons    errors i-neurons'
            for i in range(0, len(network.indicator_neurons)):
                i_neuron = network.indicator_neurons[i]
                f_axon = test_neuron.f_axons[i]
                print'           ', i  , '        %-18f%-11f' % (f_axon.weight, i_neuron.errors[training_index])
            print ''
            
            
            
            print '\n        clear network_run'
            '''
            network.clear_network_run()
            
            
            
        #for neuron in network.indicator_neurons:
            #print neuron.bias
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
            
        #pprint.pprint(["    errors of layer1 neuron: ", network.layer1_neurons[0].errors])
        
        #print ''
        #print '\n        change values'
        network.change_values(eta, batchsize)
        #print ''
        #print '            t-neuron old_bias: ', test_neuron.old_bias, "  |  t-neuron new_bias: ", test_neuron.bias
        #print '            t-neuron errors'
        list = []
        for l in test_neuron.errors:
            list.append('          ' + str(l))
        #pprint.pprint(list)
        #print ''
        #print '\n        clear batchrun'
        network.clear_batchrun()

        total_cost = 0.0
    
        for c in current_delta_cost[int(batchsize) -2]:
            total_cost += c**2
        
        
        total_cost /= len(current_delta_cost[int(batchsize) -2]) *2   
        
        #plotter.update('cost', [total_cost], 25)
        #plotter.update('hit_ratio', [correct_guessed_numbers_batch/batchsize], 200)
        #eta += eta * 0.1
        
        smooth_plotter.update('cost', total_cost)
        smooth_plotter.update('hit_ratio', correct_guessed_numbers_batch/batchsize)
        #smooth_plotter.update('eta', eta)        
        
        
        #eta = input_manager.eta
        
        #print "                        eta: ", eta
        #plotter.update('eta', [eta], 1)
        
        
        
        
        correct_guessed_numbers_batch = 0
        #print '    end Batchrun'
        print "    total_cost: ", total_cost, "batch_index: ", batch_index
        #print '\n\n\n\n'        
        current_delta_cost = []
      
        
      
      
      
    
    
    print 'training end'   
    print 'correct_gessed_numbers: ', correct_guessed_numbers, '   |   false_gessed_numbers: ', false_guessed_numbers 
    path_name = ''.join(("network_states/network_cost: ", str(total_cost), " | ", datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"), ".pkl"))
    print 'save network as: ', path_name

    with open(path_name, 'wb') as f:
        pickle.dump('network', f, pickle.HIGHEST_PROTOCOL)

    
    
    
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
    print "expectation: ", float(correct_numbers)/ (correct_numbers + false_numbers)
         
    
    



        
#display = Display()        
network = network()  
train_network()
validate_network()      
#plotter.plt.show(block=True)
smooth_plotter.plt.show(block=True)
























'''class indicator_neuron():
    def __init__(self, id, axons):
        
        self.id = id
        self.stimulus_sum = 0
        self.bias = uniform(-10,10)
        self.output_value = -1
        self.axons = axons
        
    def stimulate(self, input_value):
        self.stimulus_sum += input_value
print "WARNING: z > 40:", 
    def stimulation_complete(self):
        total_output_value = float(self.stimulus_sum) + self.bias
        self.output_value = 1. / (1. + math.exp(-float(total_output_value)))
    
        

    def clear(self):
        self.stimulus_sum = 0
        self.output_value = 0
        
'''
        