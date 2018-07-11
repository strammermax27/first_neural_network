#from display_digits import Display
import mnist_loader as loader
import numpy as np
from numpy import random as np_random
import time


plot_cost = True
#import plotter
if plot_cost:
    import smooth_plotter
#import input_manager
#from statsmodels.tsa.interp.denton import indicator
#from conda.core import index

epochs = 300
batchsize = 10
eta = 0.18

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

errors_hl_2 = np.array([0.] * n2)
errors_hl_3 = np.array([0.] * n3)
errors_ol = np.array([0.] * 10)

errors_hl_2_sum = np.array([0.] * n2)
errors_hl_3_sum = np.array([0.] * n3)
errors_ol_sum = np.array([0.] * 10)    


a_2 = np.array([0.] * n2)
a_3 = np.array([0.] * n3)
a_o = np.array([0.] * 10)

input_sum = np.array([0.] * len(handwritten_numbers[0]))
a_2_sum = np.array([0.] * n2)
a_3_sum = np.array([0.] * n3)

def sigmoid(z):
            
    return 1. / (1. + np.exp(-z))

def deriv_sigmoid(z):
    
    return sigmoid(z)*(1.-sigmoid(z))

def calculate_errors(batch_index):
    
    global errors_hl_2_sum    
    global errors_hl_3_sum
    global errors_ol_sum
    
    global input_sum    
    global a_2_sum
    global a_3_sum

    errors_hl_2_sum = np.array([0.] * n2)
    errors_hl_3_sum = np.array([0.] * n3)
    errors_ol_sum = np.array([0.] * 10)    
    
    
    input_sum = np.array([0.] * len(handwritten_numbers[0]))    
    a_2_sum = np.array([0.] * n2)
    a_3_sum = np.array([0.] * n3)

    cost = 0 
        
    for index in range(batch_index, batch_index + batchsize):
        
        picture = handwritten_numbers[index]
        y = actual_numbers[index]
        
        #feedforward    
        z_2 = weights_hl2.dot(picture) + hidden_layer2
        a_2 = np.apply_along_axis(sigmoid, 0, z_2)
        z_3 = weights_hl3.dot(a_2) + hidden_layer3
        a_3 = np.apply_along_axis(sigmoid, 0, z_3)
        #z_o = np.sum(a_3 * weights_ol, axis = 1) + output_layer)    
        a_o = np.apply_along_axis(sigmoid, 0, (weights_ol.dot(a_3) + output_layer))
        

        input_sum += picture
        a_2_sum += a_2
        a_3_sum += a_3

            
        #calc cost for debugging
        if plot_cost:        
            cost_v = np.log((1- a_o))
            cost_v[y] = np.log(a_o[y])        
            cost -= np.sum(cost_v)/10
        
        #implement y
        a_o[y] -= 1
        
        
        #calc output_errors
        #errors_ol = current_a * np.apply_along_axis(deriv_sigmoid, 0, z_o)#doesn't work untli z_o is defined with quadratic cost funktion.
        errors_ol = a_o #with cross entropy cost funktion
            
        #backprop errors  
        #not 100% sure weather this is correct
        #errors_hl_3 = errors_ol.dot(weights_ol) * deriv_sigmoid(z_3)
        #errors_hl_2 = errors_hl_3.dot(weights_hl3)  * deriv_sigmoid(z_2)
        errors_hl_3 = weights_ol.transpose().dot(errors_ol) * deriv_sigmoid(z_3)
        errors_hl_2 = weights_hl3.transpose().dot(errors_hl_3) * deriv_sigmoid(z_2)           
    
    
        errors_ol_sum += errors_ol
        errors_hl_3_sum += errors_hl_3
        errors_hl_2_sum += errors_hl_2
        
    
    
    cost /= batchsize
    return cost

def gradient_descent():
    global output_layer
    global hidden_layer3
    global hidden_layer2
    
    global weights_ol
    global weights_hl3
    global weights_hl2   
    
    
    f = eta/batchsize    
    
    output_layer -= f * errors_ol_sum
    hidden_layer3 -= f * errors_hl_3_sum
    hidden_layer2 -= f * errors_hl_2_sum
   
    #print "errors_ol_sum.shape(): ", errors_ol_sum.shape  
    #print "a_3_sum: ", a_3_sum.reshape(-1,1)
   
    
    #the formula desn't work here; something has to be worng but i think this does the job anywa
    #weights_ol -= f * errors_ol_sum.dot((a_3_sum.reshape(-1,1)))
    #weights_hl3 -= f * errors_hl_3_sum * a_2_sum.reshape(-1,1)
    #weights_hl2 -= f * errors_hl_2_sum * input_sum.reshape(-1,1)
    weights_ol -= f * a_3_sum * errors_ol_sum.reshape(-1,1)
    weights_hl3 -= f * a_2_sum * errors_hl_3_sum.reshape(-1,1)
    weights_hl2 -= f * input_sum * errors_hl_2_sum.reshape(-1,1)

    
    

def run_training():
    global output_layer
    global hidden_layer3
    global hidden_layer2
    
    global weights_ol
    global weights_hl3
    global weights_hl2     
    
    #global eta
    #etas = []    
    
    #lowest_cost = 99    
    #best_eta = 0   
    
    training_examples = len(actual_numbers)  
    
    
    
    if plot_cost:
        smooth_plotter.new_variable("cost", "b")      
        p_cost = []
    current_cost = 0
    #smooth_plotter.new_variable("eta", "g")    
    
    for i in range(epochs):
        print "\n \n epoch: ", i   
        starttime = time.time() 
        for batch_index in range(0, training_examples, batchsize):
            
            #print batch_index
            
            current_cost = calculate_errors(batch_index)
            if plot_cost:    
                p_cost.append(current_cost)
            
                
                
            calculate_errors(batch_index)
            
            gradient_descent()
        
            
            
            if plot_cost and batch_index%2000 == 0:        
                smooth_plotter.update("cost", p_cost)
                p_cost = []
            
            
            #if current_cost < lowest_cost:
            #    lowest_cost = current_cost
            #    best_eta = eta
            #    print "new best eta: ", eta
                        
            
            #eta += eta * 0.00001
            
            #etas.append(eta)            
            
                    
        
        print "time needed for epoch: ", time.time() - starttime
        if plot_cost:        
            smooth_plotter.update("cost", p_cost)
            p_cost = []
        #smooth_plotter.update("eta", etas)        
                
        #print "current eta: ", eta
        validate_network()
    
    #print "best eta: ", best_eta
            

    
def validate_network():
    correct_numbers = 0
    false_numbers = 0
    
    validation_runs = len(validate_actual_numbers)
    
    
    print ''
    print 'start validation'
    print 'validation_runs :', validation_runs
    
    for i in range(1, validation_runs):
        
        
        picture = validate_handwritten_numbers[i]
        y = validate_actual_numbers[i]
        
        #feedforward    
        z_2 = np.sum(picture * weights_hl2, axis=1) + hidden_layer2
        a_2 = np.apply_along_axis(sigmoid, 0, z_2)
        z_3 = np.sum(a_2 * weights_hl3, axis = 1) + hidden_layer3
        a_3 = np.apply_along_axis(sigmoid, 0, z_3)
        #z_o = np.sum(a_3 * weights_ol, axis = 1) + output_layer)    
        a_o = np.apply_along_axis(sigmoid, 0, (np.sum(a_3 * weights_ol, axis = 1) + output_layer))
        
    
        highest_output_value = -1
        highest_output_index = -1
        output_index = 0
        for output_value in a_o:
                 
            #indi.append(neuron.output_value)
       
            if output_value > highest_output_value:
                highest_output_index = output_index + 0
                highest_output_value = output_value
            
            output_index += 1
           
        
        
        if y == highest_output_index:
            #print "CORRECT: a_n: ", y, "  |  c_n: ", highest_output_index
            correct_numbers += 1
        else:   
            #print "FALSE:   a_n: ", y, "  |  c_n: ", highest_output_index
            false_numbers += 1
         
    """        
    print "validation_complete"
    print "correct_numbers: ", correct_numbers, "  false_numbers: ", false_numbers
    print "correct_false_ratio: ", float(correct_numbers)/false_numbers
    
    print "validation_complete"
    print "correct_numbers: ", correct_numbers, "  false_numbers: ", false_numbers
    print "correct_false_ratio: ", float(correct_numbers)/false_numbers
    """
    print "validation_complete"
    print "correct_numbers: ", correct_numbers, "  false_numbers: ", false_numbers
    print "rate: ", float(correct_numbers)/validation_runs
         
    
    

run_training()     
#display = Display()        
#network = network()  
#train_network()
#validate_network()      
#plotter.plt.show(block=True)
smooth_plotter.plt.show(block=True)

