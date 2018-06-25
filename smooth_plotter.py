import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
#from numba import jit, jitclass
#from numba import int32, float32  
import numpy as np
import pprint
import thread
import time
from PIL.ImageFilter import SMOOTH



'''spec = [
    ('values', float32),               
    ('smooth_values', float32),        
    ('xs', int32),
    #('color', string),
    #('name', string)  
]



'''
#@jitclass(spec)
class graph(object):
    
    def __init__(self, name, color):
        self.values = []
        self.smooth_values = []
        
        self.xs = []
        for i in range(0,max_plt_points):
            self.xs.append(i)
            self.smooth_values.append(i)
        
        self.color = color
        self.name = name
        
        x = []
        y = []
        
        self.line = axes.plot(x, y, color + '-')[0]
        
        
        
        
    
        
        
        
        
def new_variable(name, color):
    global graphs
    graphs.append(graph(name, color))
    


def max_plt_points_update(val):
    global max_plt_points
    global graphs
    global max_y
    global min_y
    
    
    max_plt_points = int(val)
    
    axes.set_xlim(0, max_plt_points)
    #fig.canvas.draw()
    #fig.canvas.flush_events()  


    
    for graph in graphs:      #try to make this a jit loop
        
        graph.xs = []
        graph.smooth_values = []
        for i in range(0,max_plt_points):
            graph.xs.append(i)
            graph.smooth_values.append(i)
        
        
        graph.smooth_values = smooth_data(graph.values)
        
        
        
        for value in graph.smooth_values:    
            if max_y < value:
                max_y = float(value) + float(value) * 0.1
            if min_y > value:
                min_y = value





        axes.set_ylim(min_y, max_y)
        axes.set_xlim(0, max_plt_points)
                
                
        line  = graph.line
        
        line.set_xdata(graph.xs)
        line.set_ydata(graph.smooth_values)
        
    fig.canvas.draw()
    fig.canvas.flush_events()  




#@jit
def run():
    global graphs
    global max_y
    global min_y
   
    
   
    
    
    while 1:

        max_y = 0
        min_y = 0

        for graph in graphs:      #try to make this a jit loop
            graph.smooth_values = smooth_data(graph.values)
            
            
            
            
            for value in graph.smooth_values:    
                if max_y < value:
                    max_y = float(value) + float(value) * 0.1
                if min_y > value:
                    min_y = value
        
        
        
        
        
                
            #print graph.smooth_values
          
        time.sleep(10)
        #print "tack"
        





max_y = 0
min_y = 0

def update(name, value):
    global max_y
    global min_y

    
    i = 0
    index = -1
    
    for g in graphs:
    
        if g.name == name:
            index = i
            
        i += 1
    
    graph = graphs[index]
    graph.values.append(value)
    
    
    
    axes.set_ylim(min_y, max_y)
            
            
    line  = graph.line
    
    line.set_xdata(graph.xs)
    line.set_ydata(graph.smooth_values)
    
    fig.canvas.draw()
    fig.canvas.flush_events()  
    
#@jit    
def smooth_data(values):
    global averate_lists
        
    smooth_data = [[]] * max_plt_points
    conv_smooth_data = [0] * max_plt_points
    
    ''' if max_plt_points > len(values):
        for i in range(0, max_plt_points):
            if i < len(values):
                conv_smooth_data[i] = values[i]
            else:
                conv_smooth_data[i] = 0
        
        
                
    else:          '''
    interest = (float(len(values)) / float(max_plt_points))
    
    #print "interest: ", interest
    bills = 0

    i = 0
    a_csd = [] 
    for s in conv_smooth_data:
        bills += interest
        #print "bills: ", bills
        
        
        sum = 0.
        csm_i = 0.
        value_count = 0
        while bills >= 1:
            bills -= 1
            i += 1
            csm_i += 1.
            
            if i < len(values):
                sum += values[i]
                value_count += 1
            #else:
            #    print "WARNING i > len(values), i: ", i, "   len(values): ", len(values)
            
            
            
        if csm_i == 0:
            csm_i = -1    
            
        if value_count == 0:    
            value_count = -1
        
        s = sum/value_count
        #s = .1
        a_csd.append(s)
    
    
        ''' for data in smooth_data:
        sum = 0.
        i = 0.
        
        #if isinstance(data, (list,)):
        for v in data:
            sum += v 
            i += 1
        
        if sum == 0:
            conv_smooth_data.append(0)
        else:
            conv_smooth_data.append(sum/i)    
        #else:
        #    conv_smooth_data.append(data)   
        '''
    conv_smooth_data = a_csd
    
 
        
        
    return conv_smooth_data


print 1
graphs = []
max_plt_points = 100


 # You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()


fig = plt.figure()
axes = fig.add_subplot(111)
print 2
axes.set_xlim(0, max_plt_points)

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.2, 0.02, 0.65, 0.03], axisbg=axcolor)
sfreq = Slider(axfreq, 'max_plt_points', 1, 1000, valinit=max_plt_points)

sfreq.on_changed(max_plt_points_update)
print 3
fig.canvas.draw()
fig.canvas.flush_events()  



thread.start_new_thread(run, () )
print 4








