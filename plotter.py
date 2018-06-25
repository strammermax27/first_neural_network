import matplotlib.pyplot as plt
import numpy as np
import pprint

x = []#np.linspace(0, 6*np.pi, 100)
y = []#np.sin(x)


# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()
#plt.show()

fig = plt.figure()
axes = fig.add_subplot(111)


#line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

'''
for phase in np.linspace(0, 10*np.pi, 500):
    line1.set_ydata(np.sin(x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()
'''

variables = []
value_lists = []
averate_lists = []
lines = []

'''import matplotlib.pyplot as plt
import time
import random
 
ysample = random.sample(xrange(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()
 
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line, = axes.plot(xdata, ydata, 'r-')
 
for i in range(100):
    xdata.append(i)
    ydata.append(ysample[i])
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)

 
# add this if you don't want the window to disappear at the end
plt.show()
'''

#axes = plt.gca()

def new_variable(name, color):
    global variables
    global value_lists 
    global lines
    global averate_lists
    
    variables.append(name)
    
    list = []
    list2 = []
    value_lists.append(list)
    averate_lists.append(list2)
    
    lines.append(axes.plot(x, y, color + '-')[0])


max_x = 0
max_y = 0
min_y = 0

def update(name, value, bs_averange):
    global max_x
    global max_y
    global min_y

    
    i = 0
    index = -1
    
    for var in variables:
        if var == name:
            index = i
            
        i += 1
    
    value = smooth_data(value, index, bs_averange)
    
    
    value_lists[index].append(value)
    
    xs = []
    i = 0
    #axes.set_ylim(0, .5)
    
    for value in value_lists[index]:
        xs.append(i)
        
        if max_y < value:
            max_y = value + value * 0.05
            axes.set_ylim(min_y, max_y) 
        if min_y > value:
            min_y = value
            axes.set_ylim(min_y, max_y)
            
            
        i += 1
        
    if max_x < len(xs):
        max_x = len(xs)  
        axes.set_xlim(0, max_x)
    
    line  = lines[index]

    
    
    line.set_xdata(xs)
    line.set_ydata(value_lists[index])
    
    
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    
def smooth_data(value, index, bs_averange):
    global averate_lists
        
    
    
    averate_lists[index].extend(value)    
            
    av = 0.
    
    if len(averate_lists[index]) > bs_averange:
            print 'deleting'
            del averate_lists[index][0]
        
    for n in averate_lists[index]:
        av += float(n)
        
    av /= float(len(averate_lists[index]))
            
    
        
    #return value[0]
    return av








