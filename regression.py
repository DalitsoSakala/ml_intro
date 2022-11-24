import numpy as np
import scipy.stats as stat
import random
from sklearn.metrics import r2_score, accuracy
import matplotlib.pyplot as plot

# data
x = random.sample( sorted(range(1, 20)), 20)
y = np.random.uniform(80, 80, 20)

# make polynomial
model = np.poly1d( n.polyfit(x, y, 3))
line  = np.linspace( min(x), max(x))

# function to map line
reg    = stat.linregress(x, y)
mapper = lambda x: x*reg.slope + reg.intercept

print('R-value of original data is %', (reg.rvalue,))
print('R2 score of scaled data is %', ( r2_score(x, list( map(mapper, x)) ),))

plot.plot(line, model(line))
plot.plot(x, list( map(mapper, x)))
plot.scatter(x, y)
plot.show()
