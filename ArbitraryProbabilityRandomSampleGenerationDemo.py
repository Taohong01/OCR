import bisect
import random
import matplotlib.pylab as plt


# for i in range(100):
#     a = random.uniform(0,1)
#     print a
#     print bisect.bisect_left([0.1,.2,.5,.55,.57,.9, 1.0], a)

def func(x):
    return np.exp(-x ** 4) * np.abs(np.sin(x+.5))



import numpy as np

x = np.linspace(-10, 10, 400)

state = []
for eachx in x:
    state.append(func(eachx))
statesum = sum(state)
print statesum
# note although x stands for spatial coordinates
# we are not considering spatial integral.
# the proabability distribution is not a probability
# density ditribution. So every x just stands for
# one spot in space.
# when we sum up the total probability, we just
# sum up all the item in the state
# to make a normalization of the total probability,
# we make a division below
state = np.array(state)/statesum

print 'after normalization, we get the total probabilty of 1 as shown by', sum(state)

plt.plot(x, state, 'o')

# now we calculate the probability accumulation below:
accum = 0
paccum = []
for p in state:
    accum += p
    paccum.append(accum)



#plt.plot(x,paccum,'x')
#plt.show()


# generate random x samples and then plot the histogram to show
#its probability distribution is identical with the probability
# given the state.
xsample = []
for i in range(1000000):
    a = random.uniform(0,1)
    xsample.append(x[bisect.bisect_left(paccum, a)])

plt.hist(xsample,bins=100)
plt.show()
