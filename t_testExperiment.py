# t_testExperiment.py
"""
Created on Mon Aug 17 19:03:53 2015

@author: Tao
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def sampleGenerator(mean, std, size):
    sample = np.random.normal(mean, std, size)
    return sample


def sampleFromSameOrigin(mean, std, size1, size2):
    sample1 = sampleGenerator(mean, std, size1)
    sample2 = sampleGenerator(mean, std, size2)
    return (sample1, sample2)

def sampleFrom2Origin(mean1, std1, size1, mean2, std2, size2):
    sample1 = sampleGenerator(mean1, std1, size1)
    sample2 = sampleGenerator(mean2, std2, size2)
    return (sample1, sample2)

def sampleSize_Mean_Variance(s1):
    size_s1 = len(s1)
    mean_s1 = sum(s1)/size_s1
    # the calculation of variance is a different for a population and a sample 
    # in the case of whole population, var = sum((s1-mean)**2)/n, 
    # but in the case of sample of a population, var = sum((s1-mean)**2)/(n-1)
    variance_s1 = sum((s1 - mean_s1)**2)/(size_s1 - 1) 
    return (size_s1, mean_s1, variance_s1)

def tValueOf2Samples(s1,s2):
    n1, mean_s1, var_s1 = sampleSize_Mean_Variance(s1)
    n2, mean_s2, var_s2 = sampleSize_Mean_Variance(s2)
    t = (mean_s2 - mean_s1) / np.sqrt(var_s1/n1 + var_s2/n2)
    return t


def t_test(s1,s2, tailnumber = 2, pvalue =0.05):
    tvalue = tValueOf2Samples(s1,s2)
    tc = stats.t.ppf(1-pvalue/tailnumber, len(s1)+len(s2)-2)
    if abs(tvalue) >= tc:
        print 'the null hypothesis should be rejected'
        return (tvalue, tc, False)
    else:
        print 'It is insufficient to reject the null hypothesis'
        return (tvalue, tc, True)

    
# test: 
# in the test below, we generate 2 sets of samples, repeatedly. 
#    (from 10 samples, up to 300) 
# by control the number of samples in each set and the center and 
# standard deviation of each set, we want to show if a null hypothesis 
# can be sucessfully rejected depends on the number of samples as well as 
# the actually distinguishability of the two original population for sampling.
# i.e., the centers of the two normal distributions. 
tr = np.zeros((300,3))
for i in range(10, 300):
    s1,s2 = sampleFrom2Origin(10,16,i, 12.5,6, i)
    tr[i,] = t_test(s1,s2)
    
    
plt.plot(tr[:,0])
plt.plot(tr[:,1])

plt.xlabel('number of one set of samples')
plt.ylabel('t value and tc value')
plt.show()

#print  t_test(s1,s2)

