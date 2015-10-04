import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl


X = np.array(AtomFringes1['y']*SingleCs1.LengthUnit[0]*1000)
#X = np.atleast_2d(AtomNumberVsFringeVsPosy1['y']*SingleCs1.LengthUnit[0]*1000).T
#print 'X=', X

#print AtomNumberVsFringeVsPosy1
# Observations
y = np.atleast_2d(AtomFringes1['|1, 0hbarkeff>']).ravel()

DataForSave=pd.DataFrame(X.T, columns=['X',])
DataForSave['y']= y.T
DataForSave.to_csv('Data2hbark.dat')
#print DataForSave


import numpy as np
from scipy.optimize import curve_fit
def func(x, a, b, c, d, e):
    return a*np.exp(-b * x**2) * (np.sin(c*x+d) +e)

xxdata = np.linspace(-20, 20, 1000)
xdata=np.array( X.T)
#y = func(xdata, 2.5, .3, 10, 2, 3)
#ydata = y + 0.2 * np.random.normal(size=len(xdata))
ydata = np.array(y)
print xdata
close()
pl.plot(xdata, ydata, 'o')
popt, pcov = curve_fit(func, xdata, ydata, p0=[1.91, .0001, 1.5, 3.14, .5])
print popt
perr = np.sqrt(np.diag(pcov))
pl.plot(xxdata, func(xxdata, popt[0], popt[1], popt[2], popt[3], popt[4]), linewidth = 3.0)
pl.xlabel('Position (mm)', fontsize = 20)
pl.ylabel('Fringe (arb.)', fontsize = 20)
#pl.setp(lines, color='r', linewidth=2.0)
pl.xticks(fontsize = 15)
pl.yticks(fontsize = 15)
print 'parameter:', popt
print 'SNR:', 1/perr

FitParameters = pd.DataFrame(popt, columns=['p',])
FitParameters['std']= perr
FitParameters['SNR'] = 1/perr
#FitParameters.index = ['abcde']
print FitParameters

FitParameters.to_csv('parameters.dat')

#std1 = np.sqrt(np.sum((y-func(X, popt[0], popt[1], popt[2], popt[3], popt[4]))**2)/X.size)
#print 'standard deviation:', std1
#pl.errorbar(X, y, yerr=std1)

pl.show()