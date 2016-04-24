# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:29:58 2016

@author: Tao
"""
#from xgoogle.search import GoogleSearch
"""
from bs4 import BeautifulSoup
import requests
from termcolor import colored 
resp = requests.get('http://www.trulia.com/real_estate/Pleasanton-California/market-trends/')
soup =  BeautifulSoup(resp.text, 'html.parser')
print soup.prettify()

print colored('------------------------------', 'red')
ss = soup.findAll( text = 'Median')
print ss
"""
"""
for str1 in soup.text.split(' '):  
    if str1.find('25.19') > 0:
        print [str1,]
        #print 'ok'
"""        
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt

resp = requests.get('http://www.trulia.com/real_estate/Seattle-Washington/market-trends/')
#resp = requests.get('http://www.trulia.com/real_estate/Pleasanton-California/market-trends/')
soup =  BeautifulSoup(resp.text, 'html.parser')
import ast
CONTs = soup.body.contents

index = 0
for cont in CONTs:
    print index
    print cont
    index = index + 1
    
soup2string = CONTs[31].string
#print soup2string

ss = soup2string[24:].split(';')




soupitems = ss[0].split('\n')

index = 0
for s in soupitems:
    print index
    print s
    print 
    print
    index = index + 1

print '----------------------------------------------------------------------'
newstr = ast.literal_eval(soupitems[1][27:-1])
print 'the length of newstr is :  ', len(newstr)
print 'the elements in newstr is : \n'

for item in newstr:
    print item
    print

#newstr = ast.literal_eval(soupitems[1][27:-1])

#print soup2string[24:].split(';')[0].split('\n')[4][27:100]

import pandas as pd
#note: the index of newstr[i] controls the number of bedrooms
PD1bed = pd.DataFrame(newstr[3]['points'])
#print PD1bed
pddate = pd.to_datetime(PD1bed.date)
pdvalue = pd.to_numeric(PD1bed.value)
#print pddate
plt.plot(pddate, pdvalue,'-')
plt.show()
