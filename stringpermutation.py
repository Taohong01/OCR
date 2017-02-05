# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 20:22:22 2016

@author: Tao
"""

def permutation(s):
    #print s
    ss = []
    if len(s) > 1:
        for i in range(len(s)):
            c = s[i]
            sub = s[0:i] + s[(i+1):]
            #print sub
            ss = ss + [c+comb for comb in permutation(sub)]
            
    else:
        return [s,]
        
    return ss
        
print permutation('abc')

