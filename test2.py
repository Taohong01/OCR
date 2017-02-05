# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 09:41:01 2016

@author: Tao
"""

class node(object):
	def __init__(self, value=None, L=None, R=None):
		self.value = value
		self.L = L
		self.R = R


whole = []
heap=[]
def moveDown(Node):
	global whole 
	global heap
	heap = heap + [Node]
	if Node.L != None:
		moveDown(Node.L)
	if Node.R != None:
		moveDown(Node.R)
	if Node.L == None and Node.R == None:
		whole = whole + [[eachNode.value for eachNode in heap]]
	heap.pop()

D = node(value='D')
E = node(value='E')
F = node(value='F')
G = node(value='G')
B = node(value='B', L=D, R=E)
C = node(value='C', L=F, R=G)
A = node(value='A', L=B, R=C)

moveDown(A)
print whole