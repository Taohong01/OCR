# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:19:09 2015

@author: Tao
"""
# this program is to do a quick sorting on a input list by comparing 
# with a given value 
# les's assume the given value is 3 for now
# and the input list is [3, 2, 1, 4, 5]
GivenValue = 3
InputList = [4, 5, 6, 2, 3, 2, 1, 0, 9]
def quickSorting(inputList=InputList, givenValue=GivenValue):
    length = len(inputList)
    smallerList = []
    largerList = []
    
    for anelement in inputList:
        if anelement > givenValue:
            largerList.append(anelement)
        else:
            smallerList.append(anelement)
    print smallerList
    print largerList
    finalList = smallerList + largerList
    
    return finalList
    
def swap(a, b):
    return (b, a)
    
def quickSorting2(inputList=InputList):
    pivotIndex = 0
    length = len(inputList)
    for i in range(length):
        if inputList[i] < inputList[pivotIndex]:
            inputList[pivotIndex], inputList[i] = swap(inputList[pivotIndex], inputList[i])
            pivotIndex = i
        print inputList
    return inputList
    
    
    
def main():
    print 'input list is :', InputList
    sortedList = quickSorting2()
    print 'sorted list is', sortedList
    x1, x2 = 0, 2
    x1, x2 = swap(x1, x2)
    
    print x1, x2
if __name__=='__main__':
    main()
    
    
    