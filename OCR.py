import cv2
import numpy as np
from sklearn import svm as sksvm 

# each digit image size specification (based on the specific image we got):
SZ=20

# descrete number of the edge orientation angle
bin_n = 16 # Number of bins


# Support Vector Machine parameter specification:
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

# definition of 2 image preprocessing functions:
# deskew() is for correct the tilting of each digits
# hog() is for image edge orientation statistics
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

# import image file to variable img
img = cv2.imread('digits.png',0)
print 'original digits image dimensions:', img.shape
# as we have seen from the print, the dimension is 1000 rows x 2000 columns

# dice the image into small peices so that each small peiced contains
# only one digit for training and identification 
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
print 'pick one cell and print the dimension', np.array(cells[0][0]).shape
# as we have seen from the print out of the above command,
# the dimension of each cell is composed of 20x20 pixels.
# the original digits image is divided into an digit image array of 50 rows x 100 columns
# as save as a list with a name of cells

print len(cells[0])
# as we can see from the above print, there are 100 elements for cells in row 0,
# this is just a confirmation that there are 100 columns. 


# randomly pick one image dice and show it with its descrewed image side by side
imgc = cells[30][50]
imgc1 = deskew(imgc)
imgc = np.hstack([imgc1,imgc])
cv2.imshow('',imgc)

#cv2.waitKey(10000)
cv2.waitKey(0) 
# in mac to turn of the pop up window, we need to click and select
# the window and then click esc to close the window and the python program will
# continue to run the next step. The waitKey(0) still give a pause.
cv2.destroyAllWindows()
#cv2.waitKey(1)



# divide the data into 2 sets:
# First half is trainData, remaining is testData
# in the future, we will use cross vallidation training-testing data
# division function to process this
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]
# for i in cells assigns each row of cells to i, so there are 50 in total, 
# starting from 0, ending with 49. 
# in each row, the data is divided into left and right two partion, 
# each of them has 50 elements in a row.
# the effective result is to break the original cells array at the middle
# into left and right. Left will be used for training and right will be
# used for testing. So the test train_cells is a list array of 50 x 50 images.
# similar for the test_cells. 



# to find out how the data are splitted, we print out their dimensions
print 'training data total row number is', len(train_cells)
print 'training data total column number is', len(train_cells[0])


######     Now training      ########################

deskewed = [map(deskew,row) for row in train_cells]
hogdata = [map(hog,row) for row in deskewed]

print 'hogdata row number is ', len(hogdata)
print 'hogdata column number is', len(hogdata[0])
# as we have seen, the hogdata still have the shape of 50 x 50, although now 
# each element is changed into 64-element edge orientation angle statistical data.

"""Below is a simple test for showing how list in generated with *
testarray = [[i]*10 for i in range(5)]
print 'list generation test', testarray"""


trainData = np.float32(hogdata).reshape(-1,64)
print 'trainData dimensions:', trainData.shape
# generate y labels for the hand writing digits.
# according to the arrangement in the original image, 
# we know that there are 5000 samples in total
# and 500 samples for each digits (0~9)
# in the training set, we have 250. 
# in the testing set, we have the other 250. 
# base on this information and the order of digits
# showing up in the training and testing test,
# we could generate correct labels for the images, as follows
responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])
# np.float32 is to convert the data format that can be accepted by opencv C code
# np.arange(10) is to generate 0~9 arrange
# np.repeat is to repeat()
#  

# below is to adapter data format to the scikit learn SVM format
# so that we can use it to train sklearn package svm and make comparison
# also we could later use cross-validation etc other packaged tools for ML.
X_train = np.array(trainData)
y_train = np.array(responses.ravel())
#below just double check to see if the data formats are right or not.
print 'y_train dimensions', y_train.shape
print 'confirm the data format are right: ', X_train.shape, y_train.shape



# Initiate an opencv svm instance and proceed training
svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)
svm.save('svm_data.dat')

################################################################
# Initiate a scikit learn svm and proceed model training

"""
# this is a preliminary test of the svm model training
clf = sksvm.SVC(C=3.5, gamma=15.383 , kernel='linear')
clf.fit(X_train, y_train)
#print 'sklearn svm support vector original index:', clf.support_
print 'number of support vectors for each class', clf.n_support_
print 'settings of the svm', clf
"""
from sklearn import cross_validation
from sklearn import metrics

clf = sksvm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5, scoring='recall')

print 'recall scores of cross validation', scores   



######     Now testing      ########################

imgcc=np.vstack(np.hstack(test_cells))
"""
cv2.imshow('',imgcc)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
deskewed = [map(deskew,row) for row in test_cells]
hogdata = [map(hog,row) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict_all(testData)
print result.ravel()

X_test = np.array(testData)
y_test = np.array(responses.ravel())
####clfresult = clf.predict(X_test)
#print 'sklearn svm result:',clfresult-result


#######   Check Accuracy   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print correct*100.0/result.size