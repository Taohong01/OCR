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
print img

# dice the image into small peices so that each small peiced contains
# only one digit for training and identification 
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# randomly pick one image dice and show it with its descrewed image side by side
print len(cells[0])

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




# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

######     Now training      ########################

deskewed = [map(deskew,row) for row in train_cells]
hogdata = [map(hog,row) for row in deskewed]

print 'hogdata', len(hogdata)

trainData = np.float32(hogdata).reshape(-1,64)
print 'trainData dimensions:', trainData.shape
responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])
X_train = np.array(trainData)
y_train = np.array(responses.ravel())
print 'y_train dimensions', y_train.shape
print X_train.shape, y_train.shape

svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)
svm.save('svm_data.dat')

clf = sksvm.SVC(C=3.5, gamma=15.383 , kernel='linear')
clf.fit(X_train, y_train)
print 'sklearn svm support vector original index:', clf.support_
print 'number of support vectors for each class', clf.n_support_
print 'settings of the svm', clf

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
clfresult = clf.predict(X_test)
print 'sklearn svm result:',clfresult-result


#######   Check Accuracy   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print correct*100.0/result.size