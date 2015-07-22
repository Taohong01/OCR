# pcaocr.py
# this package is an application of PCA for OCR.
# written by Tao Hong, 7/20/2015

#header files:
import cv2
import numpy as np
from sklearn import svm as sksvm 
from sklearn import cross_validation
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import pandas as pds
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA


#class for loading file
class OCR(object):
    def __init__(self, filename = 'digits.png'):        
        # import image file to variable img
        self.img = cv2.imread('digits.png',0)
        #self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        print 'original digits image dimensions:', self.img.shape
        # as we have seen from the print, the dimension is 1000 rows x 2000 columns

    def imageDicing(self, SZ=20):
        # each digit image size specification (based on the specific image we got):
        self.SZ = SZ
        # dice the image into small peices so that each small peiced contains
        # only one digit for training and identification 
        img = self.img
        self.cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
        print 'pick one cell and print the dimension', np.array(self.cells[0][0]).shape
        # as we have seen from the print out of the above command,
        # the dimension of each cell is composed of 20x20 pixels.
        # the original digits image is divided into an digit image array of 50 rows x 100 columns
        # as save as a list with a name of cells
        
        print len(self.cells[0])
        # as we can see from the above print, there are 100 elements for cells in row 0,
        # this is just a confirmation that there are 100 columns. 

    def showImage(self, img):
        #show the designated image
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # definition of 2 image preprocessing functions:
    # deskew() is for correct the tilting of each digits
    # hog() is for image edge orientation statistics
    def deskew(self, img, affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR):
        SZ = self.SZ
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
        img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
        return img
    
    def hog(self, img, bin_n = 16):
        # descrete number of the edge orientation angle, Number of bins
        SZ = self.SZ
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)     # hist is a 64 bit vector
        return hist
        
    def dataDividing(self):
        # divide the data into 2 sets:
        # First half is trainData, remaining is testData
        # in the future, we will use cross vallidation training-testing data
        # division function to process this
        self.train_cells = [ i[:50] for i in self.cells ]
        self.test_cells = [ i[50:] for i in self.cells]
        # for i in cells assigns each row of cells to i, so there are 50 in total, 
        # starting from 0, ending with 49. 
        # in each row, the data is divided into left and right two partion, 
        # each of them has 50 elements in a row.
        # the effective result is to break the original cells array at the middle
        # into left and right. Left will be used for training and right will be
        # used for testing. So the test train_cells is a list array of 50 x 50 images.
        # similar for the test_cells. 
        
        
        
        # to find out how the data are splitted, we print out their dimensions
        print 'training data total row number is', len(self.train_cells)
        print 'training data total column number is', len(self.train_cells[0])
        
        
        
    def myPCA(self):
        #gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        gray = self.img

        # Now we split the image to 5000 cells, each 20x20 size
        cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
        
        # Make it into a Numpy array. It size will be (50,100,20,20)
        x = np.array(cells)
        
        # Now we prepare train_data and test_data.
        train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
        test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
        
        nn =1000
        X = np.array(train[(nn):(nn+250),:])
        print 'data X\n', X.shape
        num_componets =70
        pca = PCA(n_components=num_componets)
        pca.fit(X)
        #print 'pca components', pca.components_
        #print 'pca explained variance ratio', pca.explained_variance_ratio_
        
        img0 = cv2.resize(pca.mean_.reshape(20,20), (0,0), fx =5, fy=5)
        for ii in range(num_componets):
            img1 = cv2.resize(pca.components_[ii,:].reshape(20,20), (0,0), fx =5, fy=5)
            img0 = np.hstack((img0, img1))
        #img1 = cv2.resize(pca.components_[0,:].reshape(20,20), (0,0), fx =10, fy=10)
        #img2 = cv2.resize(pca.components_[1,:].reshape(20,20), (0,0), fx =10, fy=10)
        #img3 = cv2.resize((pca.components_[0,:]+pca.components_[1,:]+pca.components_[2,:]+pca.components_[3,:]+pca.components_[4,:]).reshape(20,20), (0,0), fx =10, fy=10)
        cv2.imshow('',img0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print 'pca n_components', pca.n_components_
        
        print(sum(pca.explained_variance_ratio_))
                
def main():
    DigitOCR = OCR()
    DigitOCR.imageDicing()
    DigitOCR.dataDividing()
    #DigitOCR.showImage(DigitOCR.img)
    DigitOCR.myPCA()
        
if __name__ == '__main__':
    main()
    
    
    
    
        
        
        