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
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
from scipy import signal


import logging
from time import time

from numpy.random import RandomState


from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition


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

        # Now we prepare train_data and test_data.
        train = np.array(self.train_cells)[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
        test = np.array(self.test_cells)[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
        print 'looks good'
        nn =0
        X = np.array(train[(nn):(nn+250),:])
        print 'data X\n', X.shape
        num_componets =10
        pca = PCA(n_components=num_componets)
        pca.fit(X)
        #print 'pca components', pca.components_
        #print 'pca explained variance ratio', pca.explained_variance_ratio_
        
        img0 = cv2.resize(pca.mean_.reshape(20,20), (0,0), fx =5, fy=5)
        img0 = np.zeros(img0.shape)

        for ii in range(num_componets):
            img1 = cv2.resize(pca.components_[ii,:].reshape(20,20), (0,0), fx =5, fy=5)
            img0 = np.hstack((img0, img1))
            #img1 = cv2.resize(pca.components_[0,:].reshape(20,20), (0,0), fx =10, fy=10)
            #img2 = cv2.resize(pca.components_[1,:].reshape(20,20), (0,0), fx =10, fy=10)
            #img3 = cv2.resize((pca.components_[0,:]+pca.components_[1,:]+pca.components_[2,:]+pca.components_[3,:]+pca.components_[4,:]).reshape(20,20), (0,0), fx =10, fy=10)
        self.showImage(img0*10)

        print 'pca n_components', pca.n_components_
        
        print(sum(pca.explained_variance_ratio_))
        
    def printALine(self):
        print '-----------------------------------------------------'
        
        
        
    ###############################################################################
    def plotGallery(self, title, images, image_shape=(64,64), n_col=3, n_row=2):
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)

        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                       interpolation='nearest', vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        plt.show()

    ###############################################################################

    def myICA(self):
        # Now we prepare train_data and test_data.
        train = np.array(self.train_cells)[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
        test = np.array(self.test_cells)[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
        print 'looks good'
        nn =1000
        n_samples = 250
        X = np.array(train[(nn):(nn+n_samples),:])
        print 'data X shape is ', X.shape
        # global centering
        X_centered = X - X.mean(axis=0)
        # local centering
        X_centered -= X.mean(axis=1).reshape(n_samples,-1)
        num_componets = 60
        ica = FastICA(n_components = num_componets, whiten = True)
        newX_centered = ica.fit(X_centered).transform(X_centered)
        print 'new X centered shape is ', newX_centered.shape

      
        img0 = cv2.resize(ica.mean_.reshape(20,20), (0,0), fx =5, fy=5)
        
        self.plotGallery('ica original test image set', X[0:6,:], image_shape=(20,20), n_row=2,n_col=3)
        self.plotGallery('ica components', ica.components_[0:6,:], image_shape=(20,20), n_row=2,n_col=3)
        
        sumImage = np.dot(newX_centered, ica.components_) \
        #+ X.mean(axis=1).reshape(n_samples,-1) +X.mean(axis=0)
        print newX_centered.shape
        print ica.components_.shape
        print sumImage.shape
        
        self.plotGallery('ica sum images', sumImage[0:6,:], image_shape=(20,20), n_row=2,n_col=3)
        self.printALine()

        
    def test(self):
        ###############################################################################
        # Generate sample data
        np.random.seed(0)
        n_samples = 2000
        time = np.linspace(0, 8, n_samples)
        
        s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
        s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
        
        S = np.c_[s1, s2, s3]
        S += 0.2 * np.random.normal(size=S.shape)  # Add noise
        
        S /= S.std(axis=0)  # Standardize data
        # Mix data
        A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
        X = np.dot(S, A.T)  # Generate observations
        
        # Compute ICA
        ica = FastICA(n_components=3)
        S_ = ica.fit_transform(X)  # Reconstruct signals
        A_ = ica.mixing_  # Get estimated mixing matrix
        
        # We can `prove` that the ICA model applies by reverting the unmixing.
        assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
        
        # For comparison, compute PCA
        pca = PCA(n_components=3)
        H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
        
        ###############################################################################
        # Plot results
        
        plt.figure()
        
        models = [X, S, S_, H]
        names = ['Observations (mixed signal)',
                 'True Sources',
                 'ICA recovered signals', 
                 'PCA recovered signals']
        colors = ['red', 'steelblue', 'orange']
        
        for ii, (model, name) in enumerate(zip(models, names), 1):
            plt.subplot(4, 1, ii)
            plt.title(name)
            for sig, color in zip(model.T, colors):
                plt.plot(sig, color=color)
        
        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
        plt.show()   
        
        
    def test2(self):
        

        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        n_row, n_col = 2, 3
        n_components = n_row * n_col
        image_shape = (64, 64)
        rng = RandomState(0)
        
        ###############################################################################
        # Load faces data
        dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
        faces = dataset.data
        
        n_samples, n_features = faces.shape
        
        # global centering
        faces_centered = faces - faces.mean(axis=0)
        
        print 'faces_centered has %d dimensions: ', faces_centered.shape
        
        # local centering
        faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
        
        print("Dataset consists of %d faces" % n_samples)
        print("each face has %d features" % n_features )
        
        # List of the different estimators, whether to center and transpose the
        # problem, and whether the transformer uses the clustering API.
        estimators = [

            ('Independent components - FastICA',
             decomposition.FastICA(n_components=n_components, whiten=True),
             True),
             
        ]
        
        
        ###############################################################################
        # Plot a sample of the input data
        
        self.plotGallery("First centered Olivetti faces", faces_centered[:n_components])

        ###############################################################################
        # Do the estimation and plot it
        
        for name, estimator, center in estimators:
            print("Extracting the top %d %s..." % (n_components, name))
            t0 = time()
            data = faces
            if center:
                data = faces_centered
            estimator.fit(data)
            train_time = (time() - t0)
            print("done in %0.3fs" % train_time)
            if hasattr(estimator, 'cluster_centers_'):
                components_ = estimator.cluster_centers_
            else:
                components_ = estimator.components_
            if hasattr(estimator, 'noise_variance_'):
                self.plotGallery("Pixelwise variance",
                             estimator.noise_variance_.reshape(1, -1), n_col=1,
                             n_row=1)
            self.plotGallery('%s - Train time %.1fs' % (name, train_time),
                         components_[:n_components])
        
        plt.show()
                
def main():
    DigitOCR = OCR()
    DigitOCR.imageDicing()
    DigitOCR.dataDividing()
    #DigitOCR.showImage(DigitOCR.img)
    DigitOCR.myICA()
    #DigitOCR.test2()
    
if __name__ == '__main__':
    main()
    
    
    
    
        
        
        