import streamlit as st
import cv2
from PIL import Image
from io import BytesIO
import numpy as np

class smoothingFiltes():
    def __init__(self,image,K = 3):
        self.k = K
        self.image = image
    
    def trimmedMean(self,array):
        trimGrayscale = 1
        flattenarray = array.flatten()
        sortedArray = sorted(flattenarray)
        arrayLength = len(array)
#         print(sortedArray)
        trimmedArray = sortedArray[trimGrayscale:arrayLength-trimGrayscale]
        trimmedMeanValue = np.mean(trimmedArray)/(arrayLength-2*trimGrayscale)
        return trimmedMeanValue

    def inverseGradient():
        pass
    
    def imagePadding(self,band,paddingValue):
        ##vertical Top padding
        band = np.insert(band,0,band[0]*np.ones(paddingValue)[:,None],axis = 0)
        ##vertical bottom padding
        band =  np.insert(band,len(band),band[-1]*np.ones(paddingValue)[:,None],axis = 0)
        ##Horizontal left padding
        band = np.insert(band, 0, band[:,0]*np.ones(paddingValue)[:,None], axis=1)
        ##Horizontal right padding
        band = np.insert(band, len(band[0]), band[:,-1]*np.ones(paddingValue)[:,None], axis=1)
        return band
    
    def convolution(self,band):
        if (self.k % 2 == 0):
            raise Exception("K should be odd number.")
            
        paddingValue = int((self.k - 1)/2)
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        outputband = np.zeros(shape=arrayshape)
        
        for rowNo in range(arrayshape[0] - paddingValue):
            for colNo in range(arrayshape[1] - paddingValue):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.k,colNo:colNo + self.k]
#                 print(neighbourhoodMatrix)
                trimmedValue = self.trimmedMean(neighbourhoodMatrix)
                outputband[rowNo:rowNo + paddingValue,colNo:colNo + paddingValue] = trimmedValue
        return outputband
    
    def display(self,outputimage):
#         plt.figure()
#         plt.imshow(outputimage) 
#         plt.show()
        st.image(outputimage[0], caption=f"Processed image", use_column_width=True,)
        # cv2.imwrite(str('aniket1'+'.'+'jpg'),)
    
    def trimmedSmoothing(self):
        outputimage = []
        img = self.image
        for bandNumber in range(img.shape[2]):
            bandarray = np.array([array[:,bandNumber] for array in img])
            outputband = self.convolution(bandarray) 
            outputimage.append(outputband)
        self.display(outputimage)
        return outputimage

    def inverseGradientSmoothing(self,P):
        pass
        

Algorithm = st.sidebar.selectbox('Which algorithm do you want to implement?', ['Trimmed Mean Filter','Inverse Gradient Filter'])
st.write("Image smoothing filter GUI")

inputimagefile = st.file_uploader('Upload the image',type = ['png','jpeg','jpg'])
show_file = st.empty()

if not inputimagefile:
    show_file.info('Please upload the image')
if isinstance(inputimagefile,BytesIO):
    imagefile = inputimagefile.read()
    st.image(imagefile, caption=f"Input image", use_column_width=True)
    
    pil_image = Image.open(inputimagefile)
    img_array = np.array(pil_image)


    if Algorithm == 'Trimmed Mean Filter':
            K_trim = st.sidebar.slider('Select size of convolution matrix: n x n',3, 9)
            ourfilter = smoothingFiltes(img_array,K=K_trim)
            outputimage = ourfilter.trimmedSmoothing()

    if Algorithm == 'Inverse Gradient Filter':
            K_inverse = st.sidebar.slider('Select size of convolution matrix: n x n ',3, 9,3)
            P = st.sidebar.slider('Select weight assigned to center pixel (p)',0.0, 1.0,0.5)
            ourfilter = smoothingFiltes(img_array,K=K_inverse)
            outputimage = ourfilter.inverseGradientSmoothing(P)