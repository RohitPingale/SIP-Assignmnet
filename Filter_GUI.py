import streamlit as st

import cv2
from PIL import Image

from io import BytesIO
import numpy as np

class smoothingFilters():
    def __init__(self,image,K = 3, P= None):
        self.k = K
        self.image = image
        self.P = P
        self.outputimage = None

    def trimmedMean(self,array):
        trimGrayscale = 1
        flattenarray = array.flatten()
        sortedArray = sorted(flattenarray)
        arrayLength = len(array)
#         print(sortedArray)
        trimmedArray = sortedArray[trimGrayscale:arrayLength-trimGrayscale]
        trimmedMeanValue = np.mean(trimmedArray)/(arrayLength-2*trimGrayscale)
        return trimmedMeanValue

    def inverseGradient(self,array,P):
        array=np.array(array, dtype=float)
        final_image=[]
        filter_array = array.flatten()
        flattenarray = array.flatten()
        #print(flattenarray)
        value_of_p=P
        center_pos=0
        
        if(len(flattenarray)%2==0):
            center_pos=int(len(flattenarray)/2)
        else:
            center_pos=int((len(flattenarray)-1)/2)
        #print(flattenarray.shape)
        
        #if(flattenarray[center_pos]==0):
            #flattenarray=flattenarray+1
        flattenarray[center_pos]=flattenarray[center_pos]+1  
        for i in range(0,len(flattenarray)):
            if(i==4):
                continue
            else:
                #print("---------"+str(i)+"------"+str(flattenarray[i]))
                #print(abs(flattenarray[4]-flattenarray[i]))
                value=abs(flattenarray[i]-flattenarray[4])
                if(value==0):
                    value=value+1
                else:
                    continue
                    
                flattenarray[i]=1/value

        #print(flattenarray)
        flattenarray[center_pos]=0
        #print(flattenarray)
        sum_total=sum(flattenarray)

        #print("SUm total is:"+str(sum_total))
            
        for i in range(0,len(flattenarray)):
            if(i==4):
                flattenarray[i]=value_of_p
            else:
                flattenarray[i]=((1-value_of_p)*(flattenarray[i]/sum_total))
        #print(flattenarray)
        Convolve_array=np.multiply(filter_array,flattenarray)
        Gradient_Inverse_value=sum(Convolve_array)
        #print(filter_array)
        # print(Gradient_Inverse_value)
        return Gradient_Inverse_value

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
    
    def convolution(self,band,algo):
        if (self.k % 2 == 0):
            raise Exception("K should be odd number.")
            
        paddingValue = int((self.k - 1)/2)
        imageBand = self.imagePadding(band,paddingValue)
        arrayshape = imageBand.shape
        outputband = []

        for rowNo in range(arrayshape[0] - paddingValue):
            outputbandCol = []
            for colNo in range(arrayshape[1] - paddingValue):
                neighbourhoodMatrix = imageBand[rowNo:rowNo + self.k,colNo:colNo + self.k]
#                 print(neighbourhoodMatrix)
                if algo == 'Trimmed Mean Filter':
                    avgValue = self.trimmedMean(neighbourhoodMatrix)
                if algo == 'Inverse Gradient Filter':
                    avgValue = self.inverseGradient(neighbourhoodMatrix,self.P)
                outputbandCol.append(avgValue)
            outputband.append(outputbandCol)
        return np.array(outputband)

    def globalmeanstd(self,imagearray):
        meanarray = [np.mean(array[~np.isnan(array)].flatten())  for array  in imagearray ]
        stdarray = [np.std(array[~np.isnan(array)].flatten()) for array  in imagearray ]
        print(meanarray,stdarray)
        return  meanarray,stdarray

    def display(self,outputimage):
        final_image=cv2.merge((outputimage[2],outputimage [1],outputimage[0]))
        cv2.imwrite('outimage.jpg',final_image)
        st.image('./outimage.jpg', caption=f"Output image", use_column_width=True)

    
    def trimmedSmoothing(self,algo=None):
        outputimage = []
        img = self.image
        inputmean, inputstd = [],[]
        for bandNumber in range(img.shape[2]):
            bandarray = np.array([array[:,bandNumber] for array in img])
            inputmean.append(np.mean(bandarray[~np.isnan(bandarray)].flatten()))
            inputstd.append(np.std(bandarray[~np.isnan(bandarray)].flatten()))
            outputband = self.convolution(bandarray,algo) 
            outputimage.append(outputband)
        self.display(outputimage)
        mean, std = self.globalmeanstd(outputimage)

        st.write(f"The input image global mean(RGB):{tuple(np.round(inputmean,2))} \n and global standard deviation(RGB):{tuple(np.round(inputstd,2))}")
        st.write(f"The output image global mean(RGB):{tuple(np.round(mean,2))} \n  and global standard deviation(RGB):{tuple(np.round(mean,2))}")
        
        return None  

    def inverseGradientSmoothing(self,algo = None):     
        outputimage = []
        img = self.image
        inputmean, inputstd = [],[]
        for bandNumber in range(img.shape[2]):
            bandarray = np.array([array[:,bandNumber] for array in img])
            inputmean.append(np.mean(bandarray[~np.isnan(bandarray)].flatten()))
            inputstd.append(np.std(bandarray[~np.isnan(bandarray)].flatten()))
            outputband = self.convolution(bandarray,algo) 
            outputimage.append(outputband)
        self.display(outputimage)
        mean, std = self.globalmeanstd(outputimage)

        st.write(f"The input image global mean(RGB):{tuple(np.round(inputmean,2))} \n and global standard deviation(RGB):{tuple(np.round(inputstd,2))}")
        st.write(f"The output image global mean(RGB):{tuple(np.round(mean,2))} \n and global standard deviation(RGB):{tuple(np.round(mean,2))}")
        
        return None  

Algorithm = st.sidebar.selectbox('Which algorithm do you want to implement?', ["Select algorithm", 'Trimmed Mean Filter','Inverse Gradient Filter'])
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

    if Algorithm == 'Select algorithm':
            pass

    if Algorithm == 'Trimmed Mean Filter':
            K_trim = st.sidebar.slider('Select size of convolution matrix: n x n',3, 9, step=2)
            ourfilter = smoothingFilters(img_array,K=K_trim)
            outputimage = ourfilter.trimmedSmoothing(algo = 'Trimmed Mean Filter' )

    if Algorithm == 'Inverse Gradient Filter':
            K_inverse = st.sidebar.slider('Select size of convolution matrix: n x n ',3, 9,step=2)
            P = st.sidebar.slider('Select weight assigned to center pixel (p)',0.0, 1.0,value=0.5)
            ourfilter = smoothingFilters(img_array,K=K_inverse, P = P)
            outputimage = ourfilter.inverseGradientSmoothing(algo = 'Inverse Gradient Filter')