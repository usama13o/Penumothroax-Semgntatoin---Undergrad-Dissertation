
from IPython import get_ipython

# #This file is concerned with creating classifier data from segmentation dataset. The samples are traversed and any sample found to have a matching mask with postive pixels is put in the 'disease' folder. Any images that have blacnk masks are put into the 'no_disease' folder. This only needs to be once to get the data into the right folders and then training can be done many times without repeating this setp. 
# # load libraries

#loading required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
from tqdm import tqdm_notebook

# # Use train_rle.csv to create classifier training set
# ### A csv that contains every filename in the dataset is used to check whether a file contains postive pixles or not.
# ### Indices of each cooresponding filenames are stored in their respective variables and used to extract them into seprate folders.


train_rle = pd.read_csv('/cs/home/khfy6uat/data/classfication_128/train-rle.csv')
train_rle.columns = ['ImageId', 'EncodedPixels']


# 
train_rle.head()


# 
im_id_disease = train_rle[train_rle.EncodedPixels!=' -1'].ImageId # for storing postive smaples 
im_id_no_disease = train_rle[train_rle.EncodedPixels==' -1'].ImageId # for the negative ones


# 
get_ipython().system('mkdir -p classifier_data/disease # creating folder if not there ')


# 

train_path = '/cs/home/khfy6uat/data/data1024/train/train' # path to the original segmentation dataset
cls_data_path = './classifier_data/disease' # new postive data folder 
# the following loop traverses the previously retrived indices and simply copies the filenames in the first column, adds the folder path, and moves them into the folder
for im_id in tqdm_notebook(im_id_disease):
    im_id = im_id +'.png'
    shutil.copy(os.path.join(train_path, im_id),os.path.join(cls_data_path, im_id))
    
print("Samples in disease folder: "+str(len(os.listdir(cls_data_path))))


# 
get_ipython().system('mkdir classifier_data/no_disease')


# 
# same process but for negative samples
train_path = '/cs/home/khfy6uat/data/data1024/train/train'
cls_data_path = './classifier_data/no_disease'
for im_id in tqdm_notebook(im_id_no_disease):
    im_id = im_id +'.png'
    shutil.copy(os.path.join(train_path, im_id),os.path.join(cls_data_path, im_id))
    
print("Samples in no_disease folder: "+str(len(os.listdir(cls_data_path))))


# 


