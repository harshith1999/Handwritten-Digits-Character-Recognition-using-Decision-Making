
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:13:38 2019

@author: HARSHITH
"""
import matplotlib.pyplot as pt #To plot the image
import pandas as pd # to extract the Dataset from csv file
from sklearn.tree import DecisionTreeClassifier #To predict the accuracy using Decision Tree Classifier


# In[2]:


#Read the Training Dataset
x=pd.read_csv("F:/Project work 1/Character + Digits data/characters-digits-train.csv")
print(x)


# In[3]:


#Read the Testing Dataset
y=pd.read_csv("F:/Project work 1/Character + Digits data/characters-digits-test.csv")
print(y)


# In[4]:


#Convert the Training Dataset into Matrix form
data=x.as_matrix()
print(data)


# In[5]:


#Convert the Testing Dataset into Matrix form
data1=y.as_matrix()
print(data1)


# In[73]:


clf=DecisionTreeClassifier()
#Taken the Training data contains from 0 to 112798 and Classify from first column (0th Column is Labels)
xtrain=data[0:112798,1:]
#Taken the Training data contains from 0 to 112798 that considers 0th column as train_label
train_label=data[0:112798,0]
#List of features ``and it's Labels fitted to form classification 
clf.fit(xtrain,train_label)


# In[86]:


#Taken the testing Dataset contains from 0 to 18978 and classify from first column (0th column is labels)
xtest=data1[0:18798,1:]
#Taken the testing data contains from 0 to 18978 that considers 0th column as train_label
actual_label=data1[0:18798,0]


# In[84]:


#Taken an image to predict value from Testing Dataset
d=xtest[65]
#Plot the image in 28X28.
d.shape=(28,28)
#255-d is to remove the back background and display the white background and images shown in Gray form.
pt.imshow(255-d,cmap='gray')
pt.show()


# In[83]:


#Predict the Image value with the Trained set that is classified.
print(clf.predict([xtest[65]]))


# In[85]:


#Find the Accuracy between the Features compared with Testing feature
p=clf.predict(xtest)
count=0
for i in range(0,18798):
    count+=1 if p[i]==actual_label[i] else 0
print("accuracy",(count/18978)*100)

