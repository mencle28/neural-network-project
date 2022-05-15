# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:20:05 2021

@author: TRABET Clément
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2 
 
#loading
(train_X, train_y), (test_X, test_y) = mnist.load_data()

img1="1.jpg"

def resultat_form(result): 
    """ met sous la forme (0,0,0,1,0,0,0,0,0,0) """
    
    new_res=[]
    i=0
    for y in result:
        Y=np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],float)
        Y[y][0]=1
        new_res.append(Y)
        i=i+1
    return new_res

########################
def to_28_28 ( image ) :
    new = cv2 . resize ( image ,(28 ,28), interpolation= cv2.INTER_LINEAR )
    return new

def to_vector ( image ) :
    x = cv2 . cvtColor ( to_28_28(image) , cv2 . COLOR_BGR2GRAY )
    return np . reshape (1.0 - x /255 , (784 , 1) )

########################

def convertion_image(entreeM):
    """ met la matrice sous la forme d'un vecteur de taille (784,1)"""
    new_entree=[]
    for x in entreeM:
        X=[np.concatenate([np.diagonal(x[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-x.shape[0], x.shape[0])])]
        new_entree.append(1/255*np.transpose(X))
        #rajouter 1/255 devant np.transpose
    return new_entree

def concat(X,Y):
    tab=[]
    for i in range(len(X)):
        x=X[i]
        y=Y[i]
        tab.append((x,y))
    return tab
        
tab_apprend=concat(convertion_image(train_X[:40000]),resultat_form( train_y[:40000]))
tab_test=concat(convertion_image(test_X), test_y)





""" tab est une liste de tuples tq (matrice, resultat sous forme tablau avec 1 et 0) """

def load ():
    return tab_apprend,tab_test

def load_img():
    return (test_X[:20], train_y[:20])

def load2 ():
    return convertion_image(train_X[:70000]),resultat_form( train_y[:70000]), tab_test







#on a a dans le tableau train_X les images et train_y la valeur

#shape of dataset
"""
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
"""
#plotting
"""
print(train_y[0])
plt.imshow(train_X[0], cmap=plt.get_cmap('gray'))
plt.show()
"""

(img,valeur)=test_X[9],test_y[9]

plt.imshow(img, cmap=plt.get_cmap('binary'))






