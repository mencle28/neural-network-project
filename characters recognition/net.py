# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:38:54 2021

@author: TRABET Clément

implementation of a neural network from scratch 

data preovided by emnist 
picture of 28*28 pixels

training_data and test_data : type array of tuple (int, array) 
where int is the integer in the picture represented by the second array
"""

import random
import matplotlib.pyplot as plt
import data as D
import numpy as np
import matplotlib.image as img
import cv2
#import emnist as emn

training_data,test_data=D.load()

class Network(object):
        def __init__(self, lst):
            self.nb_couches = len(lst)
            self.list_taille = lst
            self.biais = [np.random.randn(y, 1) for y in lst[1:]] #randn est un tableau de taille les parametres de valeurs d'esperance nul de variance 1 gaussiène
            self.poids = [np.random.randn(y, x) for x, y in zip(lst[:-1], lst[1:])]
            
        def calc_forward(self, v,f):
            """calcul la sortie du neurone pour un vecteur donné v """
            for b, w in zip(self.biais, self.poids):
                v = f(np.dot(w, v)+b)
            return v

        def update_mini_batch(self, mini_batch, eta,f,df):
            """calcul des derivé partielles des biais et des poids"""
            deriv_b = [np.zeros(b.shape) for b in self.biais]
            deriv_w = [np.zeros(w.shape) for w in self.poids]
            for x, y in mini_batch:
                lst_deriv_b, lst_deriv_w = self.calc_backward(x, y,f,df)
                deriv_b = [nb+dnb for nb, dnb in zip(deriv_b, lst_deriv_b)]
                deriv_w = [nw+dnw for nw, dnw in zip(deriv_w, lst_deriv_w)]
            self.poids = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.poids, deriv_w)]
            self.biais = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biais, deriv_b)]

        def calc_backward(self, x, y,f,df):
            """backward propagation """
            deriv_b = [np.zeros(b.shape) for b in self.biais]
            deriv_w = [np.zeros(w.shape) for w in self.poids]

            activation = x
            activations = [x]
            zs = []
            for b, w in zip(self.biais, self.poids):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)

            delta = self.cost_derivative(activations[-1], y) *  df(zs[-1])
            deriv_b[-1] = delta
            deriv_w[-1] = np.dot(delta, activations[-2].transpose())

            for l in range(2, self.nb_couches):
                z = zs[-l]
                delta = np.dot(self.poids[-l+1].transpose(), delta) * df(z)
                deriv_b[-l] = delta
                deriv_w[-l] = np.dot(delta, activations[-l-1].transpose())
            return (deriv_b, deriv_w)


        def eval(self, test_data,f):
            """evalue la convergence du neurone test sur le dataset le nombre de réponses correct"""
            test_results = [(np.argmax(self.calc_forward(x,f)), y)
                            for (x, y) in test_data]
            return sum(int(x == y) for (x, y) in test_results)

        def cost_derivative(self, output_activations, y):
            """dérivé de la fonction cout"""
            return (output_activations-y)
        
        def cost(self,test_data,f):
            s=0
            for x,y in test_data:
                a=self.calc_forward(x,f)
                for j in range(len(a)):
                    if j==y:
                        s=s+(a[j]-1)**2
                    else:
                        s=s+(a[j])**2
            return s

        def calc(self, vect,f):
            """calcul la sortie troiver pour un vecteur"""
            return np.argmax(self.calc_forward(vect,f))
               
        
        def descente_de_gradient(self, training_data, epochs, taille_mini_batch, eta, test_data,f,df):
            """ descente de gradient sur les minibatch """
            X_epoch=[0]  #initialisation des listes/compteur pour afficher la convergence
            Y=[0]
            Xcout=[]
            Yc=[]
            Ycout=[]
            i=0
            plt.clf()
            n = len(training_data)
            nd= len(test_data)
            for j in range(epochs):
                random.shuffle(training_data)  #on mélange les batchs
                mini_batches = [training_data[k:k+taille_mini_batch] for k in range(0, n, taille_mini_batch)]
                for mini_batch in mini_batches:
                    i=i+1
                    Xcout.append(i)
                    #Ycout.append(self.cost(test_data,f)/nd)
                    #Yc.append( self.eval(test_data,f) /len(test_data))
                    self.update_mini_batch(mini_batch, eta,f,df)
                X_epoch.append(j)
                Y.append( self.eval(test_data,f) /len(test_data))
                    
            plt.title("Croissance de la précision au fil des calculs")
            plt.plot(X_epoch,Y,label="epochs")
            #plt.plot(Xcout,self.normalise_lst(Ycout),label="fonction cout")
            #plt.plot(Xcout,Yc,label="batch")
            plt.legend(loc=1)
            print(max(Y))
            plt.show()
            
        def normalise_lst(self,lst):
            l=[]
            m=max(lst)
            for i in range (len(lst)):
                l.append(lst[i]/m)
            return l
        
        def convergence_fun_lr(self, training_data, epochs, taille_mini_batch, test_data,f,df):
            """ étude de la convergence en fontion du learning rate pour sigmoid """
            
            plt.clf()
            l_r_lst=[0.001,0.01,0.1,1,10,100]
            n = len(training_data)
            i_biais=self.biais
            i_poids=self.poids
            for l_r in l_r_lst:
                self.biais=i_biais
                self.poids=i_poids
                
                X=[]  #initialisation des listes/compteur pour afficher la convergence
                Y=[]
                for j in range(epochs):
                    random.shuffle(training_data)  #on mélange les batchs
                    mini_batches = [training_data[k:k+taille_mini_batch] for k in range(0, n, taille_mini_batch)]
                    for mini_batch in mini_batches:
                        self.update_mini_batch(mini_batch, l_r,f,df)
                    X.append(j)
                    
                    Y.append( self.eval(test_data,f) /len(test_data))

                plt.plot(X,Y,label=str(l_r))
            plt.legend(loc=1) 
            plt.title("Convergence du neurone en fonction du learning rate")
            plt.show()
            
        def convergence_fun_f(self, training_data, epochs, taille_mini_batch, lr, test_data):
            """ étude de la convergence en fonction de la fonction d'activation """
            
            plt.clf()
            f_lst=[sigmoid,atan,tanh]
            nom_f_lst=['sigmoid','atan','tanh']
            df_lst=[dsigmoid,datan,dtanh]
            n = len(training_data)
            i_biais=self.biais
            i_poids=self.poids
            for (f,df,nf) in zip(f_lst,df_lst,nom_f_lst):
                self.biais=i_biais
                self.poids=i_poids
                i=0
                X=[]  #initialisation des listes/compteur pour afficher la convergence
                Y=[]
                for j in range(epochs):
                    random.shuffle(training_data)  #on mélange les batchs
                    mini_batches = [training_data[k:k+taille_mini_batch] for k in range(0, n, taille_mini_batch)]
                    for mini_batch in mini_batches:
                        self.update_mini_batch(mini_batch, lr,f,df)
                    Y.append( self.eval(test_data,f) /len(test_data))
                    X.append(j)
                plt.plot(X,Y,label=str(nf))
            plt.legend(loc=1)
            plt.title("Convergence du neurone en fonction de la fonction d'activation")
            plt.show()
            
        def convergence_fun_batch_taille(self, training_data, epochs, lr, test_data,f,df):
            """ étude de la convergence en fonction de la fonction d'activation """
            
            plt.clf()
            lst=[10,50,100,1000]
            n = len(training_data)
            i_biais=self.biais
            i_poids=self.poids
            for taille_mini_batch in lst:
                self.biais=i_biais
                self.poids=i_poids
                i=0
                X=[]  #initialisation des listes/compteur pour afficher la convergence
                Y=[]
                for j in range(epochs):
                    random.shuffle(training_data)  #on mélange les batchs
                    mini_batches = [training_data[k:k+taille_mini_batch] for k in range(0, n, taille_mini_batch)]
                    for mini_batch in mini_batches:
                        self.update_mini_batch(mini_batch, lr,f,df)
                        X.append(i)
                        i=i+taille_mini_batch
                        Y.append( self.eval(test_data,f) /len(test_data))
                plt.plot(X,Y,label=str(taille_mini_batch))
            plt.legend(loc=1)
            plt.title("Convergence du neurone en fonction de la taille du batch")
            plt.show()
            
        def convergence_fun_taille_neurone(self, training_data, epochs, taille_mini_batch, eta, test_data,f,df):
            """ descente de gradient sur les minibatch """
            X=[]  #initialisation des listes/compteur pour afficher la convergence
            Y=[]
            i=0
            n = len(training_data)
            for j in range(epochs):
                random.shuffle(training_data)  #on mélange les batchs
                mini_batches = [training_data[k:k+taille_mini_batch] for k in range(0, n, taille_mini_batch)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta,f,df)
                    X.append(i)
                    i=i+1
                    Y.append( self.eval(test_data,f) /len(test_data))
            return (X,Y)
        
        def calc_cout(self, training_data, epochs, taille_mini_batch, eta, test_data,f,df):
            """ descente de gradient sur les minibatch """
            X_epoch=[]  #initialisation des listes/compteur pour afficher la convergence
            Y=[]
            Ycout=[]     
            plt.clf()
            n = len(training_data)
           
            for j in range(epochs):
                random.shuffle(training_data)  #on mélange les batchs
                mini_batches = [training_data[k:k+taille_mini_batch] for k in range(0, n, taille_mini_batch)]
                for mini_batch in mini_batches:                
                    #Yc.append( self.eval(test_data,f) /len(test_data))
                    self.update_mini_batch(mini_batch, eta,f,df)
                Ycout.append(self.cost(test_data,f))
                X_epoch.append(j)
                Y.append( self.eval(test_data,f) /len(test_data))
                    
            plt.title("Croissance de la précision au fil des calculs")
            plt.plot(X_epoch,Y,label="epochs")
            plt.plot(X_epoch,self.normalise_lst(Ycout),label="fonction cout")
            #plt.plot(Xcout,Yc,label="batch")
            plt.legend(loc=1)
            print(max(Y))
            plt.show()
        

def convergence_f_taille_neurone():
    lst=[[784,200,100,50,10],
         [784, 80, 30, 10],
         [784, 50, 10],
         [784, 10]]
    
    for l in lst:
        Net=Network(l)
        (X,Y)=Net.convergence_fun_taille_neurone(training_data, 1, 100, 2, test_data,sigmoid,dsigmoid)
        plt.plot(X,Y,label=str(len(l)))
    plt.legend(loc=1)
    plt.title("Convergence du neurone en fonction de la taille du neurone")
    plt.show()
    
    
#### fonction d'activations

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))


def atan(z):
    return np.arctan(z)

def datan(z):
    return 1/(1+z*z)

def tanh(z):
    return np.tanh(z)
def dtanh(z):
    return 4/((np.exp(-z)+ np.exp(z))**2)


#### fonction traitement d'image

def to_28_28(image):
    down_width = 28
    down_height = 28
    down_points = (down_width, down_height)
    resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
    #cv2.imshow('Resized Down by defining height and width', resized_down)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return resized_down

def to_vector ( image ) :
    x = cv2 . cvtColor ( to_28_28(image) , cv2 . COLOR_BGR2GRAY )
    return 255*np . reshape (1.0 - x /255 , (784 , 1) )

def image_test():
    lst=['tt7.png','tt1.png','tt0.png','tt9.png']
    n_lst=[0,2,3,9]
    fig=plt.figure(figsize=(10,7))
    rows = 2
    columns = 2
    plt.axis('off')
    for i in range (1,5):
        
        image=img.imread(lst[i-1])
        fig.add_subplot(rows,columns,i)
        plt.imshow(image,cmap='Greys')
        plt.axis('off')
        #vect=to_vector(cv2.imread(lst[i-1]))
        y=Net.calc(test_data[n_lst[i-1]][0],sigmoid)
        y_2=test_data[n_lst[i-1]][1]
        
        plt.title(str(y)+' '+str(y_2),fontsize="40")
    plt.axis('off')
    plt.show()
    
    
def single(i):
    (v,yv)=test_data[i]
    #image=img.imread('1.jpg')
    #vect=to_vector(image)
    y=Net.calc(v,sigmoid)
    return y,yv

img=cv2.imread("1.jpg")

def image(img,net):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize ( imgray ,(28 ,28), interpolation= cv2.INTER_LINEAR )
    v=np . reshape (1.0 - img/255 , (784 , 1) )
    return Net.calc(v,sigmoid)


#### fonction test sur neurones

Net=Network([784,50,10])
#Net.descente_de_gradient(training_data, 100, 100, 2, test_data,sigmoid,dsigmoid)
#print(image(img,Net))
#Net.calc_cout(training_data, 15, 20, 1, test_data[:100], sigmoid, dsigmoid)
Net.descente_de_gradient(training_data, 15, 100, 2, test_data,sigmoid,dsigmoid)
#Net.convergence_fun_lr(training_data, 15, 100, test_data,sigmoid,dsigmoid)
#Net.convergence_fun_f(training_data,20, 100, 2, test_data)
#Net.convergence_fun_batch_taille(training_data, 15, 2, test_data,sigmoid,dsigmoid)
#convergence_f_taille_neurone()
print("le neurone à appris")