#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   TP1 FAA Matthieu Caron 2016

import matplotlib.pyplot as plt
import numpy as np
import math
import random as r




################# LECTURE #################

def lecture(pathname) :
    fichier = open(pathname,'r')
    res = []
    for ligne in fichier :
        res.append(float(ligne))
    fichier.close()
    return res 

def lecture2(pathname) :
    fichier = open(pathname,'r')
    (dim1,dim2) = ([],[])
    for ligne in fichier :
        (taille,poids) = ligne.split()
        dim1.append(float(taille))
        dim2.append(float(poids))
    return (dim1,dim2) 

################# VARIABLES #################

(tmp_taille_f,tmp_poids_f) = lecture2('taillepoids_f.txt')
(tmp_taille_h,tmp_poids_h) = lecture2('taillepoids_h.txt')
taille_f = np.array(tmp_taille_f,float)
taille_h = np.array(tmp_taille_h,float)

nbFilles = len(taille_f)
nbHommes = len(taille_h)

poids_f = np.array(tmp_poids_f,float)
poids_h = np.array(tmp_poids_h,float)

lesZeros = np.zeros(nbFilles)
lesUns = np.ones(len(taille_h))

############# FONCTIONS #############
def sigmoidVecteur(vecteur) :
    newVecteur = np.copy(vecteur)
    for i in range(len(vecteur)) :
        newVecteur[i] = sigmoid(vecteur[i])
    return newVecteur

def sigmoid(x) :
    return 1 / (1+math.exp(-x))

def matrixDegresN(degres, vecteur) :
    '''
    entree degres (int) 
    sortie matrix de taille 
    '''
    newMatrix = np.zeros((degres+1,len(vecteur)))
    newMatrix[0,:] = np.ones(len(vecteur))
    for i in range(1,degres+1) :
        for j in range(len(vecteur)) :
            newMatrix[i][j] = vecteur[j]**i
    return newMatrix

def donneY(teta,matrix) :
    return np.dot(matrix.T,teta)
############# MESURE DE PERF #############
def risqueEmpirique(x,y,teta,N):
    sigmo = sigmoidVecteur(np.dot(x.T,teta))
    interm = np.dot(-y,np.log(sigmo)) - np.dot((1-y),np.log(1-sigmo))
    return np.sum(interm)/float(N)

def mesureAbs(x1,y,teta,N=100) :
    vecteur = y - np.dot(x1.T,teta)
    return np.sum(np.absolute(vecteur))/N

def mesureNormal2(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    return np.dot(vecteur.T, vecteur)/N

def mesureNormal1(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    interm = np.dot(vecteur.T, vecteur)
    return math.sqrt(interm)/N

def mesureNormalinf(x1,y,teta,N=100):
    vecteur = y - np.dot(x1.T,teta)
    return np.amax(np.absolute(vecteur))


def moindresCarres(matrix, vecteur):
    gauche = np.dot(matrix, matrix.T)
    droite = np.dot(matrix,vecteur) 
    return  np.dot(np.linalg.inv(gauche),droite)

def alpha(t) :
    return 1./(1.+4000.*float(t))

def descenteGradient(x1,y,teta,t, epsilon=0.00000001,N=100) :
    tempsActuel = t
    tetaActuel = teta
    while(True) :
        intermediaire = np.dot(x1,(y - np.dot(x1.T,tetaActuel))) #gradient
        tetaPlusPlus = tetaActuel + np.dot(np.dot(alpha(t),intermediaire),1./float(N))
        #test de convergence 
        if math.fabs(mesureNormal2(x1,y,tetaPlusPlus)-mesureNormal2(x1,y,tetaActuel)) <= epsilon :
            return tetaActuel
        else :
            tetaActuel = tetaPlusPlus
            tempsActuel += 1

def descenteGradientStochastique(matrix,y,teta,t,epsilon=0.00000001,N=100) :
    '''
    a corriger
    '''
    tempsActuel = t
    tetaActuel = teta
    while(True) :
        i = r.randint(0,99)
        vecX = np.array([matrix[0][i],matrix[1][i]],float)
        intermediaire = np.dot(vecX,(y[i] - np.dot(tetaActuel.T,vecX))) #gradient
        tetaPlusPlus = tetaActuel + np.dot(np.dot(alpha(t),intermediaire),1./float(N))
        #test de convergence 
        if math.fabs(mesureNormal2(vecX,y[i],tetaPlusPlus)-mesureNormal2(vecX,y[i],tetaActuel)) <= epsilon :
            return tetaActuel
        else :
            tetaActuel = tetaPlusPlus
            tempsActuel += 1

def descenteGradientSigmoide(matrix,y,teta,t,epsilon,N):
    tempsActuel = t
    tetaActuel = teta
    while(True) :
        intermediaire = np.dot(matrix,(y - sigmoidVecteur(np.dot(matrix.T,tetaActuel))))
        tetaPlusPlus = tetaActuel + np.dot(np.dot(alpha(t),intermediaire),1./float(N))
        if risqueEmpirique(matrix,y,tetaActuel,N) - risqueEmpirique(matrix,y,tetaPlusPlus,N) <= epsilon :
            return tetaActuel
        else :
            tetaActuel = tetaPlusPlus
            tempsActuel += 1

def randTeta(n) :
    res = []
    for i in range(n) :
        res.append(np.array([r.random(),r.random()],float))
    return res 

def descenteMultiple(matrix,y,listeTeta,t,epsilon,N) :
    res = []
    for teta in listeTeta :
        res.append(descenteGradientSigmoide(matrix,y,teta,t,epsilon,N))
    return res

################# SCRIPT #################


# (teta0,teta1) = moindresCarres(matrix0,y0)
# plt.plot(x0,y0,'ro')
# plt.plot(x0,teta1*x0+teta0,'bo')
# plt.show()

# plt.plot(x1,y1,'ro')
# matrix = matrixDegresN(2,x1)
# teta = moindresCarres(matrix,y1)
# y = donneY(teta,matrix)
# plt.plot(x1,y,'bo')
# plt.plot(x2,y2,'ro')

classe0 = np.zeros(nbFilles)
classe1 = np.ones(nbHommes)

probUnSachantX = float(nbHommes/(nbHommes+nbFilles))

matrix = np.ones((2,(nbHommes+nbFilles)))
matrix[1,:] = np.concatenate([taille_f,taille_h])
lesClasses = np.concatenate([classe0,classe1])


listeTeta = randTeta(10)
lesTetasLearn = descenteMultiple(matrix,lesClasses,listeTeta,1,0.000001,nbHommes+nbFilles)
seuil = 0.6


plt.ylim(-1,2)
plt.plot(taille_h,classe1,'bo')
plt.plot(taille_f,classe0,'ro')
for tetaL in lesTetasLearn :
    A = tetaL[1]
    b = tetaL[0]
    X = (1/A*math.log(1-seuil)-math.log(seuil)+b)
    plt.axvline(X)

plt.show()