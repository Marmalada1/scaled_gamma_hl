# Mark IX, Standard Gewichtung ist uniform
import pandas as pd
import numpy as np
import errno, sys
from scipy.stats import rankdata
from numba import jit

# Gibt das Gewicht für den jeweiligen Rang an.
# Gewichte werden anhand des prozentualen Anteils zum höchsten Rang eingeteilt Modus gibt an wie gewichtet werden soll 
# entweder top und bottom lastig oder nur top oder bottom etc.

def weighter (data,i,column,modus):
    
 length = len(data)
 x = data[i,column]/length
 if x == 1:
     return None
 if modus == 'top bottom':
    if x<=0.5:
        return 1-2*x
    elif x<1 and x>0.5:
        return 2*x-1
 elif modus == 'top':
        if x<1:
                return 1-x
 elif modus == 'bottom':
            if x<1:
                return x        
 elif modus == 'middle':
        if x<=0.5:
                return 2*x  
        elif x<1 and x >0.5:
                return 2-2*x
 elif modus == 'top bottom exp':
        if x<1:
            return 4*(x-0.5)**2
 else:
     print('Modus für Gewichtung nicht definiert')
     
    


# Gibt an wie viel kleiner bzw. unterschiedlich die zwei Ränge voneinander sind 
def R(data,x,y,column,d):
    if data[x,column]>=data[y,column]:
        return 0
    else:
        return d(data,x,y,column)
    


# die Distanz wird anhand der Gewichte berechnet, diese werden anhand des Ranges in der Position x und y im Dataframe ermittelt. Ausgweählt wird die Gewichtung anhand des Index.

# Distanzberechnung mit dem Maximum  

@jit(nopython=True)
def d_max(data,i,j,column):
    return max(data[int(min(data[i,column],data[j,column]))-1:int(max(data[i,column],data[j,column])-1),column+2])
# Distanzberechnung mit der Summe   
@jit(nopython=True)
def d_sum(data,i,j,column):
    return min(1,sum(data[int(min(data[i,column],data[j,column]))-1:int(max(data[i,column],data[j,column])-1),column+2]))
  

# wählt die Distanzberechnungsmethode aus
def distance_selector(distance):
    if distance=='max':
        return d_max
    elif distance=='sum':
        return d_sum
    else:
        print('Modus für Distanzberechnung nicht definiert')
        sys.exit(errno.EACCES)

# Die T-Normen und T-Conormen 
def t_norm_product(a,b):
    return a*b

def t_conorm_product(a,b):
    return a+b-a*b

def t_norm_luka(a,b):
    return max(a+b-1,0)

def t_conorm_luka(a,b):
    return min(a+b,1)

#Wählt die Norm aus, entweder Produkt oder Łukasiewicz
def tnorm_selector(mode):
    if mode == 'product':
        tnorm=t_norm_product
        tconorm=t_conorm_product
    elif mode == 'luka':
        tnorm=t_norm_luka
        tconorm=t_conorm_luka
    else:
        print('Modus für t_norm nicht definiert')
        sys.exit(errno.EACCES)
    return tnorm,tconorm

# Standardparameter für die Hauptfunktion
default_configuration=dict(
    weighting='uniform',
    tnorm='product',
    distance='max',
    weights=[]
)

# Funktion um die Eingaben für die Hauptfunktion vorzubereiten
def data_prep(x,y,weights,weighting):
    lenght=len(x)
    
    data1 = np.column_stack((rankdata(x), rankdata(y)))
    sort_indices = np.argsort(data1[:, 0])
    data1=data1[sort_indices]

    data2 = np.column_stack((rankdata(x), rankdata(y)))
    sort_indices2 = np.argsort(data2[:, 1])
    data2=data2[sort_indices2]
    # Auswahl zwischen eigener Gewichtung, oder einer vordefinierten Gewichtung
    if not len(weights):
        weight1=np.ones(lenght)
        weight2=np.ones(lenght)
        if weighting != "uniform":
            for i in range(lenght):   
                weight1[i]=weighter(data1,i,0,weighting)  
                weight2[i]=weighter(data2,i,1,weighting)
        return np.column_stack((data1,weight1,weight2))
    elif len(weights)==(lenght-1):
        weights.append(np.nan)
        return np.column_stack((data1,weights,weights))
    elif len(weights) and len(weights)!=(lenght-1):
        print('Die Länge der Gewichtungen entspricht nicht n-1!')
        sys.exit(errno.EACCES)   
    # vordefinierte Gewichtung falls eigener Gewichtungsvektor leer ist. Gewichtung top bottom wird genommen falls nicht spezifiziert.
   
    

# Hauptfunktion

def scaled_gamma(data1,data2,**kwargs):
    # Eingabe von Parametern
    kwargs = {**default_configuration, **kwargs}
    weighting=kwargs['weighting']
    weights=kwargs['weights']
    tnorm=kwargs['tnorm']
    distance=kwargs['distance']
    # Datenvorbereitung
    data =data_prep(data1,data2,weights,weighting)
    # Auswahl der Norme
    t_norm,t_conorm=tnorm_selector(tnorm)
    # Auswahl der Distanzberechnung
    d=distance_selector(distance)
    # Initilisierung der Anfangswerte 
    concordant = 0
    discordant = 0
    ties = 0
   
    
    # Berechnung der Konkordanz, Diskordanz und Ties für jedes Paar
    for i in range (len(data)):
        for z in range ((i+1),len(data)):
            # Falls ein Merkmal unentschieden ist, wird die Paarung als unentschieden gewertet
            if ((data[i,0] == data[z,0]) or (data[i,1] == data[z,1])):
                ties += 1
            else:
                ties += t_conorm(1-d(data,i,z,0),1-d(data,i,z,1))
                concordant += t_norm(R(data,i,z,0,d),R(data,i,z,1,d)) + t_norm(R(data,z,i,0,d),R(data,z,i,1,d))
                discordant += t_norm(R(data,i,z,0,d),R(data,z,i,1,d)) + t_norm(R(data,z,i,0,d),R(data,i,z,1,d))
    if (concordant == 0) and (discordant == 0):
        return np.nan
    else:
        return (concordant-discordant)/(concordant+discordant)
    # Gibt Gamma zurück
