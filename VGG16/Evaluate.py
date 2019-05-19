# -*- coding: utf-8 -*-

# Método para evaluar la predicción sobre un paciente como suma de las predicciones de cada uno de sus cortes, comparado con el valor de dosis impartido en el plan original. 
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from random import sample
from scipy.interpolate import interp1d


# Definimos la función Evaluate
def Evaluate(X,y,NHC,model,comment = '',verbose = 0) : 

    c=[]
    c2=[]
    if verbose == 1:
        f = plt.figure(figsize=(20,15))
        f.suptitle('Comparación del DVH para el ' + str(comment) +' set', fontsize=16)
        ax0=plt.subplot(int(len(NHC))/3, 4, 1)
        ax0.grid(True)
        ax0.set_title('Predicciones')
        ax0.set_ylabel('Volumen relativo')
        ax0.set_xlabel('Dosis (Gy)')
        ax2=plt.subplot(int(len(NHC))/3, 4, 2)
        ax2.grid(True)
        ax2.set_title('Planes Reales')
        ax2.set_ylabel('Volumen relativo')
        ax0.set_xlabel('Dosis (Gy)')
                
        
    for i in range(len(NHC)):

        Min=0
        Max=80 
        Pasos= 20
        Rango = np.arange(Min,Max, (Max-Min)/Pasos)
        
        # Dosis impartida en realidad.
        yt = np.sum(np.array(y[i]),axis = 0 ) 
        # Predicción por corte y suma total.
        prediction = model.predict(X[i])
        ym=  np.sum(prediction,axis = 0)
        
        # Plot Normalizado
        yt1 =yt[0:20]/yt[0]
        ym1 =ym[0:20]/ym[0]
        
        if (yt.shape[0]>20):
            yt2 =yt[20:40]/yt[20]
            ym2 =ym[20:40]/ym[20]
            c2.append(1/len(ym2)*np.sum((yt2-ym2)))


    
        c.append(1/len(ym1)*np.sum((yt1-ym1)))
       
        if verbose ==1:

            ax1=plt.subplot(int(len(NHC))/3, 4, i+3)

            ax1.plot(Rango,yt1,color='red',marker='o',linestyle='-',label='Plan Real')  
            ax1.plot(Rango,ym1,color='red',marker='',linestyle='--',label='Predicción') 
            
            if (yt.shape[0]>20):

                ax1.plot(Rango,yt2,color='blue',marker='',linestyle='-',label='Plan Real')  
                ax1.plot(Rango,ym2,color='blue',marker='',linestyle='--',label='Predicción') 
            


            ax1.grid(True)
            ax1.legend()
            ax1.set_title(NHC[i])
            ax1.set_ylabel('Volumen relativo')
            ax1.set_xlabel('Dosis (Gy)')
            
            ax2.plot(Rango,yt1,color='red',marker='',linestyle='--',label='Plan Real')  
            ax0.plot(Rango,ym1,color='blue',marker='',linestyle='--',label='Predicción')     
            if i == 0:
                ax0.legend()
                ax2.legend()

      

    plt.savefig(comment+'_image.png')
    plt.show()
  

   
    return c,c2



