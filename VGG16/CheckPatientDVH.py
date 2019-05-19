# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt



def checkdvh(X,y,SC,model,bucle,verbose = 0) :   
    SliceConts=np.concatenate(([0],SC))


    print( SliceConts)
    for i in range(bucle):
        
        print('Analizing Patient ' + str(i) + '/'+ str(len(SliceConts)-1))
        Min=0
        Max=80 
        Pasos= 20
        Rango = np.arange(Min,Max, (Max-Min)/Pasos)
        
        yt = np.sum(np.array(y[SliceConts[i]:SliceConts[i+1],:]),axis = 0 ) 


        if verbose ==1:
            print (i)
            plt.plot(Rango,yt/yt[0],color='red',marker='o',linestyle='--')  
            plt.grid(True)
            plt.title("DVH Recto")
            plt.show()
            
      