# -*- coding: utf-8 -*-

# Método para preparar los sets Train, Val y test. La función requiere la lista de pacientes sobre la que realizar el sorteo, y 
# permite excluir pacientes proporcionando otra lista de paciente excluidos. También permite preseleccionar la lista de pacientes que 
# quieres para el test, y que no sea aleatorio, pasandole una lista de pacientes test. 
import numpy as np
from scipy.interpolate import interp1d
from random import sample

def PrepareData(X,y,NHC,ExcludedPatients=[],suboptimalNHC=[],PatientTest=[]):
    print('Suboptimos NHC Excluidos: ')
    print(suboptimalNHC)
    suboptimal=[]
    for i,v in enumerate(NHC): 
        if v in suboptimalNHC:
            suboptimal.append(i)
    print('Suboptimos Excluidos: ')
    print(suboptimal)
    PatientList =list( set(range(len(X))) -set(suboptimal))
    TestCandidates =list(set(PatientList) - set(ExcludedPatients))
    
    print('Candidatos a Test : ')
    print(TestCandidates)
    
    print('Pacientes excluidos al Test : ')
    print(ExcludedPatients)
    if PatientTest==[]:
        PatientTest =list(sample(TestCandidates,10))
     
    print('Elegidos para Test : ')
    print(PatientTest)

        
        
    PatientTrain =list(set(PatientList) - set(PatientTest))
    PatientVal = list(sample(PatientTrain,10))
    PatientTrain =list(set(PatientTrain) - set(PatientVal))


    X_train = list( X[i] for i in PatientTrain )
    X_test = list( X[i] for i in PatientTest )
    X_val = list( X[i] for i in PatientVal )

    
    NHC_train = list( NHC[i] for i in PatientTrain )
    NHC_test = list( NHC[i] for i in PatientTest )
    NHC_val = list( NHC[i] for i in PatientVal )

    
    

    y_train = list( y[i] for i in PatientTrain )
    y_test = list( y[i] for i in PatientTest )
    y_val = list( y[i] for i in PatientVal )

    
    return X_train,X_val,X_test,y_train,y_test,y_val,NHC_train,NHC_test,NHC_val,PatientTest






