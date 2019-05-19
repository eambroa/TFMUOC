# TFM UOC
Este repositorio contiene el código utilizado en el Trabajo fin de máster: "Transfer Learning aplicado al campo de la radioterapia: predicción de dosis en recto y vejiga para tratamientos de próstata".

Se adjuntan diferentes notebook con el código utilizado.
También se proporciona un html con la salida del código principal.

Los ficheros son los siguientes:

- Prostata: fichero general con el código utilizado para probar diferentes arquitecturas de la red. Este código llama a diferentes funciones: 

        - vgg16: contiene la información sobre el modelo VGG-16 utilizado.

        - PrepareData: contiene el código para preparar los set de entrenamiento, validación y test.
        
        - Evaluate: contiene el método usado para evaluar la predicción del DVH de un paciente como suma de las prediccciones de cada 
        uno de sus cortes, y la comparación con el valor de DVH del plan original.
        
        - CheckPatientDVH: creación del DVH de cada pacientes.
        
        
Por motivos de espacio y confidencialidad no se porporcionan los datos utilizados.
