# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings
from keras import regularizers
from keras import optimizers
import os
from VGG16.PrepareData import PrepareData
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D,Dropout
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D ,BatchNormalization
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.optimizers import SGD
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
import datetime
import matplotlib.pyplot as plt
from VGG16.Evaluate import Evaluate

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1',trainable=False)(x)
        x = Dense(4096, activation='relu', name='fc2',trainable=False)(x)
        x = Dense(classes, activation='sigmoid', name='predictions',trainable=True)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

a = np.zeros((1,1000))

a[0,500]= 1
if __name__ == '__main__':
    model = VGG16(include_top=True, weights='imagenet')

    img_path = 'cat.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    model.compile(optimizer = 'sgd', loss ='binary_crossentropy', metrics = ["accuracy"])
    model.fit([x],[a],epochs=20)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

def CreateModelLocal(input_shape1,input_shape2,channels,output,L,Verbose= 0,FineTunning=False):
  
    
    input = Input(shape=(input_shape1,input_shape2,channels),name = 'image_input')
    base_model = VGG16(weights='imagenet', include_top=False,pooling='avg')
    
    for layer in base_model.layers:
            layer.trainable = FineTunning
            
    output_vgg16_conv = base_model(input)
    x = Dense(1000,  name='predictions1', kernel_initializer = 'glorot_uniform',activation='relu',activity_regularizer=regularizers.l2(L))(output_vgg16_conv)
  #  x = Dropout(0.1)(x)
    x = Dense(1000,  name='predictions2', kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = Dense(300,  name='predictions3', kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = Dense(100,  name='predictions4', kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = Dense(output,  name='predictions5', kernel_initializer = 'glorot_uniform')(x)
    model = Model(inputs=input, outputs=x)
    
    if Verbose ==1:    
        
        model.summary()

    return model


def FitModel (X_train, y_train,X_val,y_val,model,batch_size,LimitPatience,nepoch,Verbose,logPath):
    val_loss=[100000]
    train_loss= []
    Patience = 0
    X_val_c  = np.concatenate((X_val))
    y_val_c  = np.concatenate((y_val))

    for i in range(nepoch):
        ##Fit the model and evaluate complete patients. Regular FIT
        Patient= np.random.randint(len(X_train), size=batch_size)
        Slices=[]
        for i in range(len(Patient)):
            Slices.append(list(np.random.randint(len(X_train[Patient[i]]), size=1))[0])

        X_train_c=[None]*batch_size
        y_train_c=[None]*batch_size

        for i in range(batch_size):
            X_train_c[i] =X_train[Patient[i]][Slices[i]] 
            y_train_c[i] =y_train[Patient[i]][Slices[i]] 


        X_train_c =np.array((X_train_c))
        y_train_c  = np.array((y_train_c))


        History = model.fit(X_train_c,y_train_c,validation_data=[X_val_c,y_val_c],
                  epochs=1,shuffle=True,verbose = Verbose)
        train_loss.append((History.history['loss'][0]))
        if History.history['val_loss'][0] <np.min(val_loss): 
            Patience=0
            model.save_weights(logPath + "/BestModel")
        else:
            Patience = Patience +1
            print(Patience)
        val_loss.append((History.history['val_loss'][0]))

        if Patience ==LimitPatience:
            print("Early Stop")
            break
    return model, val_loss,train_loss


