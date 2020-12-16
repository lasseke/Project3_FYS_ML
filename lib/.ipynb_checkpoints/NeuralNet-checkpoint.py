'''
Add python files exclusive for neural network here!
'''

def initializeNN():
   
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LeakyReLU
    from keras.layers import Dropout
    from keras import regularizers
    from keras import metrics
    #import tensorflow_addons as tfa

    ### Define metrics
    metrics = [
        metrics.CategoricalAccuracy(name="accuracy"),
        metrics.FalseNegatives(name="fn"),
        metrics.FalsePositives(name="fp"),
        metrics.TrueNegatives(name="tn"),
        metrics.TruePositives(name="tp"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall"),
        metrics.AUC(name='auc')#,
        #tfa.metrics.CohenKappa(name='kappa')
    ]



    # define the keras model
    nn = Sequential()
    nn.add(Dense(256, 
                 input_dim=102,
                kernel_regularizer='l1'))#, activation='relu'))
    nn.add(LeakyReLU(alpha=0.1))
    nn.add(Dropout(0.1))

    nn.add(Dense(128))#, activation='relu'))#,kernel_regularizer='l1'))
    nn.add(LeakyReLU(alpha=0.1))
    nn.add(Dropout(0.1))

    nn.add(Dense(64))#, activation='relu'))#,kernel_regularizer='l1'))
    nn.add(LeakyReLU(alpha=0.1))
    nn.add(Dropout(0.1))

    nn.add(Dense(64))#, activation='relu'))#,kernel_regularizer='l1'))
    nn.add(LeakyReLU(alpha=0.1))
    nn.add(Dropout(0.1))

    nn.add(Dense(31, activation='softmax'))

    nn.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=metrics)
    
    return nn