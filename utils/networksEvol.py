from comet_ml import Experiment
import numpy as np
import comet_ml
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf


######### modality convert networks ##################################
def mp2ge_net(in_size, out_size):
#     in_size=in_size0[0]
    inLayer = keras.layers.Input([in_size])
    net = keras.layers.Dense(in_size, activation="relu")(inLayer)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dense(in_size//2, activation="relu")(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(out_size//4, activation="relu")(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(out_size, activation=None)(net)
    return inLayer, net

def mp2ge(inLayer,in_size, out_size):
#     in_size=in_size0[0]
#     inLayer = keras.layers.Input([in_size])
    net = keras.layers.Dense(in_size, activation="relu")(inLayer)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dense(in_size//2, activation="relu")(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(out_size//4, activation="relu")(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(out_size, activation='tanh')(net)
    return net

def ge2mp_net(inLayer, in_size, out_size):
#     in_size=in_size0[0]
#     inLayer = tf.keras.layers.Input([in_size])
    net = keras.layers.Dense(in_size, activation="relu")(inLayer)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dense(in_size*2, activation="relu")(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(out_size*4, activation="relu")(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(out_size, activation='tanh')(net)
    return net


########## Simple Linear model  ##############
from numpy.linalg import inv
# Find the mapping between CP and L1k
def linearTransform(al1k_train,acp_train,al1k_test,acp_test):
    A=np.matmul(np.matmul(inv(np.matmul(np.transpose(acp_train),acp_train)),np.transpose(acp_train)),al1k_train)
    pred=np.matmul(acp_test,A)
    return pred


######### Simple MLP  ##################################
def nnTransform(al1k_train_0,acp_train_0,al1k_test,acp_test):
    experiment = comet_ml.Experiment(
            api_key='wRu2GizrrhvMrx22c5346ocuq',
            project_name='Rosetta-cp2ge-nnTransform'
        )
#     nepoch=600 # for cp to ge
    nepoch=100 # for ge to cp
    
    pertColName=al1k_train_0.columns[0]
    l1kAl=al1k_train_0[pertColName].tolist()+al1k_test[pertColName].tolist()
#     le.fit(list(set(l1kAl)))

    nValAlleles=4;
    valAs=al1k_train_0[pertColName].sample(nValAlleles).tolist()
    valEnabled=1;
    if valEnabled:
    #         valAs=['PPP2R1A_p.C329F', 'TP53_p.V272K', 'EBNA1BP2_WT.o', 'KEAP1_WT.o']
        al1k_train=al1k_train_0[~al1k_train_0[pertColName].isin(valAs)].reset_index(drop=True)
        al1k_val=al1k_train_0[al1k_train_0[pertColName].isin(valAs)].reset_index(drop=True)

        acp_train=acp_train_0[~acp_train_0[pertColName].isin(valAs)].reset_index(drop=True)
        acp_val=acp_train_0[acp_train_0[pertColName].isin(valAs)].reset_index(drop=True)
    else:
        al1k_train=al1k_train_0.copy()
        al1k_val=al1k_train_0.copy()
        acp_train=acp_train_0.copy()
        acp_val=acp_train_0.copy()    
    
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config = configuration)

    tf.keras.backend.set_session(session)
    inLayer, net= mp2ge_net(acp_train.shape[1]-1, al1k_train.shape[1]-1)
    model = Model(inLayer, net)
#   optimizer=keras.optimizers.Adam(lr=0.0001) for treatment level
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mean_absolute_error',metrics=['mae', pcc])
    cycleEnable=False
    dgen_train = MultimodalDataGenerator(acp_train, al1k_train,cycleEnable)
    dgen_val = MultimodalDataGenerator(acp_val, al1k_val,cycleEnable)
    dgen_test = MultimodalDataGenerator(acp_test, al1k_test,cycleEnable)
    
#     callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
#         filepath="../../results/modelW",
#         save_weights_only=True,
#         save_best_only=False,
#         monitor='val_loss'
#     )
#     callback_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="../../results/modelW",monitor='val_loss', verbose=1, mode='auto')
#     cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    cb=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01,patience=5, min_lr=0.0001)
    
    callback_csv = keras.callbacks.CSVLogger(filename="../../results/modelW/log.csv")
    callbacks = [cb, callback_csv]
#     model.fit_generator(generator=dgen_train, steps_per_epoch=5,epochs=400,verbose=0,validation_data=(acp_test.iloc[:,1:].values, al1k_test.iloc[:,1:].values),callbacks=callbacks)
    model.fit_generator(generator=dgen_train, steps_per_epoch=5,epochs=nepoch,verbose=1,validation_data=dgen_val[0],callbacks=callbacks)
#     predicted_ge = model.predict(np.asarray(acp_test)[:,1:])
    predicted_ge = model.predict(dgen_test[0][0])
    return predicted_ge


######### MLP + reconstruction loss  ##################################
def nnTransformWithCycle(al1k_train,acp_train,al1k_test,acp_test):
    experiment = comet_ml.Experiment(
            api_key='wRu2GizrrhvMrx22c5346ocuq',
            project_name='Rosetta-cp2ge'
        )
    nepoch=300 # for cp to ge
#     nepoch=600 # for ge to cp
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config = configuration)

    tf.keras.backend.set_session(session)
    real_mp, synthetic_ge= mp2ge_net(acp_train.shape[1]-1, al1k_train.shape[1]-1)
#     d_ge_guess_synthetic = D_ge_static(synthetic_ge)
    #     synthetic_ge = mp2ge_net(real_mp)
    reconstructed_mp = ge2mp_net(synthetic_ge,al1k_train.shape[1]-1,acp_train.shape[1]-1)
    model_outputs = [synthetic_ge, reconstructed_mp]
    model = Model(real_mp, model_outputs)
    compile_losses=['mean_absolute_error','mean_absolute_error']
    compile_weights=[10,1]
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=compile_losses,loss_weights=compile_weights,metrics=['mae', pcc])   
    
    cycleEnable=True
    dgen_train = MultimodalDataGenerator(acp_train, al1k_train,cycleEnable)
    dgen_test = MultimodalDataGenerator(acp_test, al1k_test,cycleEnable)
    
    cb=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01,patience=5, min_lr=0.0001)
    
    callback_csv = keras.callbacks.CSVLogger(filename="../../results/modelW/log_c.csv")
    callbacks = [cb, callback_csv]
    minDim=np.minimum(acp_test.shape[0],al1k_test.shape[0])
#     print(minDim)
    model.fit_generator(generator=dgen_train, steps_per_epoch=5,epochs=nepoch,verbose=1,\
                        validation_data=(acp_test.iloc[:minDim,1:].values, [al1k_test.iloc[:minDim,1:].values,acp_test.iloc[:minDim,1:].values]),callbacks=callbacks)

    predicted_ge = model.predict(np.asarray(acp_test)[:,1:])
# #     print(predicted_ge.shape, al1k_test.shape)
# #     cc=scipy.stats.pearsonr(predicted_ge[0].T, al1k_test.iloc[:,1:].values.T)
#     cc=scipy.stats.pearsonr(predicted_ge[0].mean(axis=0).T, al1k_test.iloc[:,1:].values.mean(axis=0).T)
#     return cc
    return predicted_ge[0]


############ Metrics
def pcc(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)



############ data generators
class MultimodalDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, modA, modB,cycleEnable, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.pertColName=modA.columns.tolist()[0]
        self.modA = modA
        self.modB = modB
        self.classes = set( modA[self.pertColName].unique()).intersection( modB[self.pertColName].unique() )
        self.classes = list(self.classes)
        self.create_samples()
        self.cycle=cycleEnable
        
    def create_samples(self):
        dataA = []
        dataB = []
        classes = []
        # Generate all combinations of A and B with the same label
        for cl in self.classes:
            for idx, rowA in self.modA[self.modA[self.pertColName] == cl].iterrows():
                for jdx, rowB in self.modB[self.modB[self.pertColName] == cl].iterrows():
                    dataA.append(np.reshape(np.asarray(rowA)[1:], (1,self.modA.shape[1]-1)))
                    dataB.append(np.reshape(np.asarray(rowB)[1:], (1,self.modB.shape[1]-1)))
                    classes.append(cl)
        self.X = np.concatenate(dataA)
        self.Y = np.concatenate(dataB)
        self.Y2 = np.concatenate(dataA)
        self.Z = classes
        print("Total pairs:", len(dataA), self.X.shape, self.Y.shape)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.modA) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Initialization
        index = np.arange(0,self.X.shape[0])
        np.random.shuffle(index)
        X = self.X[index[0:self.batch_size], :]
        Y = self.Y[index[0:self.batch_size], :]
        if self.cycle:
            Y2 = self.X[index[0:self.batch_size], :]
            return X, [Y,Y2]
        else:
            
            return X,Y

