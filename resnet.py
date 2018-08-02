
import keras 
import numpy as np 
import sklearn
from utils.utils import save_logs

class Classifier_RESNET: 

    def __init__(self, output_directory, input_shape, nb_classes,nb_prototypes,classes,
                 verbose=False,load_init_weights = False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.nb_prototypes = nb_prototypes
        self.classes = classes
        if(verbose==True):
            self.model.summary()
        self.verbose = verbose
        if load_init_weights == True: 
            self.model.load_weights(self.output_directory.
                                    replace('resnet_augment','resnet')
                                    +'/model_init.hdf5')
        else:
            # this is without data augmentation so we should save inital non trained weights
            # to be used later as initialization and train the model with data augmentaiton
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)
        
        # BLOCK 1 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum 
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum 
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal 
        shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL 
        
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), 
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory+'best_model.hdf5' 

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_test,y_true):
        # convert to binary 
        # transform the labels from integers to one hot vectors
        self.enc = sklearn.preprocessing.OneHotEncoder()
        self.enc.fit(np.concatenate((y_train,y_true),axis=0).reshape(-1,1))
        y_train_int = y_train 
        y_train = self.enc.transform(y_train.reshape(-1,1)).toarray()
        y_test = self.enc.transform(y_true.reshape(-1,1)).toarray()
        
        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 16

        nb_epochs = 1

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        if len(x_train)>4000: # for ElectricDevices
            mini_batch_size = 128

        hist=self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                verbose=self.verbose, validation_data=(x_test,y_test) ,callbacks=self.callbacks)
        
        model = keras.models.load_model(self.output_directory+'best_model.hdf5')

        y_pred = model.predict(x_test)

        # convert the predicted from binary to integer 
        y_pred = np.argmax(y_pred , axis=1)
       
        keras.backend.clear_session()

        save_logs(self.output_directory, hist, y_pred, y_true, 0.0)
        
        return y_pred 