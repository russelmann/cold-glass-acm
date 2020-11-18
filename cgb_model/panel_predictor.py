"""
(c) 2020 Ruslan Guseinov (ruslan.guseinov@ist.ac.at), IST Austria
         Konstantinos Gavriil, TU WienProvided under MIT License
Train and use MDN
"""
import os
import os.path as path
import json
import pickle
import random
import time
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Add, BatchNormalization, LayerNormalization
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from cgb_utils import cp2rep, rep2cp, BBZ, MROT270
from md_layer import *


model_options_choice = {
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'batch_size': [512, 1024, 2048],
    'l2_weight_regularizer': [1e-3, 1e-4, 1e-5],
    'n_epochs': 10000,
    'layer_width': [128, 256, 512],
    'network_depth': [2, 4, 6],
    'use_residual_connections': [True, False],
    'normalization': [None, 'batch', 'layer'],
    'activation': ['relu', 'elu', 'swish'],
    'n_components': [2, 3],
    'output_dimension': 13,
    'input_canonicalization': ['canonical', '4x', '8x'],
}

default_model_options = {
    'learning_rate': 1e-4,  # [1e-2, 1e-3, 1e-4]
    'batch_size': 2048,  # [512, 1024, 2048]
    'l2_weight_regularizer': 1e-4,  # [1e-3, 1e-4, 1e-5]
    'n_epochs': 10000,
    'layer_width': 512,  # [128, 256, 512]
    'network_depth': 6,  # [2, 4, 6]
    'use_residual_connections': True,  # [True, False]
    'normalization': 'layer',  # [None, 'batch', 'layer']
    'activation': 'elu',  # ['relu', 'elu', 'swish']
    'n_components': 2,  # [2, 3]
    'output_dimension': 13,
    'input_canonicalization': '4x',  # ['canonical', '4x', '8x']
}


def swish(self, x, beta=1):
    return x * keras.backend.sigmoid(beta * x)


### epoch timing log
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, log_path='timing.log', logs=None):
        super().__init__()
        self.log_path = log_path
        with open(log_path, 'w') as f:
            f.write('epoch,time\n')

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_path, 'a') as f:
            f.write(f'{epoch},{time.time() - self.start_time}\n')


class EmergencyBrakeCallback(tf.keras.callbacks.Callback):
    def __init__(self, dname):
        super().__init__()
        self.fname = path.join(dname, 'emergency_brake.flag')

    def on_train_begin(self, logs=None):
        open(self.fname, 'w')

    def on_epoch_end(self, epoch, logs=None):
        if not path.exists(self.fname):
            print("\nEmergency brake pulled, stopping training!")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if path.exists(self.fname):
            os.remove(self.fname)


class PanelPredictor:


    @staticmethod
    def choose_model_options():
        mopt = {}
        for option, values in model_options_choice.items():
            if isinstance(values, list):
                mopt[option] = random.choice(values)
            else:
                mopt[option] = values
        return mopt


    def __init__(self):
        self.mopt = None
        self.x_train = None
        self.y_train = None
        self.scalers = None
        self.cb_list = None
        self.history = None
        self.model = None
        self.output_dims = None #TODO: currnetly unused, use this instead of mopt['output_dimension']
        self.model_path = None


    def create(self, dir_name, model_path):
        self.mopt = json.load(open(path.join(model_path, 'model_options.json')))
        mopt = self.mopt

        activation = mopt['activation']
        if activation == 'swish':
            activation = swish

        self.x_train = np.load(path.join(dir_name, 'x_train.npy'))
        self.y_train = np.load(path.join(dir_name, 'y_train.npy'))

        self.output_dims = self.y_train.shape[1]

        # load saved scalers or compute them
        self.scalers = {}
        train_io = {'in': self.x_train, 'out': self.y_train}
        for io in ['in', 'out']:
            scaler_fname = path.join(model_path, f'scaler_{io}.pkl')
            if path.exists(scaler_fname):
                self.scalers[io] = pickle.load(open(scaler_fname, 'rb'))
            else:
                self.scalers[io] = StandardScaler()
                self.scalers[io].fit(train_io[io])
                pickle.dump(self.scalers[io], open(scaler_fname, 'wb'))

        # scale data
        self.x_train = self.scalers['in'].transform(self.x_train)
        self.y_train = self.scalers['out'].transform(self.y_train)

        # input layer
        inputs = Input(shape=(self.x_train.shape[1],))

        # loop over hidden layers
        x = inputs
        for i in range(mopt['network_depth']):
            x_prev = x
            x = Dense(mopt['layer_width'], activation=activation,
                      kernel_regularizer=l2(mopt['l2_weight_regularizer']))(x)
            if i > 0 and mopt['use_residual_connections']:
                x = Add()([x, x_prev])
            # normalize
            if mopt['normalization'] == 'batch':
                x = BatchNormalization()(x)
            elif mopt['normalization'] == 'layer':
                x = LayerNormalization()(x)
            elif mopt['normalization'] is None:
                pass

        # final layer is the MDN layer
        outputs = MDN(mopt['output_dimension'], mopt['n_components'])(x)

        # define model
        self.model = Model(inputs=inputs, outputs=outputs)

        # compile model
        self.model.compile(loss=get_mixture_loss_func(mopt['output_dimension'], mopt['n_components']),
                           optimizer=Adam(mopt['learning_rate']))

        # save model summary
        with open(path.join(model_path, 'model_summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # callbacks
        self.cb_list = [
            ModelCheckpoint(path.join(model_path, 'model.hdf5'), monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min', period=1),
            CSVLogger(path.join(model_path, 'training.log'), separator=',', append=False),
            TimingCallback(log_path=path.join(model_path, 'timing.log')),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=400, verbose=0, mode='auto'),
            EmergencyBrakeCallback(dname=model_path),
        ]

        self.model_path = model_path


    def train(self, model_path):
        # fit the keras model on the dataset
        self.history = self.model.fit(self.x_train, self.y_train, epochs=self.mopt['n_epochs'],
                                      batch_size=self.mopt['batch_size'], validation_split=0.1, shuffle=True,
                                      callbacks=self.cb_list, verbose=0)
        # save history
        pickle.dump(self.history.history, open(path.join(model_path, 'history.pkl'), 'wb'))


    def load(self, dir_name):
        self.scalers = {
            'in': pickle.load(open(path.join(dir_name, 'scaler_in.pkl'), 'rb')),
            'out': pickle.load(open(path.join(dir_name, 'scaler_out.pkl'), 'rb')),
        }
        with open(path.join(dir_name, 'model_options.json')) as fin:
            self.mopt = json.load(fin)
        self.model = load_model(path.join(dir_name, 'model.hdf5'),
                                custom_objects={'MDN': MDN,
                                                'mdn_loss_func': get_mixture_loss_func(
                                                    self.mopt['output_dimension'],
                                                    self.mopt['n_components']),
                                                'swish': swish})
        self.output_dims = (self.model.layers[-1].output_shape[1] / self.mopt['n_components'] - 1) / 2


    def predict(self, cp):
        if (not len(cp.shape) in [2, 3]) or (cp.shape[-2:] != (16, 3)):
            raise ValueError(f'Wrong input shape: {cp.shape}, must be (x, 16, 3) or (16, 3)')
        input_dims = len(cp.shape)
        if input_dims == 2:
            cp = cp.reshape((1, 16, 3))

        reps, bcs, frames, nca = cp2rep(cp)

        x = reps[:,:18]
        x = self.scalers['in'].transform(x)
        pred = self.model.predict(x)

        n_components = self.mopt['n_components']
        mus, sigs, pi_logits = split_mixture_params(pred, self.mopt['output_dimension'], n_components)

        mus = mus.reshape(-1, n_components, self.mopt['output_dimension'])
        #     sigs = sigs.reshape(-1, self.n_components, self.mopt['output_dimension']

        for i in range(n_components):
            mus[:,i,:] = self.scalers['out'].inverse_transform(mus[:,i,:])

        # Mixing coefficients
        pis = softmax(pi_logits, axis=1)

        # Computations for all components
        reps_comp = np.repeat(reps[:, :18], repeats=n_components, axis=0)
        bcs_comp = np.repeat(bcs, repeats=n_components, axis=0)
        frames_comp = np.repeat(frames, repeats=n_components, axis=0)
        rep_pred = np.c_[reps_comp, mus[:,:,:-1].reshape((-1, 12))]
        cp_pred = rep2cp(rep_pred, bcs_comp, frames_comp)
        cp_pred = cp_pred.reshape((-1, n_components, 16, 3))
        cp_pred[nca] = cp_pred[nca][:,:,MROT270]
        stress_pred = mus[:,:,-1]

        for mx in range(n_components):
            cp_pred[:,mx,BBZ] = cp[:,BBZ]

        if input_dims == 2:
            cp_pred = cp_pred.reshape(cp_pred.shape[1:])
            stress_pred = stress_pred.reshape(stress_pred.shape[1:])
            pis = pis.reshape(pis.shape[1:])

        return cp_pred, stress_pred, pis


    def jacobian(self, x):
        """ Jacobian wrt raw variables, not tested """
        x = self.scalers['in'].transform(x)

        inp = tf.Variable(x, dtype=tf.float32)

        # start recording and forward-pass
        with tf.GradientTape() as tape:
            tape.watch(inp)
            pred = self.model(inp)

        # compute jacobians
        jac = tape.batch_jacobian(pred, inp, experimental_use_pfor=True)

        # convert prediction tensor to numpy array
        #pred = pred.numpy()

        # get jacobians of shape and stress (ignore mixing coefficients and sigmas)
        mus_j = jac.numpy()[:,:self.mopt['output_dimension']*self.mopt['n_components']]\
            .reshape(-1, self.mopt['n_components'], self.mopt['output_dimension'], x.shape[1])

        mus_j *= self.scalers['out'].scale_.reshape(1,1,self.mopt['output_dimension'],1)
        mus_j /= self.scalers['in'].scale_.reshape(1,1,1,18)

        j_bezier = mus_j[:,:,:-1]
        j_stress = mus_j[:,:,[-1]]

        return j_bezier, j_stress
