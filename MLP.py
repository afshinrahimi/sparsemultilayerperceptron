'''
Created on 22 Apr 2016
@author: af
'''

import pdb
import numpy as np
import sys
from os import path
import scipy as sp
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer
import logging
import json
import codecs
import pickle
import gzip
from collections import OrderedDict
from _collections import defaultdict
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

'''
These sparse classes are copied from https://github.com/Lasagne/Lasagne/pull/596/commits
'''
class SparseInputDenseLayer(DenseLayer):
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        activation = S.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
class SparseInputDropoutLayer(DropoutLayer):
    def get_output_for(self, input, deterministic=False, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        if deterministic or self.p == 0:
            return input
        else:
            # Using Theano constant to prevent upcasting
            one = T.constant(1, name='one')
            retain_prob = one - self.p

            if self.rescale:
                input = S.mul(input, one/retain_prob)

            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=input.dtype)



#copied from a tutorial that I don't rememmber!
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]





class MLP():
    def __init__(self, 
                 n_epochs=10, 
                 batch_size=1000, 
                 init_parameters=None, 
                 complete_prob=False, 
                 add_hidden=True, 
                 regul_coefs=[5e-5, 5e-5], 
                 save_results=False, 
                 hidden_layer_size=None, 
                 drop_out=False, 
                 drop_out_coefs=[0.5, 0.5],
                 early_stopping_max_down=100000,
                 loss_name='log',
                 nonlinearity='rectify'):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_parameters = init_parameters
        self.complete_prob = complete_prob
        self.add_hidden = add_hidden
        self.regul_coefs = regul_coefs
        self.save_results = save_results
        self.hidden_layer_size = hidden_layer_size
        self.drop_out = drop_out
        self.drop_out_coefs = drop_out_coefs
        self.early_stopping_max_down = early_stopping_max_down
        self.loss_name = loss_name
        self.nonlinearity = 'rectify'

    def fit(self, X_train, Y_train, X_dev, Y_dev):
        logging.info('building the network...' + ' hidden:' + str(self.add_hidden))
        in_size = X_train.shape[1]
        best_params = None
        best_val_acc = 0.0
        drop_out_hid, drop_out_in = self.drop_out_coefs
        if self.complete_prob:
            out_size = Y_train.shape[1]
        else:
            out_size = len(set(Y_train.tolist()))
        
        if self.hidden_layer_size:
            pass
        else:
            self.hidden_layer_size = min(5 * out_size, int(in_size / 20))
        logging.info('input layer size: %d, hidden layer size: %d, output layer size: %d'  %(X_train.shape[1], self.hidden_layer_size, out_size))
        # Prepare Theano variables for inputs and targets
        if not sp.sparse.issparse(X_train):
            logging.info('input matrix is not sparse!')
            self.X_sym = T.matrix()
        else:
            self.X_sym = S.csr_matrix(name='inputs', dtype='float32')
        
        if self.complete_prob:
            self.y_sym = T.matrix()
        else:
            self.y_sym = T.ivector()    
        
        l_in = lasagne.layers.InputLayer(shape=(None, in_size),
                                         input_var=self.X_sym)
        
        if self.nonlinearity == 'rectify':
            nonlinearity = lasagne.nonlinearities.rectify
        elif self.nonlinearity == 'sigmoid':
            nonlinearity = lasagne.nonlinearities.sigmoid
        elif self.nonlinearity == 'tanh':
            nonlinearity = lasagne.nonlinearities.tanh
        else:
            nonlinearity = lasagne.nonlinearities.rectify

        if self.drop_out:
            l_in = lasagne.layers.dropout(l_in, p=drop_out_in)
    
        if self.add_hidden:
            if not sp.sparse.issparse(X_train):
                l_hid1 = lasagne.layers.DenseLayer(
                    l_in, num_units=self.hidden_layer_size,
                    nonlinearity=nonlinearity,
                    W=lasagne.init.GlorotUniform())
            else:
                l_hid1 = SparseInputDenseLayer(
                    l_in, num_units=self.hidden_layer_size,
                    nonlinearity=nonlinearity,
                    W=lasagne.init.GlorotUniform())
            if self.drop_out:
                self.l_hid1 = lasagne.layers.dropout(l_hid1, drop_out_hid)
            
            self.l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=out_size,
            nonlinearity=lasagne.nonlinearities.softmax)
        else:
            if not sp.sparse.issparse(X_train):
                self.l_out = lasagne.layers.DenseLayer(
                    l_in, num_units=out_size,
                    nonlinearity=lasagne.nonlinearities.softmax)
                if self.drop_out:
                    l_hid1 = lasagne.layers.dropout(l_hid1, drop_out_hid)
            else:
                self.l_out = SparseInputDenseLayer(
                    l_in, num_units=out_size,
                    nonlinearity=lasagne.nonlinearities.softmax)
                if self.drop_out:
                    l_hid1 = SparseInputDropoutLayer(l_hid1, drop_out_hid)
        
        
    
        if self.add_hidden:
            self.embedding = lasagne.layers.get_output(l_hid1, self.X_sym, deterministic=True)
            self.f_get_embeddings = theano.function([self.X_sym], self.embedding)
        self.output = lasagne.layers.get_output(self.l_out, self.X_sym, deterministic=True)
        self.pred = self.output.argmax(-1)
        if self.loss_name == 'log':
            loss = lasagne.objectives.categorical_crossentropy(self.output, self.y_sym)
        elif self.loss_name == 'hinge':
            loss = lasagne.objectives.multiclass_hinge_loss(output, y_sym)
        loss = loss.mean()
        
        
        l1_share_out = 0.5
        l1_share_hid = 0.5
        regul_coef_out, regul_coef_hid = self.regul_coefs
        logging.info('regul coefficient for output and hidden layers are ' + str(self.regul_coefs))
        l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
        l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1-l1_share_out)
        if self.add_hidden:
            l1_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l1) * regul_coef_hid * l1_share_hid
            l2_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l2) * regul_coef_hid * (1-l1_share_hid)
        loss = loss + l1_penalty + l2_penalty
    
        if self.complete_prob:
            self.y_sym_one_hot = self.y_sym.argmax(-1)
            self.acc = T.mean(T.eq(self.pred, self.y_sym_one_hot))
        else:
            acc = T.mean(T.eq(self.pred, self.y_sym))
        if self.init_parameters:
            lasagne.layers.set_all_param_values(self.l_out, self.init_parameters)
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        
        #print(params)
        #updates = lasagne.updates.nesterov_momentum(loss, parameters, learning_rate=0.01, momentum=0.9)
        #updates = lasagne.updates.sgd(loss, parameters, learning_rate=0.01)
        #updates = lasagne.updates.adagrad(loss, parameters, learning_rate=0.1, epsilon=1e-6)
        #updates = lasagne.updates.adadelta(loss, parameters, learning_rate=0.1, rho=0.95, epsilon=1e-6)
        updates = lasagne.updates.adam(loss, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
        self.f_train = theano.function([self.X_sym, self.y_sym], [loss, acc], updates=updates)
        self.f_val = theano.function([self.X_sym, self.y_sym], [loss, acc])
        self.f_predict = theano.function([self.X_sym], self.pred)
        self.f_predict_proba = theano.function([self.X_sym], self.output)
        
        
        X_train = X_train.astype('float32')
        X_dev = X_dev.astype('float32')
    
        if self.complete_prob:
            Y_train = Y_train.astype('float32')
            Y_dev = Y_dev.astype('float32')
        else:
            Y_train = Y_train.astype('int32')
            Y_dev = Y_dev.astype('int32')
    
        logging.info('training (n_epochs, batch_size) = (' + str(self.n_epochs) + ', ' + str(self.batch_size) + ')' )
        n_validation_down = 0
        for n in xrange(self.n_epochs):
            for batch in iterate_minibatches(X_train, Y_train, self.batch_size, shuffle=True):
                x_batch, y_batch = batch
                l_train, acc_train = self.f_train(x_batch, y_batch)
                l_val, acc_val = self.f_val(X_dev, Y_dev)
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_params = lasagne.layers.get_all_param_values(self.l_out)
                n_validation_down = 0
            else:
                #early stopping
                n_validation_down += 1
            logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train) + ' ,acc ' + str(acc_train) + ' ,val_loss ' + str(l_val) + ' ,acc ' + str(acc_val) + ',best_val_acc ' + str(best_val_acc))
            if n_validation_down > self.early_stopping_max_down:
                logging.info('validation results went down. early stopping ...')
                break
        
        lasagne.layers.set_all_param_values(self.l_out, best_params)
        
        logging.info('***************** final results based on best validation **************')
        l_val, acc_val = self.f_val(X_dev, Y_dev)
        logging.info('Best dev acc: %f' %(acc_val))
        
    def predict(self, X_test):
        X_test = X_test.astype('float32')
        return self.f_predict(X_test)
    
    def predict_proba(self, X_test):
        X_test = X_test.astype('float32')
        return self.f_predict_proba(X_test)
    
    def accuracy(self, X_test, Y_test):
        X_test = X_test.astype('float32')
        if self.complete_prob:
            Y_test = Y_test.astype('float32')
        else:
            Y_test = Y_test.astype('int32')
        test_loss, test_acc = self.f_val(X_test, Y_test)
        return test_acc
    
    def score(self, X_test, Y_test):
        return self.accuracy(self, X_test, Y_test)      
        
        
if __name__ == '__main__':
    pass            
