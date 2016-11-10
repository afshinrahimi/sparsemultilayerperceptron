# sparsemultilayerperceptron
Lasagne / Theano based MultiLayer Perceptron MLP which accepts both sparse and dense matrices and is very easy to use with scikit-learn api similarity.

# Requirements
Lasagne/Theano

# Easy to use
        from mlp import MLP
        clf = MLP(n_epochs=100, 
                  batch_size=100, 
                  init_parameters=None, 
                  complete_prob=False, 
                  add_hidden=True, 
                  regul_coefs=[1e-5, 1e-5], 
                  save_results=False, 
                  hidden_layer_size=hiddern_size, 
                  drop_out=True, 
                  drop_out_coefs=[0.5, 0.5],
                  early_stopping_max_down=10,
                  loss_name=loss_function,
                  nonlinearity='rectify')
        clf.fit(X_train, Y_train, X_dev, Y_dev)
        acc = clf.accuracy(X_test, Y_test)

# Features
supports both dense and sparse matrices

supports drop-out and hidden layer

Supports complete probability distribution instead of one-hot labels so supports multilabel training.

Supports scikit-learn like API (fit, predict, accuracy, etc.)

Is very easy to configure and modify
