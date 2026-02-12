import os
import warnings

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf

from tensorflow import keras
import tensorflow_probability as tfp
from keras import optimizers, initializers

from scipy.stats import norm

from tqdm.keras import TqdmCallback

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

class ModelNN(keras.models.Model):

    def __init__(self, parameters, loglikelihood_loss, neural_network_structure = None, neural_network_call = None,  neural_network_call_nolast = None,
                 input_dim = None, seed = None):
        super().__init__()
        self.parameters = parameters
        self.loglikelihood_loss = loglikelihood_loss
        self.neural_network_structure = neural_network_structure
        self.neural_network_call = neural_network_call
        self.neural_network_call_nolast = neural_network_call_nolast
        self.n_acum_step = tf.Variable(0, dtype = tf.int32, trainable = False)

        self.input_dim = input_dim
        self.seed = seed

        self.total_hessian = None
        self.weights_covariance = None
        
        self.define_structure()
    
    def define_structure(self):
        # Goes through the list of parameters for the model and filter them by their classes:
        # - "nn" will be treated as an output from a given neural network that receives the variables x as input.
        # - "independet" will be treated an an individual tf.Variable, trainable object. It is still trained in tensorflow, but is constant for all subjects
        # - "fixed" will be treated as a non-trainable tf.Variable. Basically just a known constant.
        # - "manual" will be treated as a non-trainable tf.Variable, but its value will be eventually updated manually using user provided functions (useful in cases where closed forms can be obtained)
        # - "dependent" will be treated simply as a deterministic function of other parameters and will be updated after training

        self.nn_pars = []
        self.independent_pars = []
        self.fixed_pars = []
        self.manual_pars = []
        for parameter in self.parameters:
            par = self.parameters[parameter]
            if(par["par_type"] == "nn"):
                self.nn_pars.append( parameter )
            elif(par["par_type"] == "independent"):
                self.independent_pars.append( parameter )
            elif(par["par_type"] == "fixed"):
                self.fixed_pars.append( parameter )
            elif(par["par_type"] == "manual"):
                self.manual_pars.append( parameter )
            else:
                raise Exception("Invalid parameter {} type: {}".format(parameter, par["par_type"]))

        # If at least one parameter is to be modeled as a neural network output, define its architecture here
        if( len(self.nn_pars) > 0 ):
            if(self.neural_network_structure is None):
                raise Exception("Parameters {} defined as 'nn'. Please, provide a structure for their neural network.".format(self.nn_pars))
            # Define the neural network structure based on the user's input
            self.neural_network_structure(self, self.seed)

            # It may be the case that the user includes a neural network component, but does not want it to be trainable.
            # Then they would set all its layers as trainable = False, but we would still detect len(self.nn_pars) > 0 and break training
            # To resolve that, we can count how many layers are trainable. If none is trainable, we also set self.neural_network_use to False,
            # as there would be no neural network weights to be trained
            at_least_one_trainable_layer = False
            for layer in self.layers:
                if(layer.trainable):
                    at_least_one_trainable_layer = True

            # If there is at least a single layer to be trained, use the neural network structure. Otherwise, do not bother to define anything
            if(at_least_one_trainable_layer):
                self.neural_network_use = True
            else:
                self.neural_network_use = False
        else:
            # If no parameter depends on the neural network component, we simply do not create any component for that
            self.neural_network_use = False

        # False if no independent parameter is defined
        self.independent_pars_use = len(self.independent_pars) > 0
        
        # Dictionary with all parameters that are its individual weights
        self.model_variables = {}

        # For the independent parameters covariance afterward, it is useful to know which parameter we are considering by each index of weight
        # over the final trained model. For example, if we have three parameters modeled as independent weights:
        # alpha (single value) ; beta (2 elements vector) ; gamma(single value),
        # then,
        # independent_index_to_vars[0] = "alpha"
        # independent_index_to_vars[1] = "beta[0]"
        # independent_index_to_vars[2] = "beta[1]"
        # independent_index_to_vars[3] = "gamma"
        # That answers the question: "Which parameter does this index correspond to?"
        self.independent_index_to_vars = {}
        independet_par_index = 0
        
        # Include variables that do not depend on the variables x, but are still trained by tensorflow
        for parameter in self.independent_pars:
            par = self.parameters[parameter]

            # If shape is None, set it to 1
            if(par["shape"] is None):
                par["shape"] = 1
                
            raw_init = par["link_inv"]( par["init"] )

            # Name for the new, transformed parameter
            raw_parameter = "raw_" + parameter
            self.model_variables[raw_parameter] = self.add_weight(
                name = raw_parameter,
                shape = np.atleast_1d( par["shape"] ),
                initializer = keras.initializers.Constant( raw_init ),
                trainable = True,
                dtype = tf.float32
            )

            if(par["shape"] == 1):
                self.independent_index_to_vars[independet_par_index] = "raw_" + parameter
            else:
                for j in range(par["shape"]):
                    self.independent_index_to_vars[independet_par_index+j] = "raw_" + parameter + "[" + str(j) + "]"
            independet_par_index += par["shape"]

        # Number of independent parameters outputs
        self.independent_output_size = sum( [self.parameters[par]["shape"] for par in self.independent_pars] ) # Number of independent outputs (b)

        # Include variables that are not trained by tensorflow (known, fixed constants or manual trained variables)
        for parameter in np.concatenate([self.fixed_pars, self.manual_pars]):
            par = self.parameters[parameter]
            
            raw_parameter = "raw_" + parameter
            raw_init = par["link_inv"]( par["init"] )
            
            self.model_variables[raw_parameter] = self.add_weight(
                name = raw_parameter,
                shape = par["shape"],
                initializer = keras.initializers.Constant( raw_init ),
                trainable = False,
                dtype = tf.float32
            )

        # Organize trainable variables information, so each variable can get mapped to an index in the self.trainable_variables and its gradients
        self.vars_to_index = {}
        # Before we build the model, the only variables that appear in here are the ones corresponding to "independent" parameters
        for i, var in enumerate(self.trainable_variables): 
            # From the variable path, get its name (raw_<variable>)
            var_name = var.path.split("/")[-1]
            # Save its corresponding index
            self.vars_to_index[var_name] = i
            
        # For the neural network parameter, it is useful to know which parameter we are considering by giving its corresponding index
        # over the final nn output. For example, if we have two parameters modeled as a nn output:
        # alpha (single value) ; beta (2 elements vector) ; gamma(single value),
        # then,
        # nn_index_to_vars[0] = "alpha"
        # nn_index_to_vars[1] = "beta[0]"
        # nn_index_to_vars[2] = "beta[1]"
        # nn_index_to_vars[3] = "gamma"
        # That answers the question: "Which parameter does this index correspond to?"
        self.nn_index_to_vars = {}
        nn_par_index = 0
        
        # We must also include in this list the indices for "nn" parameters
        for i, parameter in enumerate(self.nn_pars):
            par = self.parameters[ parameter ]
            if(par["shape"] is None):
                par_shape = 1
            else:
                # The parameter must be at most a 1-dimensional array, whose indices will be saved for future location in the neural network output results
                par_shape = par["shape"]

            # The indices corresponding to par in the output are given by the current index plus the dimension of par
            self.vars_to_index["raw_" + parameter] = tf.constant( np.arange(nn_par_index, nn_par_index+par_shape), dtype = tf.int32 )
            if(par_shape == 1):
                self.nn_index_to_vars[nn_par_index] = "raw_" + parameter
            else:
                for j in range(par_shape):
                    self.nn_index_to_vars[nn_par_index+j] = "raw_" + parameter + "[" + str(j) + "]"
                    
            nn_par_index += par_shape

        # Number of outputs to our neural network
        self.nn_output_size = nn_par_index # Number of outputs to the neural network (d)

        # ALERT!!
        # If output dimension does not match this value it may be interesting to add an alerto for that!
        
        # Once the entire structure has been defined, force the model to build all the weights properly
        if(self.neural_network_use):
            dummy_input = keras.Input(self.input_dim)
            self(dummy_input)
        
        # In the future, it might be interesting to allow the user to specify an optimizer for each single parameter in the model.
        # For now, they will specify one for the independent parameters and other for the neural network weights

        # Not that the model is built and all the trainable variables instantiated, we define the gradient variables

        # Only create the gradient accumulator if independent parameters are in use
        if(self.independent_pars_use):
            self.gradient_accumulation_independent_pars = [
                tf.Variable(tf.zeros_like(v, dtype = tf.float32), trainable = False) for v in self.trainable_variables[ :len(self.independent_pars) ]
            ]

        # Only create the gradient accumulator if neural network is in use
        if(self.neural_network_use):
            # The gradient values for the neural network component always comes right after the weights for the independent parameters
            self.gradient_accumulation_nn = [
                tf.Variable(tf.zeros_like(v, dtype = tf.float32), trainable = False) for v in self.trainable_variables[ len(self.independent_pars): ]
            ]

        if( len(self.trainable_variables) == 0 ):
            warnings.simplefilter("always", UserWarning)
            warnings.warn(
                "The model does not contain any trainable variables.\n" + \
                "It can be evaluated but does not require training.",
                category = UserWarning,
            )
            warnings.simplefilter("default", UserWarning)
            

    def copy(self):
        new_model = FrailtyModelNN(parameters = self.parameters,
                                   loglikelihood_loss = self.loglikelihood_loss,
                                   neural_network_structure = self.neural_network_structure,
                                   neural_network_call = self.neural_network_call,
                                   input_dim = self.input_dim, seed = self.seed)        
        new_model.set_weights( self.get_weights() )
        return new_model

    def call(self, x_input, training = True):
        if(self.neural_network_call is None):
            return None
        x = self.neural_network_call(self, x_input)
        if(training):
            return x

        # If not on training mode, returns the neural network as a 4 indices tensor for plotting (better to change this in the future!)
        return tf.reshape(x, (x.shape[0], x.shape[1], 1, 1))

    def get_feature_extractor(self):
        """
            Create a new model that mimics the self.neural_network_call function provided by the user, but stops exactly before the last layer execution. 
            This function is used to obtain a confidence interval for the output based on the weights from the previous layer.
        """
        dummy_input = keras.Input(shape = self.input_dim)
        
        # The function runs, but instead of calculating numbers, it builds a connection graph.
        final_call_function = self.neural_network_call(self, dummy_input)
        
        # Every Keras tensor knows its history via '_keras_history'. It returns (Layer, NodeIndex, TensorIndex)
        last_layer, node_index, _ = final_call_function._keras_history
        
        # Get the output from the penultimate layer from the network architecture
        penultimate_tensor = last_layer.get_input_at(node_index)
        
        # 5. Build the New Model
        # This model takes x -> network -> stops before last layer -> returns features
        feature_extractor = keras.Model(inputs = dummy_input, outputs = penultimate_tensor)
        
        return feature_extractor

    def get_variable(self, parameter, nn_output = None):
        """
            Once that all variables have been properly defined and mapped, this method uses their proper link functions to transform from
            the variables 'raw' state into their proper values used in the likelihood.

            If nn_output is passed, we automatically assume that the parameter is an output from the neural network and proceed by taking its
            value differently than if it was an independent parameter.
        """
        # Get the raw name for that parameter
        raw_parameter = "raw_" + parameter
        # Filter the desired parameter from the list
        par = self.parameters[parameter]

        # If nn_output is None, assume the parameter is independent from the data x and get it directly as a transformed weight
        if(nn_output is None):            
            # Get the transformed parameter from its raw version, considering its proper link function
            par_value = par["link"]( self.model_variables[raw_parameter] )
            # return self.format_variable( par_value )
            return par_value
        
        # If nn_output is not None, assume the parameter came as a neural network output and return it from its positions in the output
        par_value = par["link"]( tf.gather(nn_output, self.vars_to_index[raw_parameter], axis = 1) )
        
        return par_value

    def train_step(self, data):
        """
            Called by each batch in order to evaluate the loglikelihood and accumulate the parameters gradients using training data.
        """
        # The first component from data is always the nn-variables. If there is no neural network involved, data[0] is None
        x = data[0]

        self.n_acum_step.assign_add(1)
        with tf.GradientTape() as tape:
            # If there is a neural network structure, call it. If not, this is simply a None object from self.call
            nn_output = self(x, training = True)

            # I'm interested in flexibilizing this part so that self.loglikelihood_loss receives a general,
            # data file which is only decompressed inside the function
            loss_value = self.loglikelihood_loss(self, nn_output = nn_output, data = data)

        # The first weights are always destined to the independent parameters
        # The neural network related weights comes after those in the self.trainable_variables object
        gradients = tape.gradient(loss_value, self.trainable_variables)

        # If the loss does not depend on a specific parameter, its corresponding gradient will be None
        # To avoid crash problems in that case, we simply replace None with a zero like gradient, so those weights do not get updated
        # It is the user's responsibility to build a loss that depends on all the trainable parameters, but we allow that to happen in this case
        # for generality and to avoid unneccessary crashes when testing new models
        gradients = [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(gradients, self.trainable_variables)
        ]

        independent_gradients = gradients[ :len(self.independent_pars) ]
        nn_gradients = gradients[ len(self.independent_pars): ]

        # Only cumulate independent gradients if in use
        if(self.independent_pars_use):
            for i in range( len(self.gradient_accumulation_independent_pars) ):
                self.gradient_accumulation_independent_pars[i].assign_add( independent_gradients[i] )

        # Only cumulate neural network gradients if in use
        if(self.neural_network_use):
            for i in range( len(self.gradient_accumulation_nn) ):
                self.gradient_accumulation_nn[i].assign_add( nn_gradients[i] )

        tf.cond(tf.equal(self.n_acum_step, self.gradient_accumulation_steps), self.apply_accumulated_gradients, lambda: None)

        return {"likelihood_loss": loss_value}

    def test_step(self, data):
        x = data[0]
        nn_output = self(x, training = True)
        likelihood_loss = self.loglikelihood_loss(model = self, nn_output = nn_output, data = data)
        return {"likelihood_loss": likelihood_loss}

    def apply_accumulated_gradients(self):
        # ----------------------------------- Independent parameters component -----------------------------------
        if(self.independent_pars_use):
            # Apply the accumulated gradients to the trainable variables
            self.optimizer_independent_pars.apply_gradients( zip(self.gradient_accumulation_independent_pars, self.trainable_variables[ :len(self.independent_pars) ]) )
            # Resets all the cumulated gradients to zero
            for i in range(len(self.gradient_accumulation_independent_pars)):
                self.gradient_accumulation_independent_pars[i].assign(tf.zeros_like(self.trainable_variables[ :len(self.independent_pars) ][i], dtype = tf.float32))

        # Only update neural network weights if in use.
        if(self.neural_network_use):
            # ----------------------------------- Neural network component -----------------------------------
            self.optimizer_nn.apply_gradients( zip(self.gradient_accumulation_nn, self.trainable_variables[ len(self.independent_pars): ]) )
            # Resets all the cumulated gradients to zero
            for i in range(len(self.gradient_accumulation_nn)):
                self.gradient_accumulation_nn[i].assign(tf.zeros_like(self.trainable_variables[ len(self.independent_pars): ][i], dtype = tf.float32))

        # Reset the gradient accumulation steps counter to zero
        self.n_acum_step.assign(0)

    def compile_model(self, optimizer_independent_pars, optimizer_nn, run_eagerly):
        """
            Defines the configuration for the model, such as batch size, training mode, early stopping.
        """
        # optimizers.Adam(learning_rate = learning_rate, gradient_accumulation_steps = None),
        self.optimizer_independent_pars = optimizer_independent_pars
        self.optimizer_nn = optimizer_nn
        self.compile(
            run_eagerly = run_eagerly
        )
        
    def train_model(self, x, data,
                     epochs, shuffle,
                     validation = False, val_prop = None, x_val = None, data_val = None,
                     optimizer_independent_pars = optimizers.Adam(learning_rate = 0.001),
                     optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                     train_batch_size = None, val_batch_size = None,
                     buffer_size = 4096, gradient_accumulation_steps = None,
                     early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10, early_stopping_warmup = 0,
                     run_eagerly = True, verbose = 1):

        # If there are no trainable variables, there is no reason to train such a model
        if( len(self.trainable_variables) == 0 ):
            raise RuntimeError(
                "Training failed: the model does not contain any trainable variables. "
                "This model is fixed and cannot be trained."
            )
        
        self.validation = validation

        # Cast the neural network input to tf.float32 if x is given
        if(x is not None):
            x = tf.cast(x, dtype = tf.float32)
            # If input is a vector, transform it into a column
            if(len(x.shape) == 1):
                x = tf.reshape( x, shape = (len(x), 1) )

        # Cast all variables from data to tf.float32 and pass them to tf.arrays if neccessary
        for i in range(len(data)):
            data[i] = tf.cast(data[i], dtype = tf.float32)
            if(len(data[i].shape) == 1):
                data[i] = tf.reshape( data[i], shape = (len(data[i]), 1) )

        # Save original processed data in object
        self.x = x
        self.data = data
        self.n = len(data[0]) # Sample size

        if(self.validation):
            # If all validation data was given
            if(x_val is not None and t_val is not None and delta_val is not None):
                x_val = tf.cast(x_val, dtype = tf.float32)
                # If input is a vector, transform it into a column
                if(len(x_val.shape) == 1):
                    x_val = tf.reshape( x_val, shape = (len(x_val), 1) )
                
                # Cast all variables from data to tf.float32 and pass them to tf.arrays if neccessary
                for i in range(len(data)):
                    data[i] = tf.cast(data[i], dtype = tf.float32)
                    if(len(data[i].shape) == 1):
                        data[i] = tf.reshape( data[i], shape = (len(data[i]), 1) )
                
                self.x_val, self.data_val = x_val, data_val
                self.x_train, self.data_train = self.x, self.data
            else:
                # If validation is desired, but no data was given, select val_prop * 100% observations as validation set
                # Take the first list from data for indices
                self.indexes_train = np.arange( self.n )
                if(shuffle):
                    self.indexes_train = tf.random.shuffle( self.indexes_train )

                if(self.x is not None):
                    x_shuffled = tf.gather( self.x, self.indexes_train )
                
                data_shuffled = []
                for i in range(len(data)):
                    data_shuffled_i = tf.gather( data[i], self.indexes_train )
                    data_shuffled.append( data_shuffled_i )

                if(val_prop is None):
                    raise Exception("Please, provide the size of the validation set (between 0 and 1).")
                # Selects the subsample as validation data
                val_size = int(self.n * val_prop)
                self.n_val = val_size
                self.n_train = self.n - self.n_val

                self.x_val = None
                self.x_train = None
                if(self.x is not None):
                    self.x_val = x_shuffled[:val_size]
                    self.x_train = x_shuffled[val_size:]

                data_train = []
                data_val = []
                # For each variable in data, separate into train and test
                for i in range(len(data)):
                    data_train.append( data_shuffled[i][:val_size] )
                    data_val.append( data_shuffled[i][val_size:] )

                self.data_train, self.data_val = data_train, data_val
        else:
            # If no validation step should be taken, training data is the same as validation data
            self.n_train = self.n
            self.n_val = 0
            self.x_train, self.data_train = self.x, self.data
            self.x_val, self.data_val = self.x, self.data
        
        # Declara os callbacks do modelo
        self.callbacks = [ ]
        
        if(verbose >= 1):
            self.callbacks.append( TqdmCallback(verbose = 0, position = 0, leave = True) )
        
        if(early_stopping):
            # Avoids overfitting and speeds training
            if(self.validation):
                metric = "val_likelihood_loss"
            else:
                metric = "likelihood_loss"
            es = keras.callbacks.EarlyStopping(monitor = metric,
                                               mode = "min",
                                               start_from_epoch = early_stopping_warmup,
                                               min_delta = early_stopping_min_delta,
                                               patience = early_stopping_patience,
                                               restore_best_weights = True)
            self.callbacks.append(es)

        # If batch_size is unspecified, set it to be the training size. Note that decreasing the batch size to smaller values, such as 500 for example, has previously lead the model to converge too early, leading to a lot of time of investigation.
        # When dealing with neural networks in the statistical models context, we recommend to use a single batch in training. Alternatives in the case that the sample is too big might be to consider a "gradient accumulation" approach.
        self.train_batch_size = train_batch_size
        if(self.train_batch_size is None):
            self.train_batch_size = self.n_train

        self.val_batch_size = val_batch_size
        if(self.val_batch_size is None):
            self.val_batch_size = self.n_val
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if(self.gradient_accumulation_steps is None):
            # The number of batches until the actual weights update (we ensure that the weights are updated only once per epoch, even though we might have multiple batches)
            self.gradient_accumulation_steps = int(np.ceil( self.n_train / self.train_batch_size ))

        self.compile_model(optimizer_independent_pars = optimizer_independent_pars, optimizer_nn = optimizer_nn, run_eagerly = run_eagerly)

        # Create the training dataset
        self.buffer_size = buffer_size
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, *self.data_train))
        train_dataset = train_dataset.shuffle(buffer_size = self.buffer_size).batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE)
        self.train_dataset = train_dataset
        
        val_dataset = None
        if(validation):
            # Create the validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, *self.data_val))
            val_dataset = val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)
            self.val_dataset = val_dataset

        self.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs = epochs,
            verbose = 0,
            callbacks = self.callbacks,
            batch_size = self.train_batch_size,
            shuffle = shuffle
        )

        # Obtain covariance estimates for the neural network induced parameters
        self.get_covariances()

    def get_covariances(self, jitter = 1.0e-6, max_retries = 5):
        """
            Supposing the weights from the last-layer are proper statistical parameters, together with the independent parameters,
            we can recover their hessian matrix, whose inverse corresponds to an approximation to the MLE estimator covariance matrix.

            The prior_weights variable correspond to the prior variance we assume for the weights in the neural network.
            It ensures the loss hessian will be invertible.
        """
        vars_to_differentiate = []

        num_independent_params = 0
        # Obtain covariance matrices for all independent estimators (independent on data x)
        if(self.independent_pars_use):
            for i in range( len(self.independent_pars) ):
                vars_to_differentiate.append( self.trainable_variables[i] )
        
            # Number of weights associated to independent parameters
            num_independent_params = sum([tf.size(v).numpy() for v in vars_to_differentiate])

        num_nn_params = 0
        # Obtain confidence intervals for all outputs from the network
        if(self.neural_network_use):
            vars_to_differentiate.append( self.layers[-1].trainable_variables[0] )
        
            # Number of weights associated to the neural network component
            num_nn_params = tf.size( self.layers[-1].trainable_variables[0] )

        # Total number of real weights we consider as statistical parameters
        num_params = num_independent_params + num_nn_params
        total_hessian = tf.zeros((num_params, num_params))
        
        for batch in self.train_dataset:
            x = batch[0]
            
            with tf.GradientTape(persistent = True) as tape2:
                with tf.GradientTape() as tape1:
                    nn_output = self(x, training = True)
                    loss_value = self.loglikelihood_loss(self, nn_output = nn_output, data = batch)
                
                # First Derivative
                grads = tape1.gradient(loss_value, vars_to_differentiate)

                # ----------------------------------------------------------------------------------------------------------------------------------
                # This routine is designed to identify singular hessian problems and which parameters they may correspond to before all calculations
                # ----------------------------------------------------------------------------------------------------------------------------------
                # List of parameters that are not used in the loss function. That results in a non-invertible hessian matrix
                lack_independent_pars = []
                lack_nn_pars = []
                halt_hessian = False
                # Check if any grad value is None
                # If there is a None grad, it means the loss function does not depend on that parameter, and therefore, can not obtain covariance matrix
                for i, grad in enumerate(grads):
                    if(grad is None):
                        # Halt the hessian calculations, given there is a problem
                        halt_hessian = True
                        var_name = vars_to_differentiate[i].path.split("/")[-1]
                        # If gradient refers to an independent parameter, recover which one
                        if( len(self.independent_pars) ):
                            # Include the variable name for the user to see
                            lack_independent_pars.append( self.independent_pars[ self.vars_to_index[var_name] ] )
                        # If gradient refers to the nn output and it is None, that means all nn parameters are not used in the loss function
                        else:
                            # All parameters lack in the loss function
                            lack_nn_pars = self.nn_pars
                    else:
                        # If grad is not None, but corresponds to a vector or matrix of weights, we must verify that all columns have at least a single nonzero value
                        # If we have an independent parameter and it is not None, we check if there is more than a single value
                        if( i < len(self.independent_pars) ):
                            # If we are dealing with a single independent parameter
                            if( len(grad.shape) == 0 ):
                                # If gradient is equal to zero, it is not considered in the log-likelihood at all.
                                # For it to be not None, it is possible that there are (theta / theta) or (theta - theta) somewhere
                                if(tf.math.abs(grad) == 1.0e-12 ):
                                    var_name = vars_to_differentiate[i].path.split("/")[-1]
                                    lack_independent_pars.append( self.independent_pars[ self.vars_to_index[var_name] ] )
                                    halt_hessian = True
                            # If we are dealing with a vector, independent parameter, check the same as above, but for all its values
                            if( len(grad.shape) > 0 ):
                                for j, g in enumerate(grad):
                                    if( tf.math.abs(g) == 0.0 ):                                   
                                        var_name = vars_to_differentiate[i].path.split("/")[-1]
                                        lack_independent_pars.append( "{}[{}]".format(self.independent_pars[ self.vars_to_index[var_name] ], j) )
                                        halt_hessian = True
                        # If we have a neural network weight and it is not None, check whether there is a null column on its gradient
                        else:
                            # Goes through all the columns in the weights matrix checking if at least one value is nonzero
                            for j in range(grad.shape[1]):
                                # If all values in the nn column weights are zero, there is a problem with that parameter
                                if( tf.reduce_all( tf.math.abs(grad[:,j]) == 0.0 ) ):
                                    var_name = self.nn_index_to_vars[j][4:] # Get the variable name, removing the "raw_" substring
                                    lack_nn_pars.append(var_name)
                                    halt_hessian = True
                                    
                # If any parameter is problematic in the loss function, the hessian will automatically be singular
                # Tells the user which parameters present problems in the log-likelihood
                # This detects trivial missidentification of parameters in the loss function
                if( halt_hessian ):
                    warnings.simplefilter("always", RuntimeWarning)
                    warnings.warn(
                        "Covariance matrix could not be computed because the loss function does not depend on:\n{}\n".format(lack_independent_pars + lack_nn_pars) + \
                        "Please, double check your loss function definition.",
                        category = RuntimeWarning
                    )
                    warnings.simplefilter("default", RuntimeWarning)
                    return
                # ----------------------------------------------------------------------------------------------------------------------------------
            
                # Flatten gradients to a single vector for easier Jacobian computation
                # Suppose we have k neurons on the last linear layer (no bias!) and d outputs. Then:
                # - The first group of k weights will correspond to the weights to the first output
                # - The second group of k weights will correspond to the weights to the second output
                grads_flat = tf.concat([tf.reshape(tf.transpose(g), [-1]) for g in grads], axis = 0)
        
            hessian_batch = tape2.jacobian(grads_flat, vars_to_differentiate)
        
            # Once the second derivatives for all weights have been obtained, check if there are None type derivates
            # A derivative will be returned as None by tensorflow if the derivative with respect to the parameter is zero everywhere
            # In our case, even though a parameter end up having zero correlation with the other ones, we would like to preserve the zeros
            for i in range(len(hessian_batch)):
                # If the independent parameter is a constant, the second derivative gradient will be a 1d vector
                # In that case, ensure this vector is a column so we can join all indepedent parameter derivatives into a single column
                if hessian_batch[i] is None:
                    hessian_batch[i] = tf.zeros( (num_params, tf.size(vars_to_differentiate[i])) )
                if( len(hessian_batch[i].shape) == 1 ):
                    hessian_batch[i] = hessian_batch[i][:,None]

            self.hessian_batch = hessian_batch
            
            # If there are both neural network parameters and independent ones
            if(self.neural_network_use and self.independent_pars_use):
                # Concatenate the second derivatives for all independent parameters into a single (num_params,num_independent_params) matrix
                hessian_batch_independent = tf.concat( hessian_batch[:-1], axis = 1 )
                # Reshape the jacobian for the neural network weights accordingly to transform it into a single (num_params,num_nn_params) matrix
                hessian_batch_nn = tf.reshape( tf.transpose( hessian_batch[-1], perm = [0,2,1] ), (num_params,num_nn_params) )
                # Merge the independent parameters and the neural network weights second derivatives, resulting in the final, hessian matrix for the model
                hessian_final_batch = tf.concat( [hessian_batch_independent, hessian_batch_nn], axis = 1 )
            # If there are only neural network parameters
            elif(self.neural_network_use):
                hessian_batch_nn = tf.reshape( tf.transpose( hessian_batch[-1], perm = [0,2,1] ), (num_params,num_nn_params) )
                hessian_final_batch = hessian_batch_nn
            # If all parameters are independent from input data x
            elif(self.independent_pars_use):
                hessian_batch_independent = tf.concat( hessian_batch, axis = 1 )
                hessian_final_batch = hessian_batch_independent
            else:
                warnings.simplefilter("always", RuntimeWarning)
                warnings.warn(
                    "Covariance matrix could not be computed because the model does not contain any trainble parameter.",
                    category = RuntimeWarning,
                )
                warnings.simplefilter("default", RuntimeWarning)
            
            # Manually delete tape2
            del tape2
            
            total_hessian += hessian_final_batch
            self.total_hessian = total_hessian

            for i in range(max_retries):
                try:
                    # Try to invert with current jitter
                    self.weights_covariance = tf.linalg.inv( self.total_hessian + jitter * tf.eye( num_params ) )
                    self.hessian_jitter = jitter
                    return
                except tf.errors.InvalidArgumentError:
                    # If damped matrix continues to be singular, try to increase jitter by a factor of 10
                    jitter *= 10
                    
            # If for all retries the hessian could not be inverted, return a warning that the covariance structure could not be obtained
            warnings.simplefilter("always", RuntimeWarning)
            warnings.warn(
                "Covariance matrix could not be computed because the log-likelihood Hessian is singular (or near singular).\n" + \
                "The model may not be identified..\n",
                category = RuntimeWarning,
            )
            warnings.simplefilter("default", RuntimeWarning)
                

    def apply_link(self, raw_pars):
        """
            Given a tensor of raw parameters, cycle through it, applying to each value its respective link function.
            Example:
            Let [[0.0, 1.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0, 2.0]]]
            be a list of 3 independent parameters and a neural network based parameter. The 2 rows represent two different inputs, x.
            Given a tensor of raw parameters, cycle through it, applying to each value its respective link function.
            Example:
            Let [[0.0, 1.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0, 2.0]]
            be a list of 3 independent parameters and a neural network based parameter. The 2 rows represent two different inputs, x.
            If the link functions are [identity, exp, logit, exp], respectively. Then, this function returns
            [[0.0, exp(1), 0.5, exp(1)],
             [0.0, exp(1), 0.5, exp(2)]]
        """
        link_evaluations = []
        # Independent parameters
        for i in range(raw_pars.shape[1]):
            if(i < self.independent_output_size):
                # Take the name of the parameter in this respective position
                var_name = self.independent_index_to_vars[i][4:].split("[")[0]
                link_evaluations.append( self.parameters[var_name]["link"]( raw_pars[:,i] )[:, None] )
            else:
                j = i - self.independent_output_size
                var_name = self.nn_index_to_vars[j][4:].split("[")[0]
                link_evaluations.append( self.parameters[var_name]["link"]( raw_pars[:,i] )[:, None] )
        pars = tf.concat(link_evaluations, axis = 1)
        return pars
        
        
    def covariance_output(self, x = None):
        """
            Given an input, x, obtain the asymptotic covariance matrices for the model weights estimators.
            If x is not given, return only the covariance matrix from the independent parameters, that are constant for every input.
        """
        # Number of independent parameter values as outputs (may be different from len(self.independent_pars), if vectors are considered)
        b = self.independent_output_size
        # Number of parameters as outputs to the neural network (may be different from len(self.nn_pars), if vectors are considered)
        d = self.nn_output_size

        # I_d \otimes Y^{(-2)} matrix for neural network weights
        H_tilde = None
        # I_b identity matrix for independent components covariance
        Ib = None
        
        if(self.neural_network_use):
            # If there are no independent parameters and also no input x was given, raise an Error
            if(not self.independent_pars_use and x is None):
                raise TypeError("Please, provide a list of input values, x.")
            elif(x is None):
                warnings.simplefilter("always", UserWarning)
                warnings.warn(
                    "Model supports both neural network modeled parameters and independent parameters.\n" + \
                    "As a list of input values, x, was not provided, obtaining the confidence intervals only for {}.".format(self.independent_pars),
                    category = UserWarning,
                )
                warnings.simplefilter("default", UserWarning)
            # If there are independent pars and x was given, simply obtain tilde{H} = I_d \otimes Y^{(-2)}
            else:
                x = tf.cast(x, dtype = tf.float32)
                # Let m be the number of entries in x
                # Y^{(-2)} dimension: (m, n_neurons_last_layer)
                Y_2 = self.neural_network_call_nolast(self, x)
        
                # Take the final layer weights and flatten then column-wise (each column stacked on top of the other) -> IMPORTANT! MUST MATCH HESSIAN CALCULATIONS!
                W = np.transpose( self.get_weights()[-1] ).flatten()
        
                m = x.shape[0] # Numer of inputs
                k = Y_2.shape[-1] # Number of neurons on the penultimate layer
                
                # For each entry, x_i, we need to obtain I_d \otimes Y^{(-2)}(x_i)
                # To do that, we must consider the Einstein summation formula, since np.kron always suppose 2d matrices
                # \tilde{H} = I_d \otimes Y^{(-2)}(x_i)
                # Therefore, H must have dimensions (m, d, kd) as it represents the transformation from the weights (normally distributed)
                # to the neural network output, considering multiplication with the penultimate layer, Y_2
                H_tilde = tf.einsum("ij, ...kl -> ...ijkl", tf.eye(d), Y_2[:,:,None]) # (m, d, k, d, 1) tensor
                H_tilde = tf.reshape(H_tilde, (m, d, k*d))

        if(self.independent_pars_use):
            Ib = tf.eye(b)
            if(self.neural_network_use and x is not None):
                Ib = tf.reshape(Ib, (1, b, b))
                Ib = tf.tile(Ib, [m, 1, 1])
        
        # Ib exists and H_tilde exists
        if(self.independent_pars_use and H_tilde is not None):
            Ib = tf.linalg.LinearOperatorFullMatrix(Ib)
            H_tilde = tf.linalg.LinearOperatorFullMatrix(H_tilde)
            H = tf.linalg.LinearOperatorBlockDiag([Ib, H_tilde]).to_dense()

            # Cycle through all independent parameters and flatten their values into a single vector of real values
            independent_pars = tf.concat([ tf.reshape(v, [-1]) for v in self.get_weights()[:len(self.independent_pars)] ], axis = 0)
            independent_pars = tf.reshape(independent_pars, (1, self.independent_output_size))
            independent_pars = tf.tile(independent_pars, [m, 1])
            # Obtain the raw expression for each parameter modeled as a nn output
            nn_pars = self.layers[-1](Y_2)

            # Concatenate all parameters into a single vector. It will be used to get the gradients to the link functions
            raw_pars = tf.concat([independent_pars, nn_pars], axis = 1)
            raw_cov = tf.einsum("...il, lj, ...ju -> ...iu", H, self.weights_covariance, tf.transpose(H, perm = [0,2,1]))
        # Ib exists and H_tilde do not
        elif(self.independent_pars_use and H_tilde is None):
            # Cycle through all independent parameters and flatten their values into a single vector of real values
            independent_pars = tf.concat([ tf.reshape(v, [-1]) for v in self.get_weights()[:len(self.independent_pars)] ], axis = 0)
            raw_pars = tf.reshape(independent_pars, (1, self.independent_output_size))
            raw_cov = self.weights_covariance[:self.independent_output_size, :self.independent_output_size]
        # Ib do not exist and H_tilde does (consequently, x was given)
        else:
            raw_pars = self.layers[-1](Y_2)
            raw_cov = tf.einsum("...il, lj, ...ju -> ...iu", H_tilde, self.weights_covariance, tf.transpose(H_tilde, perm = [0,2,1]))

        # Compute the Jacobian J for link functions over each individual
        with tf.GradientTape() as tape:
            tape.watch(raw_pars)
            theta_pars = self.apply_link( raw_pars )

        # (m, b+d, b+d)
        J = tape.batch_jacobian(theta_pars, raw_pars)
        
        # Obtain the covariance matrices for the transformed estimators according to the delta method
        theta_cov = tf.einsum("...il, ...lj, ...ju -> ...iu", J, raw_cov, tf.transpose(J, perm = [0,2,1]))
        
        return theta_cov

    def summary(self, x = None, alpha = 0.05):
        # Obtain the covariance matrices for all inputs, x
        theta_cov = self.covariance_output(x)
        pars_summary = {"index": np.arange(len(x))+1}
        z_norm = norm.ppf(1-alpha/2)
        
        for i in range(theta_cov.shape[1]):
            if(i < self.independent_output_size):
                # Take the name of the parameter in this respective position
                par_index_var = self.independent_index_to_vars[i][4:]
                nn_output = None
            else:
                j = i - self.independent_output_size
                par_index_var = self.nn_index_to_vars[j][4:]
                nn_output = self(x)
        
            par_index_var_split = par_index_var.split("[")
            par_name = par_index_var_split[0]
            # If name matches the index_to_vars result, parameter is a single number (not a vector)
            if(par_name == par_index_var):
                par_index = 0
            else:
                par_index = int( par_index_var_split[-1].split("]")[0] )

            if(nn_output is None):
                par_value = np.repeat( self.get_variable(par_name, nn_output)[par_index], theta_cov.shape[0] )
            else:
                par_value = self.get_variable(par_name, nn_output)[:,par_index]

            par_se = np.sqrt(theta_cov[:,i,i])
            par_lower = par_value - z_norm * np.sqrt(theta_cov[:,i,i])
            par_upper = par_value + z_norm * np.sqrt(theta_cov[:,i,i])
            
            pars_summary[par_index_var] = par_value
            pars_summary[par_index_var + "_se"] = par_se
            pars_summary[par_index_var + "_lower"] = par_lower
            pars_summary[par_index_var + "_upper"] = par_upper
        return pd.DataFrame(pars_summary)
    
    def plot_loglikelihood(self, par1, par2, par1_low, par1_high, par2_low, par2_high, n = 1000, colorscale = 'Inferno', local_maxima = True, neighborhood_range = 1.0):
        """
            Plot the profile log-likelihood for two chosen parameters from the model. If local_maxima is enabled, the surface is concentrated around the local maxima region,
            ignoring the log-likelihood in points that are further away from it. That may improve visualization when the likelihood value varies too much from a region to the other,
            which may end up blowing up the plot scale.
        """
        model_copy = self.copy()
        par1_values = tf.linspace(par1_low, par1_high, n)
        par2_values = tf.linspace(par2_low, par2_high, n)

        # Get the config object for par1 from the dictionary
        # and set the model_copy variables as their raw parameters
        par1_obj = self.parameters[par1]
        raw_par1 = "raw_" + par1
        raw_par1_values = par1_obj["link_inv"]( par1_values )
        par2_obj = self.parameters[par2]
        raw_par2 = "raw_" + par2
        raw_par2_values = par2_obj["link_inv"]( par2_values )

        # Both variables of interest gets replaced by tensors, with extra dimensions so the loss function return a results from broadcasting
        # When we call the model with training = False, every possible higher rank tensor gets remapped to have rank 4, so that this part of the code does not break
        model_copy.model_variables[raw_par1] = tf.Variable(
            tf.constant(raw_par1_values, dtype = tf.float32, shape = (1, 1, len(par1_values), 1)), trainable = False
        )
        model_copy.model_variables[raw_par2] = tf.Variable(
            tf.constant(raw_par2_values, dtype = tf.float32, shape = (1, 1, 1, len(par2_values))), trainable = False
        )

        nn_output = model_copy(self.x_train, training = False)
        x = tf.reshape(self.x_train, shape = (self.x_train.shape[0], self.x_train.shape[1], 1, 1))
        t_reshaped = tf.reshape(self.t_train, shape = (self.t_train.shape[0], 1, 1, 1))
        delta_reshaped = tf.reshape(self.delta_train, shape = (self.delta_train.shape[0], 1, 1, 1))
        
        # Obtain the log-likelihood values for different values of parameter 1 and 2
        # Since the final loss shape is given by (1, dim_par1, dim_par2):
        #     - (The first dim is reduced in reduce_main. The second one is temporary to a possible nn_output that is a vector)
        loss_values_par1_par2 = model_copy.loglikelihood_loss(model = model_copy, nn_output = nn_output, x = x, t = t_reshaped, delta = delta_reshaped)

        # If True, only plot the loh-likelihood function around the local maxima encountered, with a given neighborhood range
        if(local_maxima):
            par1_values_mesh, par2_values_mesh = np.meshgrid(par1_values, par2_values)

            # Obtain the distance between each point in the parametric subspace from the optimal point found by the gradient descent method
            distances_from_maxima = np.sqrt( ( np.transpose(par1_values_mesh) - self.get_variable(par1))**2 + (np.transpose(par2_values_mesh) - self.get_variable(par2))**2 )
            
            # Points that are too far away from the local maxima get removed from the plot by having the value np.nan
            loss_values_par1_par2 = np.where(distances_from_maxima <= neighborhood_range, loss_values_par1_par2, np.nan)
    
        fig = go.Figure(data=[go.Surface(x = par1_values, y = par2_values, z = -np.transpose( loss_values_par1_par2 ), colorscale = colorscale)])
        fig.update_layout(
            title = dict(text = r"Profile-Loglikelihood surface ({} x {})".format(par1, par2)),
            autosize = False,
            width = 500, height = 500,
            margin = dict(l = 65, r = 50, b = 65, t = 90)
        )

        self_nn_output = self(self.x_train, training = True)
        current_loglikelihood_loss = self.loglikelihood_loss(model = self, nn_output = self_nn_output, x = x, t = self.t_train, delta = self.delta_train)

        camera = dict(
            eye=dict(x=-1.5, y=-1.5, z=1.5),  # negative x and y rotates 180° in XY
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
        fig.update_layout(scene_camera = camera, scene = dict(
            xaxis = dict(
                tickangle=45,
                title = dict(
                    text = "{}".format(par1)
                )
            ),
            yaxis = dict(
                tickangle=-90,
                title = dict(
                    text = "{}".format(par2)
                )
            ),
            zaxis = dict(
                title = dict(
                    text = "Profile-Loglikelihood"
                )
            ),
        ))
        fig.add_trace(go.Scatter3d(
            x=[self.get_variable(par1)],
            y=[self.get_variable(par2)],
            z=[-current_loglikelihood_loss],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='circle'),
            text=['Maximum estimate'],
            textposition='top center'
        ))

        return fig

    def plot_loglikelihood_contour(self, par1, par2, par1_low, par1_high, par2_low, par2_high, n = 1000, colorscale = 'Inferno', local_maxima = True, neighborhood_range = 1.0, fig = None, ax = None):
        model_copy = self.copy()
        
        par1_values = tf.linspace(par1_low, par1_high, n)
        par2_values = tf.linspace(par2_low, par2_high, n)
        par1_values_mesh, par2_values_mesh = np.meshgrid( par1_values, par2_values )

        # Get the config object for par1 from the dictionary
        # and set the model_copy variables as their raw parameters
        par1_obj = self.parameters[par1]
        raw_par1 = "raw_" + par1
        raw_par1_values = par1_obj["link_inv"]( par1_values )
        par2_obj = self.parameters[par2]
        raw_par2 = "raw_" + par2
        raw_par2_values = par2_obj["link_inv"]( par2_values )

        # Both variables of interest gets replaced by tensors, with extra dimensions so the loss function return a results from broadcasting
        # When we call the model with training = False, every possible higher rank tensor gets remapped to have rank 4, so that this part of the code does not break
        model_copy.model_variables[raw_par1] = tf.Variable(
            tf.constant(raw_par1_values, dtype = tf.float32, shape = (1, 1, len(par1_values), 1)), trainable = False
        )
        model_copy.model_variables[raw_par2] = tf.Variable(
            tf.constant(raw_par2_values, dtype = tf.float32, shape = (1, 1, 1, len(par2_values))), trainable = False
        )

        nn_output = model_copy(self.x_train, training = False)
        x = tf.reshape(self.x_train, shape = (self.x_train.shape[0], self.x_train.shape[1], 1, 1))
        t_reshaped = tf.reshape(self.t_train, shape = (self.t_train.shape[0], 1, 1, 1))
        delta_reshaped = tf.reshape(self.delta_train, shape = (self.delta_train.shape[0], 1, 1, 1))
        
        # Obtain the log-likelihood values for different values of parameter 1 and 2
        # Since the final loss shape is given by (1, dim_par1, dim_par2):
        #     - (The first dim is reduced in reduce_main. The second one is temporary to a possible nn_output that is a vector)
        loss_values_par1_par2 = model_copy.loglikelihood_loss(model = model_copy, nn_output = nn_output, x = x, t = t_reshaped, delta = delta_reshaped)

        # If True, only plot the loh-likelihood function around the local maxima encountered, with a given neighborhood range
        if(local_maxima):
            par1_values_mesh, par2_values_mesh = np.meshgrid(par1_values, par2_values)

            # Obtain the distance between each point in the parametric subspace from the optimal point found by the gradient descent method
            distances_from_maxima = np.sqrt( ( np.transpose(par1_values_mesh) - self.get_variable(par1))**2 + (np.transpose(par2_values_mesh) - self.get_variable(par2))**2 )
            
            # Points that are too far away from the local maxima get removed from the plot by having the value np.nan
            loss_values_par1_par2 = np.where(distances_from_maxima <= neighborhood_range, loss_values_par1_par2, np.nan)

        if(fig is None or ax is None):
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,6))
        # mesh = ax.pcolormesh(par1_values_mesh, par2_values_mesh, -np.transpose( loss_values_par1_par2 ), cmap = "jet", vmin = vmin, vmax = vmax)
        mesh = ax.pcolormesh(par1_values_mesh, par2_values_mesh, -np.transpose( loss_values_par1_par2 ), cmap = "jet")
        ax.set_title("L({}, {})".format(par1, par2), fontsize = 20)
        ax.set_xlabel(par1, fontsize = 16)
        ax.set_ylabel(par2, fontsize = 16)
        fig.colorbar(mesh, ax = ax, orientation='vertical', fraction=0.046, pad=0.04)
        
    def plot_grid_3d(self, figs, nrows, ncols, figsize = (12,8), vspace=0.05, hspace=0.05):   
        specs = []
        for i in range(nrows):
            specs_row = []
            for i in range(ncols): 
                specs_row.append( {'type':'surface'} )
            specs.append( specs_row )
        
        fig = make_subplots(rows = nrows, cols = ncols, specs=specs)
    
        i, j = 1, 1
        # Add traces
        for k, f in enumerate(figs):
            cb_x = (i - 0.5) / ncols + 0.5 / ncols - 0.05  # shift to right of subplot
            cb_y = 1 - (j - 0.5) / nrows  # position vertically per row
            
            for trace in f.data:
                new_trace = copy.deepcopy(trace)
                # Ensure it's a surface trace before modifying colorbar
                if isinstance(new_trace, go.Surface):
                    new_trace.update(
                        showscale=True,
                        showlegend=False,
                        colorbar=dict(
                            x = cb_x + hspace / 2,  # shift colorbar horizontally
                            y = cb_y - vspace / 2,
                            len=(1 - (nrows - 1) * vspace) / nrows * 0.8,
                            title = ""  # you can customize
                        ),
                    )
                fig.add_trace(new_trace, row = j, col = i)
    
            i += 1
            if((i-1) % ncols == 0):
                i = 1
                j += 1
    
        # Copy the plots layouts into each cell
        for k, f in enumerate(figs, start=1):
            scene_name = f'scene{k}'
            fig.update_layout({
                scene_name: dict(
                    xaxis_title = f.layout.scene.xaxis.title.text if f.layout.scene.xaxis.title.text else '',
                    yaxis_title = f.layout.scene.yaxis.title.text if f.layout.scene.yaxis.title.text else '',
                    zaxis_title = f.layout.scene.zaxis.title.text if f.layout.scene.zaxis.title.text else '',
                    camera = f.layout.scene.camera if 'camera' in f.layout.scene else None
                )
        })
    
        # Adjust layout to remove empty space between subplots
        fig.update_layout(
            height = nrows*500,
            width = ncols*500,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        return fig
    
            