import os
import warnings

import random

import time
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf

from tensorflow import keras
import tensorflow_probability as tfp
from keras import optimizers, initializers

import logging

global_determinism = False

def set_global_determinism():
    # 1. Force TensorFlow to use deterministic C++ operations
    tf.config.experimental.enable_op_determinism()
    global global_determinism

    global_determinism = True

    # if(verbose):
    #     warnings.simplefilter("always", UserWarning)
    #     warnings.warn(
    #         "Enabling TensorFlow determinism for GPU training. All subsequent operations in the current Python session are locked into deterministic mode. To revert to standard high-performance execution, the Python environment (or Jupyter kernel) must be restarted.",
    #         category = UserWarning,
    #     )
    #     warnings.simplefilter("default", UserWarning)
    
def set_global_seed(seed = 42, verbose = False):
    # 2. Lock down all standard random number generators
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)    
    if(verbose):
        print("Global seed set to {}.".format(seed))

# Hides the retracing warnings
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from scipy.stats import norm

from tqdm import tqdm
from tqdm.keras import TqdmCallback

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config = config)

class EpochTracker(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs = None):
        self.model.current_epoch = epoch

    def on_epoch_end(self, epoch, logs = None):
        if(self.model.reduce_lr):
            learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            model_min_lr = float(tf.keras.backend.get_value(self.model.min_lr))
            if( tf.math.abs(learning_rate - model_min_lr) <= 1.0e-6 ):
                print("\nStopping: LR reached: {}".format(learning_rate))
                self.model.stop_training = True

    def on_train_batch_end(self, epoch, logs = None):
        """
            After updating the weights, obtain how much did the final parameters changed from their previous values:
                - Independent parameters are compared using the usual Euclidean distance
                - Neural network parameters are compared by taking the average Euclidean distance between all points in the sample
                - After the distances for all parameters are gathered, we take the whole, average norm
        """
        pass
        # # If the signal that the weights updated is False or missing, do nothing.
        # if not logs or not logs.get("weights_updated", False):
        #     return
        # # If the signal is True, then we can verify whether the model has converged here

        # tolerance = 1.0e-9
        
        # new_independent_predictions = []
        # for par in self.model.independent_pars:
        #     new_independent_predictions.append( self.model.get_variable(par, get_raw_value = True) )
        # new_independent_predictions = tf.concat(new_independent_predictions, axis = 0)

        # # For comparisons we will be using the raw value in order to avoid potential link functions exponential explosions
        # if(self.model.validation):
        #     nn_pars_predictions = self.model.predict(self.model.x_val, get_raw_value = True)
        # else:
        #     nn_pars_predictions = self.model.predict(self.model.x_train, get_raw_value = True)

        # nn_predictions = []
        # for par in nn_pars_predictions:
        #     nn_predictions.append( nn_pars_predictions[par] )

        # # Concatenate all neural network outputs into a single matrix
        # new_nn_predictions = tf.concat(nn_predictions, axis = -1)

        # if(self.model.independent_pars_use and self.model.neural_network_use):
        #     if(self.model.previous_nn_predictions is not None and self.model.previous_independent_predictions is not None):
        #         nn_distances = tf.reduce_max( ( (new_nn_predictions - self.model.previous_nn_predictions)/(tf.math.abs(self.model.previous_nn_predictions) + 1.0) )**2, axis = 0 )
        #         independent_distances = ( (new_independent_predictions - self.model.previous_independent_predictions)/(self.model.previous_independent_predictions + 1.0) )**2
        #         all_distances = tf.concat([independent_distances, nn_distances], axis = 0)
    
        #         distances_norm = tf.norm(all_distances, ord = 2)
        #         self.model.distances_norm.assign( distances_norm )
                
        #         if(distances_norm < tolerance):
        #             print("Stopping. Model has converged.")
        #             self.model.stop_training = True
        
        #     self.model.previous_nn_predictions = new_nn_predictions
        #     self.model.previous_independent_predictions = new_independent_predictions
        # else:
        #     pass
        

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

        dummy_tensor = tf.constant(0.0, dtype = tf.float32)
        self.device = dummy_tensor.device
        # Detects whether tf is running in a CPU or GPU device
        self.gpu_use = ( self.device.split(":")[-2].lower() == "gpu" )
    
        self.input_dim = input_dim
        self.seed = seed

        self.configured = False
        self.training = False
        self.pre_training = False
        self.current_epoch = tf.Variable(0, dtype = tf.int32, trainable = False, name = "current_epoch")
        
        self.total_hessian = None
        self.weights_covariance = None
        
        self.previous_nn_predictions = None
        self.previous_independent_predictions = None
        
        self.define_structure()

    def define_gradients(self):
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

        for parameter in self.parameters.keys():
            # Format all initial values to float32 and create init if not given
            if("init" in self.parameters[parameter] and self.parameters[parameter]["init"] is not None):
                self.parameters[parameter]["init"] = tf.cast(self.parameters[parameter]["init"], dtype = tf.float32)
            else:
                # Set the parameter initial value to be link(0)
                self.parameters[parameter]["init"] = self.parameters[parameter]["link"]( tf.constant(0.0, dtype = tf.float32) )

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

            # Name for the new, transformed parameter
            raw_parameter = "raw_" + parameter
            raw_init = par["link_inv"]( self.parameters[parameter]["init"] )
            
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
            raw_init = par["link_inv"]( self.parameters[parameter]["init"] )
            
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
        # If output dimension does not match this value it may be interesting to add an alert for that!
        
        # Once the entire structure has been defined, force the model to build all the weights properly
        if(self.neural_network_use):
            dummy_input = keras.Input(self.input_dim)
            self.training = True
            # Initialize all weights and trainable variables
            self(dummy_input)
            self.training = False
            
            # Take all trainable variables related to the neural network
            nn_last_layer_vars = self.layers[-1].trainable_variables
            # If nn_vars has more than a single weights matrix, that means the last layer admits a bias vector
            # We use that to format the weights properly in the hessian step for covariance calculations
            self.bias_use = False
            if( len(nn_last_layer_vars) > 1 ):
                self.bias_use = True

        # Now that the model is built and all the trainable variables instantiated, we define the gradient variables
        self.define_gradients()

    def loglikelihood_loss_pretrain(model, nn_output, data):
        pre_train_loss = 0.0
        for par in model.nn_pars:
            # We consider the parameter raw value to avoid explosions due to the link function
            # If link is exponential for example, the square of a distance of exponential quantities as a function of the weights
            # explodes and easily becomes unstable
            raw_par_init = tf.cast( model.parameters[par]["link_inv"]( model.parameters[par]["init"] ), dtype = tf.float32 )
            
            # Obtain the variable corresponding to the parameter
            raw_par_value = model.get_variable(par, nn_output, force_true = True, get_raw_value = True)
            # The pre-train is simply a quadratic loss over the initial raw values
            pre_train_loss += tf.reduce_sum( (raw_par_value - raw_par_init)**2 )
        
        return pre_train_loss
    
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
        x = self.neural_network_call(self, x_input, training = training)
        return x
        # if(training):
        #     return x

        # # If not on training mode, returns the neural network as a 4 indices tensor for plotting (better to change this in the future!)
        # return tf.reshape(x, (x.shape[0], x.shape[1], 1, 1))

        # If not training, dispose to the user all the neural network evaluations in an easy manner

    def predict(self, var_input, get_raw_value = False, training = False):
        # If x_input is a string, the user want a
        if isinstance(var_input, str):
            return self.get_variable(var_input, get_raw_value = get_raw_value).numpy()
        
        x_input = tf.cast(var_input, dtype = tf.float32)
        # If input is a vector, transform it into a column
        if(len(x_input.shape) == 1):
            x_input = tf.reshape( x_input, shape = (len(x_input), 1) )

        nn_output = self.neural_network_call(self, x_input, training = training)

        nn_output_parameters = {}
        for par in self.nn_pars:
            par_values = self.get_variable(par, nn_output, get_raw_value = get_raw_value)
            nn_output_parameters[par] = par_values
        return nn_output_parameters

    def get_variable(self, parameter, nn_output = None, get_raw_value = False, force_true = False, current_epoch = 0):
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

            # If user want to get the variable raw value, do not apply the link function
            if(get_raw_value):
                par_value = self.model_variables[raw_parameter]
            else:
                par_value = par["link"]( self.model_variables[raw_parameter] )

            # If user is tracking a function for the Delta method (variable_function_covariance), track the final variable for Auto diff
            if hasattr(self, '_delta_tape') and self._delta_tape is not None:
                try:
                    self._delta_tape.watch(par_value)
                    self._tracked_theta_tensors[parameter] = par_value
                except:
                    raise ValueError("tf.watch received a type Variable instead of tf.Tensor. Please, if you used lambda x : x as a link function, consider instead only tf.identity.")
                
            # return par_value
        else:
            # If nn_output is not None, assume the parameter came as a neural network output and return it from its positions in the output
            if(get_raw_value):
                par_value = tf.gather(nn_output, self.vars_to_index[raw_parameter], axis = 1)
            else:
                par_value = par["link"]( tf.gather(nn_output, self.vars_to_index[raw_parameter], axis = 1) )
    
            # If user is tracking a function for the Delta method (variable_function_covariance), track the final variable for Auto diff
            if hasattr(self, '_delta_tape') and self._delta_tape is not None:
                try:
                    self._delta_tape.watch(par_value)
                    self._tracked_theta_tensors[parameter] = par_value
                except:
                    raise(ValueError, "tf.watch received a type Variable instead of tf.Tensor. Please, use @tf.function functions only. For example, if you used lambda x : x as a link function, consider instead tf.identity.")

        par_has_warmup = "warmup_time" in par
        # If model is training and user specified a warmup_time for the parameter, return its constant, initial value
        # instead of the actual variable. That ensures the frozen variable will not be updated until a specific epoch
        if(not force_true and self.training and par_has_warmup and par["warmup_time"] > 0):

            # Force warmup_time to be a Tensor so the comparison happens in the graph
            warmup_tensor = tf.constant(par["warmup_time"], dtype = tf.int32)
            # if(get_raw_value):
            #     par_value = tf.cast( par["link_inv"]( par["init"] ), dtype = tf.float32 )
            # else:
            #     par_value = tf.cast( par["init"], dtype = tf.float32 )

            # # If the parameter corresponds to a neural network output, repeat its initial value the number of samples
            # if(nn_output is not None):
            #     par_value = tf.tile(np.atleast_2d(par_value), (nn_output.shape[0], par["shape"]))

            par_value = tf.cond(
                tf.math.less(self.current_epoch, warmup_tensor),
                lambda: tf.stop_gradient(par_value),
                lambda: par_value
            )
            
        return par_value

    @tf.function
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

            # If model is pre-training, consider the custom quadratic loss function for initial values
            if(self.pre_training):
                loss_value = self.loglikelihood_loss_pretrain(nn_output = nn_output, data = data)
            # If model is properly training, consider the user defined loss function
            else:
                loss_value = self.loglikelihood_loss(self, nn_output = nn_output, data = data)
            # loss_value = self.loglikelihood_loss(self, nn_output = nn_output, data = data)

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

        # Since self.n_acum_step is altered in apply_accumulated_gradients, we make a copy of it
        n_acum_step = tf.identity( self.n_acum_step )
        
        # If the necessary number of accumulation steps for update to occur happened, call the function to properly apply them to the weights
        tf.cond(tf.equal(n_acum_step, self.gradient_accumulation_steps), self.apply_accumulated_gradients, lambda: None)
        # If weights were just updated, return a True signal log for the Epochs callback to detect and verify if model converged
        weights_updated = tf.cond(tf.equal(n_acum_step, self.gradient_accumulation_steps), lambda : True, lambda: False)
        
        return_dict = {"likelihood_loss": loss_value, "weights_updated": weights_updated}
        
        if(self.reduce_lr):
            return_dict["learning_rate"] = self.optimizer_nn.learning_rate
            
        return return_dict

    def test_step(self, data):
        x = data[0]
        nn_output = self(x, training = True)
        likelihood_loss = self.loglikelihood_loss(model = self, nn_output = nn_output, data = data)
        
        if(self.reduce_lr):
            return {"likelihood_loss": likelihood_loss, "parameter_distances": self.parameter_distances, "learning_rate": self.optimizer_nn.learning_rate}
        
        return {"likelihood_loss": likelihood_loss}

    def apply_accumulated_gradients(self):
        # ----------------------------------- Independent parameters component -----------------------------------
        if(self.independent_pars_use):
            # Apply the accumulated gradients to the trainable variables
            self.optimizer_independent.apply_gradients( zip(self.gradient_accumulation_independent_pars, self.trainable_variables[ :len(self.independent_pars) ]) )
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
            

    def compile_model(self, optimizer_independent, optimizer_nn):
        """
            Defines the configuration for the model, such as batch size, training mode, early stopping.
        """
        # In the future, it might be interesting to allow the user to specify an optimizer for each single parameter in the model.
        # For now, they will specify one for the independent parameters and other for the neural network weights

        # optimizers.Adam(learning_rate = learning_rate, gradient_accumulation_steps = None),
        self.optimizer_independent = optimizer_independent
        self.optimizer_nn = optimizer_nn

    @tf.function(jit_compile = False, reduce_retracing = True)
    def _compiled_training_loop_optimized(self, x_full, data_full,
                                          epochs, batch_size,
                                          shuffle = True,
                                          metrics_update_freq = tf.constant(1, dtype = tf.int32),
                                          early_stopping = True,
                                          early_stopping_tolerance = tf.constant(1.0e-6, dtype = tf.float32),
                                          early_stopping_warmup = tf.constant(0, dtype = tf.int32),
                                          so_early_stopping_tolerance = tf.constant(1.0e-3, dtype = tf.float32),
                                          reduce_lr = True,
                                          reduce_lr_warmup = tf.constant(0, dtype = tf.int32),
                                          reduce_lr_factor = tf.constant(0.5, dtype = tf.float32),
                                          reduce_lr_min_delta = tf.constant(0.0, dtype = tf.float32),
                                          reduce_lr_patience = tf.constant(10, dtype = tf.int32),
                                          reduce_lr_cooldown = tf.constant(0, dtype = tf.int32),
                                          reduce_lr_min_lr = tf.constant(5e-4, dtype = tf.float32),
                                          deterministic = True,
                                          pre_training = False,
                                          verbose = True, print_freq = tf.constant(100, dtype = tf.int32)):
        """
            Executes the entire optimization loop purely in C++.
            Bypasses all Keras callbacks, progress bars, and Python overhead.
        """
        # Training variables
        final_epoch = tf.constant(0, dtype = tf.int32)
        final_loss = tf.constant(0.0, dtype = tf.float32)
        current_loss = tf.constant(0.0, dtype = tf.float32)
        stop_training = False
        lr_independent = self.optimizer_independent.learning_rate
        lr_nn = self.optimizer_nn.learning_rate
        minimal_lr_achieved = False
        
        distances_norm = tf.constant(float('inf'), dtype = tf.float32)

        # Set up history variables to track convergence profile
        num_checkups = epochs // metrics_update_freq
        history_index = 0
        loss_history = tf.TensorArray(tf.float32, size = epochs, clear_after_read = False)
        independent_distance_history = tf.TensorArray(tf.float32, size = num_checkups, element_shape=(self.independent_output_size, ), clear_after_read = False)
        nnmax_distance_history = tf.TensorArray(tf.float32, size = num_checkups, element_shape=(self.nn_output_size, ), clear_after_read = False)

        global global_determinism
        
        if(not global_determinism):
            # Grab the start time dynamically inside the C++ graph
            start_time = tf.timestamp()

        new_independent_predictions = []
        previous_independent_predictions = []
        if(self.independent_pars_use):
            previous_independent_predictions = [self.get_variable(par, get_raw_value = True) for par in self.independent_pars]
            # Concatenate all independent parameters into a single vector
            new_independent_predictions = tf.concat(previous_independent_predictions, axis = 0)
            previous_independent_predictions = new_independent_predictions

        new_nn_predictions = []
        previous_nn_predictions = []
        if(self.neural_network_use):
            nn_pars_predictions = self.predict(x_full, get_raw_value = True)
            nn_predictions = [nn_pars_predictions[par] for par in nn_pars_predictions]
            # Concatenate all neural network outputs into a single matrix
            new_nn_predictions = tf.concat(nn_predictions, axis = -1)
            previous_nn_predictions = new_nn_predictions

        previous_distances_norm = tf.constant(0.0, dtype = tf.float32)

        # Run an initial forward pass to get baseline nn predictions
        # initial_nn_output = self(x_full, training = False) if self.neural_network_use else None
        # prev_nn_preds = tf.concat([self.get_variable(par, initial_nn_output, get_raw_value=True) for par in self.nn_pars], axis=-1) if self.neural_network_use else tf.constant([[]], dtype=tf.float32)
        
        n_samples = tf.shape(data_full[0])[0]

        # ReduceLROnPlateau routine variables
        # if(reduce_lr):
        lr_wait = tf.constant(0, dtype = tf.int32)
        lr_cooldown_counter = tf.constant(0, dtype = tf.int32)
        best_metric = tf.constant(float('inf'), dtype = tf.float32)
        new_lr_ind = tf.constant(0.0, dtype = tf.float32)
        new_lr_nn = tf.constant(0.0, dtype = tf.float32)

        # Tracking distances along each epoch
        # distances_history = tf.TensorArray(dtype = tf.float32, size = epochs, dynamic_size = False)
        
        for epoch in tf.range(epochs):
            
            self.current_epoch.assign( tf.cast(epoch, tf.int32) )
            
            # ------------------------------------------------ Shuffle data at the start of each epoch, if desired ------------------------------------------------
            if(shuffle):
                indices = tf.random.shuffle( tf.range(n_samples) )
                # If we are dealing with a purely statistical model (no regression in any parameter) x_full may be None
                x_epoch = None
                if(x_full is not None):
                    x_epoch = tf.gather(x_full, indices)
                data_epoch = tuple([tf.gather(d, indices) for d in data_full])
            else:
                x_epoch = x_full
                data_epoch = data_full
            # -----------------------------------------------------------------------------------------------------------------------------------------------------

            batch_num = 0
            # for train_data in train_dataset:
            for start_idx in tf.range(0, n_samples, batch_size):
                batch_num += 1
                
                # Ensure the last batch doesn't go out of bounds
                end_idx = tf.minimum(start_idx + batch_size, n_samples)

                # Slice the batch out of RAM instantly
                x_batch = None
                if(x_full is not None):
                    x_batch = x_epoch[start_idx : end_idx]
                batch_data_tuple = tuple( [d[start_idx : end_idx] for d in data_epoch] )

                # Reconstruct full_data for the loss function
                batch_full_data = (x_batch,) + batch_data_tuple
                
                # 1. Forward Pass & Loss Computation
                with tf.GradientTape() as tape:
                    nn_output_batch = self(x_batch, training = True)
                    
                    if(pre_training):
                        loss_value = self.loglikelihood_loss_pretrain(nn_output = nn_output_batch, data = batch_full_data)
                    else:
                        loss_value = self.loglikelihood_loss(self, nn_output = nn_output_batch, data = batch_full_data)

                    loss_history = loss_history.write(epoch, loss_value)
                                      
                    # Automatic regularization from layer definitions
                    # Check if any layer in the model generated a regularization loss
                    if(self.losses):
                        # sums all tensors in the self.losses list
                        regularization_penalty = tf.math.add_n(self.losses)
                        # Add it to the base log-likelihood
                        loss_value = loss_value + regularization_penalty
                
                gradients = tape.gradient(loss_value, self.trainable_variables)
                # Clip gradients to avoid parameter explosions - TEST IT WITH MULTIPLE DATASETS!!!
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm = 1.0)

                # Gradient trap: Check if any gradient in the entire network became NaN or Inf
                has_nan_grad = tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None])
                has_inf_grad = tf.reduce_any([tf.reduce_any(tf.math.is_inf(g)) for g in gradients if g is not None])
        
                if tf.math.logical_or(has_nan_grad, has_inf_grad):
                    tf.print("\n[!] FATAL: Gradients exploded to NaN/Inf at Epoch:", epoch)

                    # lam_batch = self.get_variable("lam", nn_output_batch)
                    # rho_batch = self.get_variable("rho", nn_output_batch)
                    
                    # tf.print("-> LAST SAFE lam (min | max):", tf.reduce_min(lam_batch), "|", tf.reduce_max(lam_batch))
                    # tf.print("-> LAST SAFE rho (min | max):", tf.reduce_min(rho_batch), "|", tf.reduce_max(rho_batch))
                    
                    stop_training = True
                    break # Halt before the optimizer poisons the weights
                
                # To avoid crash problems in that case, we simply replace None with a zero like gradient, so those weights do not get updated
                # It is the user's responsibility to build a loss that depends on all the trainable parameters, but we allow that to happen in this case
                # for generality and to avoid unneccessary crashes when testing new models
                gradients = [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.trainable_variables)]

                # The first weights are always destined to the independent parameters
                # The neural network related weights come after those in the self.trainable_variables object
                independent_gradients = gradients[ :len(self.independent_pars) ]
                nn_gradients = gradients[ len(self.independent_pars): ]

                # ------------------------------------------------------------ Cumulate gradients ------------------------------------------------------------
                self.n_acum_step.assign_add(1)
                
                # Only cumulate independent gradients if iself.loglikelihood_loss_pretrain(nn_output = nn_output_batch, data = batch_full_data)n use
                if(self.independent_pars_use):
                    for i in range( len(self.gradient_accumulation_independent_pars) ):
                        self.gradient_accumulation_independent_pars[i].assign_add( independent_gradients[i] )
        
                # Only cumulate neural network gradients if in use
                if(self.neural_network_use):
                    for i in range( len(self.gradient_accumulation_nn) ):
                        self.gradient_accumulation_nn[i].assign_add( nn_gradients[i] )
                        
                if( tf.equal(self.n_acum_step, self.gradient_accumulation_steps) ):
                    if(self.independent_pars_use):
                        ind_grads = gradients[:len(self.independent_pars)]
                        self.optimizer_independent.apply_gradients( zip(ind_grads, self.trainable_variables[:len(self.independent_pars)]) )
                        # Resets all the cumulated gradients to zero
                        for i in range(len(self.gradient_accumulation_independent_pars)):
                            self.gradient_accumulation_independent_pars[i].assign( tf.zeros_like(self.trainable_variables[ :len(self.independent_pars) ][i], dtype = tf.float32) )
                        
                    if(self.neural_network_use):
                        nn_grads = gradients[len(self.independent_pars):]
                        self.optimizer_nn.apply_gradients(zip(nn_grads, self.trainable_variables[len(self.independent_pars):]))
                        # Resets all the cumulated gradients to zero
                        for i in range(len(self.gradient_accumulation_nn)):
                            self.gradient_accumulation_nn[i].assign(tf.zeros_like(self.trainable_variables[ len(self.independent_pars): ][i], dtype = tf.float32))
                    # Resets the cumulation counter
                    self.n_acum_step.assign(0)
                # --------------------------------------------------------------------------------------------------------------------------------------------                
            # --------------------------------------------------------------- Evaluate stop criteria ---------------------------------------------------------------
            # For comparisons we will be using the raw value in order to avoid potential link functions exponential explosions
            # Get the independent parameters predictions
                
            if(epoch % metrics_update_freq == 0 and epoch > 0):
                if(self.independent_pars_use):
                    new_independent_predictions = [self.get_variable(par, get_raw_value = True) for par in self.independent_pars]
                    # Concatenate all independent parameters into a single vector
                    new_independent_predictions = tf.concat(new_independent_predictions, axis = 0)
                
                if(self.neural_network_use):
                    # Pass training False so any dropout layers get evaluated as a whole
                    nn_pars_predictions = self.predict(x_full, get_raw_value = True, training = False)
                    nn_predictions = [nn_pars_predictions[par] for par in nn_pars_predictions]
                    # Concatenate all neural network outputs into a single matrix
                    new_nn_predictions = tf.concat(nn_predictions, axis = -1)
    
                # tf.print("new inde pred", new_independent_predictions)
                # tf.print("new nn pred", new_nn_predictions)

                if(self.independent_pars_use and self.neural_network_use):
                    independent_distances = ( (new_independent_predictions - previous_independent_predictions)/(tf.math.abs(previous_independent_predictions) + 1.0) )**2
                    independent_distance_history = independent_distance_history.write(history_index, independent_distances)
                    
                    nn_distances = tf.reduce_max( ( (new_nn_predictions - previous_nn_predictions) / (tf.math.abs(previous_nn_predictions) + 1.0) )**2, axis = 0 )

                    nnmax_distance_history = nnmax_distance_history.write(history_index, nn_distances)
                    
                    # Concatenate all distances into a single array and take its norm
                    all_distances = tf.concat([independent_distances, nn_distances], axis = 0)
                    distances_norm = tf.math.sqrt( tf.reduce_sum(all_distances) )
                    
                    if(early_stopping and epoch > early_stopping_warmup):
                        # If the distances norm by itself or the difference between the distance taken in the last checkup and the one takes here is small enough
                        # We consider the model has converged. Note that the distance of distances counts as a second order convergence
                        # If the model converged to a local minima, but the learning rate did not have the chance to be reduced to the right value
                        # we detect whether the oscilations are the same. If they are, probably the fluctuation is simply due to the high learning rate
                        if((distances_norm < early_stopping_tolerance) or (tf.math.abs(previous_distances_norm - distances_norm) < so_early_stopping_tolerance)):
                            if(verbose):
                                tf.print("\nStopping. Model has converged at epoch", epoch)
                            stop_training = True

                    previous_nn_predictions = new_nn_predictions
                    previous_independent_predictions = new_independent_predictions
                    

                elif(self.independent_pars_use):
                    independent_distances = ( (new_independent_predictions - previous_independent_predictions)/(tf.math.abs(previous_independent_predictions) + 1.0) )**2
                    independent_distance_history = independent_distance_history.write(history_index, independent_distances)
                    
                    distances_norm = tf.math.sqrt( tf.reduce_sum(independent_distances) )
                    
                    if(early_stopping and epoch > early_stopping_warmup):
                        if((distances_norm < early_stopping_tolerance) or (tf.math.abs(previous_distances_norm - distances_norm) < so_early_stopping_tolerance)):
                            if(verbose):
                                tf.print("\nStopping. Model has converged at epoch", epoch)
                            stop_training = True
                    
                    previous_independent_predictions = new_independent_predictions
                else:
                    nn_distances = tf.reduce_max( ( (new_nn_predictions - previous_nn_predictions) / (tf.math.abs(previous_nn_predictions) + 1.0) )**2, axis = 0 )
                    nnmax_distance_history = nnmax_distance_history.write(history_index, nn_distances)
                    
                    distances_norm = tf.math.sqrt( tf.reduce_sum(nn_distances) )
                    
                    if(early_stopping and epoch > early_stopping_warmup):
                        if( (distances_norm < early_stopping_tolerance) or (tf.math.abs(previous_distances_norm - distances_norm) < so_early_stopping_tolerance) ):
                            if(verbose):
                                tf.print("\nStopping. Model has converged at epoch", epoch)
                            stop_training = True
                    previous_nn_predictions = new_nn_predictions

                previous_distances_norm = distances_norm
                
                history_index += 1
                # distances_history = distances_history.write(epoch, distances_norm)

                # ------------------------------------ ReduceLROnPlateau custom mechanism. Hard-coded implementation needed for performance issues ------------------------------------
                if(reduce_lr and epoch > reduce_lr_warmup):
                    # Reconstruct full_data for the loss function
                    batch_full_data = (x_full,) + tuple(data_full)
                    
                    nn_output_batch = self(x_full, training = True)

                    if(self.pre_training):
                        current_loss = self.loglikelihood_loss_pretrain(nn_output = nn_output_batch, data = batch_full_data)
                    else:
                        current_loss = self.loglikelihood_loss(self, nn_output = nn_output_batch, data = batch_full_data)
                    
                    if(lr_cooldown_counter > 0):
                        lr_cooldown_counter = lr_cooldown_counter - 1
                        lr_wait = tf.constant(0, dtype = tf.int32)
                    else:
                        # Check if the loss improved by at least the min_delta
                        if(current_loss < (best_metric - reduce_lr_min_delta)):
                            best_metric = current_loss
                            lr_wait = tf.constant(0, dtype = tf.int32)
                        else:
                            lr_wait = lr_wait + 1
                            
                            if(lr_wait >= reduce_lr_patience):
                                # Decay the learning rates
                                if(self.independent_pars_use):
                                    old_lr_ind = self.optimizer_independent.learning_rate
                                    new_lr_ind = tf.maximum(old_lr_ind * reduce_lr_factor, reduce_lr_min_lr)
                                    self.optimizer_independent.learning_rate.assign(new_lr_ind)
                                    
                                if(self.neural_network_use):
                                    old_lr_nn = self.optimizer_nn.learning_rate
                                    new_lr_nn = tf.maximum(old_lr_nn * reduce_lr_factor, reduce_lr_min_lr)
                                    self.optimizer_nn.learning_rate.assign(new_lr_nn)
                                    
                                # Reset counters
                                lr_cooldown_counter = reduce_lr_cooldown
                                lr_wait = tf.constant(0, dtype = tf.int32)

                                if verbose:
                                    # Print a newline first so it doesn't break your progress bar
                                    # tf.print("\n[ReduceLROnPlateau] Epoch ", epoch, " - Loss plateaued. Reducing LR.")
                                    pass

                            # If minimum learning rate is reached, stop training
                            if( early_stopping and (tf.equal(new_lr_ind, reduce_lr_min_lr) or tf.equal(new_lr_nn, reduce_lr_min_lr)) and (not minimal_lr_achieved) ):
                                # tf.print("\nMinimal learning rate achieved on epoch", epoch, ".")
                                minimal_lr_achieved = True
                                stop_training = True
                # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
                # --------------------------------------------- Native progress tracker without great performance lose ---------------------------------------------
                if(verbose and epoch % print_freq == 0 and epoch > 0):
                    if(epoch > 0):
                        if(not global_determinism):
                            # Calculate dynamic speed
                            current_time = tf.timestamp()
                            elapsed_time = current_time - start_time
                            epochs_per_sec = tf.cast(epoch, tf.float64) / elapsed_time
                            
                            tf.print(
                                "\rOptimizing... Epoch: [", epoch, "/", epochs, "] ",
                                "| Loss: ", current_loss, 
                                "| Param Dist: ", distances_norm,
                                "| Independent Learning rate: ", self.optimizer_independent.learning_rate,
                                "| Network Learning rate: ", self.optimizer_nn.learning_rate,
                                "| Speed: ", tf.cast(epochs_per_sec, tf.int32), " it/s   ", # Cast to int for clean printing
                                end = ""
                            )
                        else:
                            tf.print(
                                "\rOptimizing... Epoch: [", epoch, "/", epochs, "] ",
                                "| Loss: ", current_loss, 
                                "| Param Dist: ", distances_norm,
                                "| Independent Learning rate: ", self.optimizer_independent.learning_rate,
                                "| Network Learning rate: ", self.optimizer_nn.learning_rate,
                                "| [Speed tracking disabled for determinism]   ",
                                end=""
                            )
                    else:
                        # Skip speed calculation on epoch 0 to avoid dividing by zero
                        tf.print(
                            "\rOptimizing... Epoch: [", epoch, "/", epochs, "] ",
                            "| Loss: ", current_loss, 
                            "| Param Dist: ", distances_norm,
                            "| Speed: Calculating...",
                            end = ""
                        )
                # --------------------------------------------------------------------------------------------------------------------------------------------------
                
            # Stop if converged or an error occurred
            if stop_training:
                break

        # For a tf.TensorArray we must stack its values before finally returning it as a Tensor
        # final_distances_tensor = distances_history.stack()

        # After training, restores the learning rates for both optimizers
        self.optimizer_independent.learning_rate.assign(lr_independent)
        self.optimizer_nn.learning_rate.assign(lr_nn)

        final_epoch = epoch

        loss_history = loss_history.stack()
        independent_distance_history = independent_distance_history.stack()
        nnmax_distance_history = nnmax_distance_history.stack()
        
        return final_loss, final_epoch, loss_history, independent_distance_history, nnmax_distance_history
    
    def train_model(self, x, data,
                    epochs, shuffle,
                    metrics_update_freq = 100,
                    fine_tune = True,
                    get_covariances = True,
                    validation = False, val_prop = None, x_val = None, data_val = None,
                    optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                    optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                    train_batch_size = None, val_batch_size = None,
                    buffer_size = 4096, gradient_accumulation_steps = None,
                    early_stopping = True, early_stopping_tolerance = 1.0e-6, early_stopping_warmup = 100,
                    so_early_stopping_tolerance = 1.0e-3,
                    reduce_lr = True, reduce_lr_warmup = 0, reduce_lr_factor = 0.5, reduce_lr_min_delta = 0.0, reduce_lr_patience = 10,
                    reduce_lr_cooldown = 0, reduce_lr_min_lr = 1e-5,
                    deterministic = True,
                    verbose = True, print_freq = 100, track_time = True):
        
        # Format the input data accordingly and prepare training and validation datasets
        self.config_training(x, data,
                             shuffle,
                             validation, val_prop, x_val, data_val,
                             optimizer_independent,
                             optimizer_nn,
                             train_batch_size, val_batch_size,
                             buffer_size, gradient_accumulation_steps,
                             verbose)
        
        # Force the optimizers to build their state variables in Python so they don't try to create them inside the C++ when function is called a second time
        # if self.independent_pars_use and not getattr(self.optimizer_independent, 'built', False):
        if(self.independent_pars_use):
            self.optimizer_independent.build( self.trainable_variables[:len(self.independent_pars)] )
        # if self.neural_network_use and not getattr(self.optimizer_nn, 'built', False):
        if(self.neural_network_use):
            self.optimizer_nn.build( self.trainable_variables[len(self.independent_pars):] )

        independent_learning_rate = tf.identity( optimizer_independent.learning_rate )
        nn_learning_rate = tf.identity( optimizer_nn.learning_rate )
        
        epochs = tf.constant(epochs, dtype = tf.int32)
        metrics_update_freq = tf.constant(metrics_update_freq, dtype = tf.int32)
        
        early_stopping_tolerance = tf.constant(early_stopping_tolerance, dtype = tf.float32)
        so_early_stopping_tolerance = tf.constant(so_early_stopping_tolerance, dtype = tf.float32)
        early_stopping_warmup = tf.constant(early_stopping_warmup, dtype = tf.int32)

        reduce_lr_warmup = tf.constant(reduce_lr_warmup, dtype = tf.int32)
        reduce_lr_factor = tf.constant(reduce_lr_factor, dtype = tf.float32)
        reduce_lr_min_delta = tf.constant(reduce_lr_min_delta, dtype = tf.float32)
        reduce_lr_patience = tf.constant(reduce_lr_patience, dtype = tf.int32)
        reduce_lr_cooldown = tf.constant(reduce_lr_cooldown, dtype = tf.int32)
        reduce_lr_min_lr = tf.constant(reduce_lr_min_lr, dtype = tf.float32)

        print_freq = tf.constant(print_freq, dtype = tf.int32)

        # If user need a deterministic outcome for reproducibility,
        # se all seeds to the global defined seed before training
        if(deterministic):
            # If GPU is being considered and user want deterministic behaviour, it is neccessary to activate
            # tf.config.experimental.enable_op_determinism()
            # This is unreversible for the Python session.  
            if(self.gpu_use):
                if(verbose):
                    print("GPU detected. Activating GPU determinism. To reverse this, the Python environment (or kernel) must be restated.")
                set_global_determinism()
            set_global_seed(seed = self.seed, verbose = verbose)
        else:
            set_global_seed(seed = None, verbose = verbose)

        if(verbose):
            print("Initializing training...")
        start_time = time.time()
        
        self.training = True
        # Compiled training routine
        final_loss, last_epoch, loss_history, independent_distance_history, nnmax_distance_history = self._compiled_training_loop_optimized(
            self.x_train,
            self.data_train,
            epochs,
            tf.constant(self.train_batch_size, dtype = tf.int32),
            shuffle = shuffle,
            metrics_update_freq = metrics_update_freq,
            early_stopping = early_stopping,
            early_stopping_tolerance = early_stopping_tolerance,
            early_stopping_warmup = early_stopping_warmup,
            so_early_stopping_tolerance = so_early_stopping_tolerance,
            reduce_lr = reduce_lr,
            reduce_lr_warmup = reduce_lr_warmup,
            reduce_lr_factor = reduce_lr_factor,
            reduce_lr_min_delta = reduce_lr_min_delta,
            reduce_lr_patience = reduce_lr_patience,
            reduce_lr_cooldown = reduce_lr_cooldown,
            reduce_lr_min_lr = reduce_lr_min_lr,
            deterministic = deterministic,
            pre_training = False,
            verbose = verbose,
            print_freq = print_freq
        )
        self.training = False

        self.last_epoch = last_epoch
        self.loss_history = loss_history
        self.independent_distance_history = independent_distance_history
        self.nnmax_distance_history = nnmax_distance_history
        
        # Resets the optimizers
        if(self.independent_pars_use):
            self.optimizer_independent.learning_rate = independent_learning_rate
            self.optimizer_independent.build( self.trainable_variables[:len(self.independent_pars)] )
        if(self.neural_network_use):
            self.optimizer_nn.learning_rate = nn_learning_rate
            self.optimizer_nn.build( self.trainable_variables[len(self.independent_pars):] )
    
        if(verbose):
            print("\nDone.")

        # If neural network is not being used, there is no need to fine-tune the model as it has already converged
        if(fine_tune and self.neural_network_use):
            if(verbose):
                print("Initializing model fine tuning (only independent parameters and last-layer)")
                # print(self.optimizer_independent.learning_rate)
                # print(self.optimizer_nn.learning_rate)
            # Format the input data accordingly and prepare training and validation datasets
            self.config_training(x, data,
                                 shuffle,
                                 validation, val_prop, x_val, data_val,
                                 optimizer_independent,
                                 optimizer_nn,
                                 train_batch_size, val_batch_size,
                                 buffer_size, gradient_accumulation_steps,
                                 verbose)
            
            # Set all but the last layers as non-trainable
            for i in range( len(self.layers)-1 ):
                self.layers[i].trainable = False
            # Redefine the gradients accumulation objects
            self.define_gradients()

            final_loss, last_epoch, loss_history, independent_distance_history, nnmax_distance_history = self._compiled_training_loop_optimized(
                self.x_train,
                self.data_train,
                epochs,
                tf.constant(self.train_batch_size, dtype = tf.int32),
                shuffle = shuffle,
                metrics_update_freq = metrics_update_freq,
                early_stopping = early_stopping,
                early_stopping_tolerance = early_stopping_tolerance,
                early_stopping_warmup = early_stopping_warmup,
                so_early_stopping_tolerance = so_early_stopping_tolerance,
                reduce_lr = reduce_lr,
                reduce_lr_warmup = reduce_lr_warmup,
                reduce_lr_factor = reduce_lr_factor,
                reduce_lr_min_delta = reduce_lr_min_delta,
                reduce_lr_patience = reduce_lr_patience,
                reduce_lr_cooldown = reduce_lr_cooldown,
                reduce_lr_min_lr = reduce_lr_min_lr,
                deterministic = deterministic,
                pre_training = False,
                verbose = verbose,
                print_freq = print_freq
            )

            self.last_epoch_finetune = last_epoch
            self.loss_history_finetune = loss_history
            self.independent_distance_history_finetune = independent_distance_history
            self.nnmax_distance_history_finetune = nnmax_distance_history
            
            self.optimizer_independent.learning_rate = independent_learning_rate
            self.optimizer_nn.learning_rate = nn_learning_rate
            
            if(verbose):
                print("\nDone.")
        
        if(get_covariances):
            if(verbose):
                print("Extracting covariance structure.")
            # Obtain covariance estimates for the neural network induced parameters
            self.get_covariances()
            if(verbose):
                print("Done.")

        execution_time = time.time() - start_time
        if(verbose and track_time):
            print("Optimization finished in {:.3f} seconds.".format(execution_time))

    def pre_train_model(self, x, data,
                        epochs, shuffle,
                        metrics_update_freq = 100,
                        validation = False, val_prop = None, x_val = None, data_val = None,
                        optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                        optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                        train_batch_size = None, val_batch_size = None,
                        buffer_size = 4096, gradient_accumulation_steps = None,
                        early_stopping = True, early_stopping_tolerance = 1.0e-6, early_stopping_warmup = 100,
                        so_early_stopping_tolerance = 1.0e-3,
                        reduce_lr = True, reduce_lr_warmup = 0, reduce_lr_factor = 0.5, reduce_lr_min_delta = 0.0, reduce_lr_patience = 10,
                        reduce_lr_cooldown = 0, reduce_lr_min_lr = 1e-5,
                        deterministic = True,
                        verbose = True, print_freq = 100, track_time = True):
        
        # Format the input data accordingly and prepare training and validation datasets
        self.config_training(x, data,
                             shuffle,
                             validation, val_prop, x_val, data_val,
                             optimizer_independent,
                             optimizer_nn,
                             train_batch_size, val_batch_size,
                             buffer_size, gradient_accumulation_steps,
                             verbose)

        # If the last layer admits a bias term, then given we initialize a parameter as a constant (instead of an actual function)
        # we simply set its last layer weights to zero while defining its intercept to match its intial value
        # Eventually, that will force the network to initially spit the initial value exactly
        if(self.bias_use):
            # self.v
            init_bias = np.zeros(self.nn_output_size)
            # For each output from the neural network, set the intercept value to match the initial value from user
            for i in range(self.nn_output_size):
                par_index_var_split = self.nn_index_to_vars[i][4:].split("[")
                var_name = par_index_var_split[0]
                
                if("init" in self.parameters[var_name] and self.parameters[var_name]["init"] is not None):
                    var_init = self.parameters[var_name]["init"]
                    # If parameter is single valued
                    if(self.parameters[var_name]["shape"] == 1):
                        # The raw output should match the initial valued applied to the inverse of the link function
                        init_bias[i] = self.parameters[var_name]["link_inv"]( var_init )
                    # If parameter is given as a vector on the model definition
                    else:
                        # Get which index from the parameter vector the ith output from the network is associated to
                        par_index_var = int( par_index_var_split[-1].split("]")[0] )
                        # If user only gave a single initial value for the whole vector
                        # consider all initial values to be the same
                        if( isinstance(var_init, (int, float)) or var_init.shape == () ):
                            init_bias[i] = self.parameters[var_name]["link_inv"]( var_init )
                        # If user gave an init vector
                        # set the weight according to that vector
                        else:
                            init_bias[i] = self.parameters[var_name]["link_inv"]( var_init[par_index_var] )
            self.layers[-1].trainable_variables[0].assign( tf.zeros_like(self.layers[-1].trainable_variables[0], dtype = tf.float32) )
            self.layers[-1].trainable_variables[-1].assign( init_bias )
        # If there is not an intercept term in the last layer of the network, the model must essentially
        # learn the constant function at the initial point by itself
        # To do that, we settle a custom loss function with quadratic error around the initial values (self.loglikelihood_loss_pretrain)
        # Using that loss function, the model tries to approximate the initial value, although its geometry may be hard to approximate it from its weights
        else:
            independent_learning_rate = optimizer_independent.learning_rate
            nn_learning_rate = optimizer_nn.learning_rate
            
            if(deterministic):
                # If GPU is being considered and user want deterministic behaviour, it is neccessary to activate
                # tf.config.experimental.enable_op_determinism()
                # This is unreversible for the Python session.  
                if(self.gpu_use):
                    if(verbose):
                        print("GPU detected. Activating GPU determinism. To reverse this, the Python environment (or kernel) must be restated.")
                    set_global_determinism()
                set_global_seed(seed = self.seed, verbose = verbose)
            else:
                set_global_seed(seed = None, verbose = verbose)

            # Force the optimizers to build their state variables in Python so they don't try to create them inside the C++ when function is called a second time
            # if self.independent_pars_use and not getattr(self.optimizer_independent, 'built', False):
            if(self.independent_pars_use):
                self.optimizer_independent.build( self.trainable_variables[:len(self.independent_pars)] )
            # if self.neural_network_use and not getattr(self.optimizer_nn, 'built', False):
            if(self.neural_network_use):
                self.optimizer_nn.build( self.trainable_variables[len(self.independent_pars):] )
            
            epochs = tf.constant(epochs, dtype = tf.int32)
            metrics_update_freq = tf.constant(metrics_update_freq, dtype = tf.int32)
            
            early_stopping_tolerance = tf.constant(early_stopping_tolerance, dtype = tf.float32)
            so_early_stopping_tolerance = tf.constant(so_early_stopping_tolerance, dtype = tf.float32)
            early_stopping_warmup = tf.constant(early_stopping_warmup, dtype = tf.int32)
    
            reduce_lr_warmup = tf.constant(reduce_lr_warmup, dtype = tf.int32)
            reduce_lr_factor = tf.constant(reduce_lr_factor, dtype = tf.float32)
            reduce_lr_min_delta = tf.constant(reduce_lr_min_delta, dtype = tf.float32)
            reduce_lr_patience = tf.constant(reduce_lr_patience, dtype = tf.int32)
            reduce_lr_cooldown = tf.constant(reduce_lr_cooldown, dtype = tf.int32)
            reduce_lr_min_lr = tf.constant(reduce_lr_min_lr, dtype = tf.float32)
    
            print_freq = tf.constant(print_freq, dtype = tf.int32)
            
            self.pre_training = True
            final_loss, last_epoch, loss_history, independent_distance_history, nnmax_distance_history = self._compiled_training_loop_optimized(
                self.x_train,
                self.data_train,
                epochs,
                tf.constant(self.train_batch_size, dtype = tf.int32),
                shuffle = shuffle,
                metrics_update_freq = metrics_update_freq,
                early_stopping = early_stopping,
                early_stopping_tolerance = early_stopping_tolerance,
                early_stopping_warmup = early_stopping_warmup,
                so_early_stopping_tolerance = so_early_stopping_tolerance,
                reduce_lr = reduce_lr,
                reduce_lr_warmup = reduce_lr_warmup,
                reduce_lr_factor = reduce_lr_factor,
                reduce_lr_min_delta = reduce_lr_min_delta,
                reduce_lr_patience = reduce_lr_patience,
                reduce_lr_cooldown = reduce_lr_cooldown,
                reduce_lr_min_lr = reduce_lr_min_lr,
                deterministic = deterministic,
                pre_training = True,
                verbose = verbose,
                print_freq = print_freq
            )
            self.pre_training = False

            self.last_epoch_pretrain = last_epoch
            self.loss_history_pretrain = loss_history
            self.independent_distance_history_pretrain = independent_distance_history
            self.nnmax_distance_history_pretrain = nnmax_distance_history

            self.optimizer_independent.learning_rate = independent_learning_rate
            self.optimizer_nn.learning_rate = nn_learning_rate
    
    def config_training(self, x, data,
                        shuffle = True,
                        validation = False, val_prop = None, x_val = None, data_val = None,
                        optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                        optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                        train_batch_size = None, val_batch_size = None,
                        buffer_size = 4096, gradient_accumulation_steps = None,
                        verbose = True):
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

        # Cast all variables from data to tf.float32 and pass them to tf.arrays if neccessarytrain_model
        for i in range(len(data)):
            data[i] = tf.cast(data[i], dtype = tf.float32)
            if(len(data[i].shape) == 1):
                data[i] = tf.reshape( data[i], shape = (len(data[i]), 1) )
        # Convert data to a tuple after reformatting it
        data = tuple(data)
        
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
            self.gradient_accumulation_steps = int(tf.math.ceil( self.n_train / self.train_batch_size ))

        self.compile_model(optimizer_independent = optimizer_independent, optimizer_nn = optimizer_nn)

        # Create the training dataset
        self.buffer_size = buffer_size
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, *self.data_train))

        # Shuffles the dataset on every call
        if(shuffle):
            train_dataset = train_dataset.cache().shuffle(buffer_size = self.buffer_size)
        train_dataset = train_dataset.batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE)
        self.train_dataset = train_dataset
        
        val_dataset = None
        if(validation):
            # Create the validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, *self.data_val))
            val_dataset = val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_dataset = val_dataset

        self.configured = True

    def compile_model_old(self, optimizer_independent, optimizer_nn, run_eagerly):
        """
            Defines the configuration for the model, such as batch size, training mode, early stopping.
        """
        # In the future, it might be interesting to allow the user to specify an optimizer for each single parameter in the model.
        # For now, they will specify one for the independent parameters and other for the neural network weights

        # optimizers.Adam(learning_rate = learning_rate, gradient_accumulation_steps = None),
        self.optimizer_independent = optimizer_independent
        self.optimizer_nn = optimizer_nn
        
        metrics = None
        if(self.reduce_lr and self.training):
            metrics = ["learning_rate"]

        # If neural network is used, then reduce_learning_rate only applied to optimizer_nn
        if(self.neural_network_use):
            optimizer = self.optimizer_nn
        # If no neural network is used, then reduce_learning_rate only applies to optimizer_independent
        else:
            optimizer = self.optimizer_independent

        self.compile(
            run_eagerly = run_eagerly,
            metrics = metrics,
            optimizer = optimizer
        )
        
    def config_training_old(self, x, data,
                            validation = False, val_prop = None, x_val = None, data_val = None,
                            optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                            optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                            train_batch_size = None, val_batch_size = None,
                            buffer_size = 4096, gradient_accumulation_steps = None,
                            early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10, early_stopping_warmup = 0,
                            reduce_lr = True, reduce_lr_factor = 0.5, reduce_lr_min_delta = 0.0, reduce_lr_patience = 10, reduce_lr_cooldown = 0,
                            reduce_lr_min_lr = 5e-4,
                            run_eagerly = False, verbose = 1):
        
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

        # Cast all variables from data to tf.float32 and pass them to tf.arrays if neccessarytrain_model
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

        if(self.validation):
            metric = "val_likelihood_loss"
        else:
            metric = "likelihood_loss"

        self.reduce_lr = reduce_lr
        if(self.reduce_lr):
            reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor = metric,
                factor = reduce_lr_factor,
                patience = reduce_lr_patience,
                min_delta = reduce_lr_min_delta,
                cooldown = reduce_lr_cooldown,
                min_lr = reduce_lr_min_lr
            )
            self.min_lr = reduce_lr_min_lr
            self.callbacks.append(reduce_lr_callback)

        self.early_stopping = False
        if(self.early_stopping):
            # Avoids overfitting and speeds training
            es = keras.callbacks.EarlyStopping(monitor = "distances_norm",
                                               mode = "min",
                                               start_from_epoch = early_stopping_warmup,
                                               min_delta = early_stopping_min_delta,
                                               patience = early_stopping_patience,
                                               restore_best_weights = True)
            self.callbacks.append(es)

        epoch_tracker = EpochTracker()
        self.callbacks.append(epoch_tracker)

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
            self.gradient_accumulation_steps = int(tf.math.ceil( self.n_train / self.train_batch_size ))

        self.compile_model(optimizer_independent = optimizer_independent, optimizer_nn = optimizer_nn, run_eagerly = run_eagerly)

        # Create the training dataset
        self.buffer_size = buffer_size
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, *self.data_train))
        train_dataset = train_dataset.cache().shuffle(buffer_size = self.buffer_size).batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE)
        self.train_dataset = train_dataset
        
        val_dataset = None
        if(validation):
            # Create the validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, *self.data_val))
            val_dataset = val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_dataset = val_dataset

        self.configured = True

    def pre_train_model_old(self, x, data,
                        epochs = 500, shuffle = True,
                        get_covariances = True,
                        validation = False, val_prop = None, x_val = None, data_val = None,
                        optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                        optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                        train_batch_size = None, val_batch_size = None,
                        buffer_size = 4096, gradient_accumulation_steps = None,
                        early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10, early_stopping_warmup = 0,
                        reduce_lr = True, reduce_lr_factor = 0.5, reduce_lr_min_delta = 0.0, reduce_lr_patience = 10, reduce_lr_cooldown = 0,
                        reduce_lr_min_lr = 5e-4,
                        run_eagerly = False, verbose = 1):

        # Format the input data accordingly and prepare training and validation datasets
        self.config_training_old(x, data,
                                 validation, val_prop, x_val, data_val,
                                 optimizer_independent,
                                 optimizer_nn,
                                 train_batch_size, val_batch_size,
                                 buffer_size, gradient_accumulation_steps,
                                 early_stopping, early_stopping_min_delta, early_stopping_patience, early_stopping_warmup,
                                 reduce_lr, reduce_lr_factor, reduce_lr_min_delta, reduce_lr_patience, reduce_lr_cooldown,
                                 reduce_lr_min_lr,
                                 run_eagerly, verbose)

        # If the last layer admits a bias term, then given we initialize a parameter as a constant (instead of an actual function)
        # we simply set its last layer weights to zero while defining its intercept to match its intial value
        # Eventually, that will force the network to initially spit the initial value exactly
        if(self.bias_use):
            # self.v
            init_bias = np.zeros(self.nn_output_size)
            # For each output from the neural network, set the intercept value to match the initial value from user
            for i in range(self.nn_output_size):
                par_index_var_split = self.nn_index_to_vars[i][4:].split("[")
                var_name = par_index_var_split[0]
                
                if("init" in self.parameters[var_name] and self.parameters[var_name]["init"] is not None):
                    var_init = self.parameters[var_name]["init"]
                    # If parameter is single valued
                    if(self.parameters[var_name]["shape"] == 1):
                        # The raw output should match the initial valued applied to the inverse of the link function
                        init_bias[i] = self.parameters[var_name]["link_inv"]( var_init )
                    # If parameter is given as a vector on the model definition
                    else:
                        # Get which index from the parameter vector the ith output from the network is associated to
                        par_index_var = int( par_index_var_split[-1].split("]")[0] )
                        # If user only gave a single initial value for the whole vector
                        # consider all initial values to be the same
                        if( isinstance(var_init, (int, float)) or var_init.shape == () ):
                            init_bias[i] = self.parameters[var_name]["link_inv"]( var_init )
                        # If user gave an init vector
                        # set the weight according to that vector
                        else:
                            init_bias[i] = self.parameters[var_name]["link_inv"]( var_init[par_index_var] )
            self.layers[-1].trainable_variables[0].assign( tf.zeros_like(self.layers[-1].trainable_variables[0], dtype = tf.float32) )
            self.layers[-1].trainable_variables[-1].assign( init_bias )
        # If there is not an intercept term in the last layer of the network, the model must essentially
        # learn the constant function at the initial point by itself
        # To do that, we settle a custom loss function with quadratic error around the initial values (self.loglikelihood_loss_pretrain)
        # Using that loss function, the model tries to approximate the initial value, although its geometry may be hard to approximate it from its weights
        else:
            self.pre_training = True
            self.fit(
                self.train_dataset,
                validation_data = self.val_dataset,
                epochs = epochs,
                verbose = 0,
                callbacks = self.callbacks,
                batch_size = self.train_batch_size,
                shuffle = shuffle
            )
            self.pre_training = False
    
    def train_model_old(self, x, data,
                        epochs, shuffle,
                        fine_tune = True,
                        get_covariances = True,
                        validation = False, val_prop = None, x_val = None, data_val = None,
                        optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                        optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                        train_batch_size = None, val_batch_size = None,
                        buffer_size = 4096, gradient_accumulation_steps = None,
                        early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10, early_stopping_warmup = 0,
                        reduce_lr = True, reduce_lr_factor = 0.5, reduce_lr_min_delta = 0.0, reduce_lr_patience = 10, reduce_lr_cooldown = 0,
                        reduce_lr_min_lr = 5e-4,
                        run_eagerly = False, verbose = 1):

        initial_independent_lr = optimizer_independent.learning_rate
        initial_nn_lr = optimizer_independent.learning_rate
        
        # Format the input data accordingly and prepare training and validation datasets
        self.config_training_old(x, data,
                             validation, val_prop, x_val, data_val,
                             optimizer_independent,
                             optimizer_nn,
                             train_batch_size, val_batch_size,
                             buffer_size, gradient_accumulation_steps,
                             early_stopping, early_stopping_min_delta, early_stopping_patience, early_stopping_warmup,
                             reduce_lr, reduce_lr_factor, reduce_lr_min_delta, reduce_lr_patience, reduce_lr_cooldown,
                             reduce_lr_min_lr,
                             run_eagerly, verbose)

        updates_per_epoch = 100

        print("Initializing model training")
        
        self.training = True
        self.fit(
            self.train_dataset,
            validation_data = self.val_dataset,
            epochs = epochs,
            verbose = 0,
            callbacks = self.callbacks,
            batch_size = self.train_batch_size,
            shuffle = shuffle
        )
        self.training = False

        print("Done.")

        # If neural network is not being used, there is no need to fine-tune the model as it has already converged
        if(fine_tune and self.neural_network_use):
            print("Initializing model fine tuning (only independent parameters and last-layer fit)")

            self.optimizer_independent.learning_rate = initial_independent_lr
            self.optimizer_nn.learning_rate = initial_nn_lr
            
            self.fine_tune_model(x, data,
                                 epochs, shuffle,
                                 get_covariances,
                                 validation, val_prop, x_val, data_val,
                                 optimizer_independent,
                                 optimizer_nn,
                                 train_batch_size, val_batch_size,
                                 buffer_size, gradient_accumulation_steps,
                                 early_stopping, early_stopping_min_delta, early_stopping_patience, early_stopping_warmup,
                                 reduce_lr, reduce_lr_factor, reduce_lr_min_delta, reduce_lr_patience, reduce_lr_cooldown,
                                 reduce_lr_min_lr,
                                 run_eagerly, verbose)
            print("Done.")
        
        if(get_covariances):
            print("Extracting covariance structure.")
            # Obtain covariance estimates for the neural network induced parameters
            self.get_covariances()

    def fine_tune_model_old(self, x, data,
                        epochs, shuffle,
                        get_covariances = True,
                        validation = False, val_prop = None, x_val = None, data_val = None,
                        optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                        optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                        train_batch_size = None, val_batch_size = None,
                        buffer_size = 4096, gradient_accumulation_steps = None,
                        early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10, early_stopping_warmup = 0,
                        reduce_lr = True, reduce_lr_factor = 0.5, reduce_lr_min_delta = 0.0, reduce_lr_patience = 10, reduce_lr_cooldown = 0,
                        reduce_lr_min_lr = 5e-4,
                        run_eagerly = False, verbose = 1):

        # Format the input data accordingly and prepare training and validation datasets
        self.config_training_old(x, data,
                             validation, val_prop, x_val, data_val,
                             optimizer_independent,
                             optimizer_nn,
                             train_batch_size, val_batch_size,
                             buffer_size, gradient_accumulation_steps,
                             early_stopping, early_stopping_min_delta, early_stopping_patience, early_stopping_warmup,
                             reduce_lr, reduce_lr_factor, reduce_lr_min_delta, reduce_lr_patience, reduce_lr_cooldown,
                             reduce_lr_min_lr,
                             run_eagerly, verbose)
        
        # Set all but the last layers as non-trainable
        for i in range( len(self.layers)-1 ):
            self.layers[i].trainable = False
        # Redefine the gradients accumulation objects
        self.define_gradients()

        self.fit(
            self.train_dataset,
            validation_data = self.val_dataset,
            epochs = epochs,
            verbose = 0,
            callbacks = self.callbacks,
            batch_size = self.train_batch_size,
            shuffle = shuffle
        )

    def get_covariances(self, jitter = 1.0e-6, max_retries = 5):
        """
            Supposing the weights from the last-layer are proper statistical parameters, together with the independent parameters,
            we can recover their hessian matrix, whose inverse corresponds to an approximation to the MLE estimator covariance matrix.

            The prior_weights variable correspond to the prior variance we assume for the weights in the neural network.
            It ensures the loss hessian will be invertible.
        """
        # Number of independent parameter values as outputs (may be different from len(self.independent_pars), if vectors are considered)
        b = self.independent_output_size
        # Number of parameters as outputs to the neural network (may be different from len(self.nn_pars), if vectors are considered)
        d = self.nn_output_size
        
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
            nn_vars = [ v for v in self.layers[-1].trainable_variables ]
            # Append the list of vars to differentiate with all weights on the last layer (linear predictor and bias weights)
            vars_to_differentiate += nn_vars
            # Number of weights associated to the neural network component
            num_nn_params = sum( tf.size(v) for v in nn_vars )
        
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
                        if( i < len(self.independent_pars) ):
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
                                if(tf.math.abs(grad) < 1.0e-12 ):
                                    var_name = vars_to_differentiate[i].path.split("/")[-1]
                                    lack_independent_pars.append( self.independent_pars[ self.vars_to_index[var_name] ] )
                                    halt_hessian = True
                            # If we are dealing with a vector, independent parameter, check the same as above, but for all its values
                            if( len(grad.shape) > 0 and grad.shape[0] > 1 ):
                                for j, g in enumerate(grad):
                                    if( tf.math.abs(g) == 0.0 ):                                   
                                        var_name = vars_to_differentiate[i].path.split("/")[-1]
                                        lack_independent_pars.append( "{}[{}]".format(self.independent_pars[ self.vars_to_index[var_name] ], j) )
                                        halt_hessian = True
                        # If we have a neural network weight and it is not None, check whether there is a null column on its gradient
                        else:
                            # Check if weights have columns (if dealing with the bias vector in the neural net part it is simply a vector)
                            if( len(grad.shape) > 1 ):
                                # Goes through all the columns in the weights matrix checking if at least one value is nonzero
                                for j in range( grad.shape[1]):
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
                # Suppose we have k neurons on the last linear layer and d outputs. Then:
                # - The first group of k weights will correspond to the weights to the first output
                # - The second group of k weights will correspond to the weights to the second output
                grads_flat = tf.concat([tf.reshape(tf.transpose(g), [-1]) for g in grads], axis = 0)
                
            hessian_batch = tape2.jacobian(grads_flat, vars_to_differentiate, experimental_use_pfor = False)
            
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

            if(self.neural_network_use):
                # If the neural network has a bias term, the independent parameters are the parameters up to the (:-2) index
                # [global, weights matrix, bias]
                if( self.bias_use ):
                    nn_start_index = -2
                # If not using bias, simply
                # [global, weights matrix]
                else:
                    nn_start_index = -1

            # If there are both neural network parameters and independent ones
            if(self.neural_network_use and self.independent_pars_use):
                # Concatenate the second derivatives for all independent parameters into a single (num_params,num_independent_params) matrix
                hessian_batch_independent = tf.concat( hessian_batch[:nn_start_index], axis = 1 )
                # Reshape the jacobian for the neural network weights accordingly to transform it into a single (num_params,num_nn_params-bias) matrix
                # in the tuple above, bias = 0 if not using a bias layer and bias = 1 otherwise
                # If self.bias_use = True: num_nn_params+nn_start_index*d+d = num_nn_params - 2d + d = num_nn_params-d
                # If self.bias_use = False: num_nn_params+nn_start_index*d+d = num_nn_params - d + d = num_nn_params
                # The dimensions match perfectly with the expected shape for the weights matrix
                hessian_batch_nn = tf.reshape( tf.transpose( hessian_batch[nn_start_index], perm = [0,2,1] ), (num_params,num_nn_params+nn_start_index*d+d) )
                # If there is a bias term, concatenate it to the hessian_batch_nn matrix before merging everything into a same hessian matrix
                if( self.bias_use ):
                    # We consider the bias terms right before the proper layer matrix weights
                    # That allows us to see this as the corresponding column terms to the vector Y^{(-2)} = [Y_0, Y_1, ..., Y_k]
                    hessian_batch_nn = tf.concat( [hessian_batch_nn, hessian_batch[-1]], axis = 1 )

                hessian_final_batch = hessian_batch_nn
                
                # Merge the independent parameters and the neural network weights second derivatives, resulting in the final, hessian matrix for the model
                hessian_final_batch = tf.concat( [hessian_batch_independent, hessian_batch_nn], axis = 1 )
            # If there are only neural network parameters
            elif(self.neural_network_use):
                hessian_batch_nn = tf.reshape( tf.transpose( hessian_batch[nn_start_index], perm = [0,2,1] ), (num_params,num_nn_params+nn_start_index*d+d) )
                if( self.bias_use ):
                    hessian_batch_nn = tf.concat( [hessian_batch_nn, hessian_batch[-1]], axis = 1 )
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
                    # Should I keep this tf.math.abs? Theoretically, the covariance should be positive definite. If it is not, maybe that should not be corrected (biased)
                    self.weights_covariance = tf.math.abs( tf.linalg.inv( self.total_hessian + jitter * tf.eye( num_params ) ) )
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
        if(x is not None):
            x = tf.cast(x, dtype = tf.float32)
            # If input is a vector, transform it into a column
            if(len(x.shape) == 1):
                x = tf.reshape( x, shape = (len(x), 1) )
        
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
                    "As a list of input values, x, was not provided, obtaining the covariances only for {}.".format(self.independent_pars),
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
        
                m = x.shape[0] # Number of inputs
                k = Y_2.shape[-1] # Number of neurons on the penultimate layer
                
                # For each entry, x_i, we need to obtain I_d \otimes Y^{(-2)}(x_i)
                # To do that, we must consider the Einstein summation formula, since np.kron always suppose 2d matrices
                # \tilde{H} = I_d \otimes Y^{(-2)}(x_i)
                # Therefore, H must have dimensions (m, d, kd) as it represents the transformation from the weights (normally distributed)
                # to the neural network output, considering multiplication with the penultimate layer, Y_2
                H_tilde = tf.einsum("ij, ...kl -> ...ijkl", tf.eye(d), Y_2[:,:,None]) # (m, d, k, d, 1) tensor
                H_tilde = tf.reshape(H_tilde, (m, d, k*d))
                
                # If there is a bias on the last layer, concatenate a I_d matrix to H_tilde
                if(self.bias_use):
                    # Create an (m,d,d) tensor with I_d in each m index
                    Id = tf.tile(tf.eye(d)[None,:,:], (m, 1, 1))
                    H_tilde = tf.concat([H_tilde, Id], axis = -1)

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

        # Delta method implementation for all parameters
        # (m, b+d, b+d)
        J = tape.batch_jacobian(theta_pars, raw_pars)
        
        # Obtain the covariance matrices for the transformed estimators according to the delta method
        theta_cov = tf.einsum("...il, ...lj, ...ju -> ...iu", J, raw_cov, tf.transpose(J, perm = [0,2,1]))
        
        return theta_cov

    def summary(self, x = None, alpha = 0.05):

        pars_summary = {"index": [1]}
        if(x is not None):
            x = tf.cast(x, dtype = tf.float32)
            # If input is a vector, transform it into a column
            if(len(x.shape) == 1):
                x = tf.reshape( x, shape = (len(x), 1) )
            pars_summary = {"index": np.arange(len(x))+1}
            
        # Obtain the covariance matrices for all inputs, x
        theta_cov = self.covariance_output(x)
        
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
                par_value = np.repeat( self.get_variable(par_name, nn_output, force_true = True)[par_index], theta_cov.shape[0] )
            else:
                par_value = self.get_variable(par_name, nn_output, force_true = True)[:,par_index]

            par_se = np.sqrt(theta_cov[:,i,i])
            par_lower = par_value - z_norm * par_se
            par_upper = par_value + z_norm * par_se
            
            pars_summary[par_index_var] = par_value
            pars_summary[par_index_var + "_se"] = par_se
            pars_summary[par_index_var + "_lower"] = par_lower
            pars_summary[par_index_var + "_upper"] = par_upper
            
        return pd.DataFrame(pars_summary)
    
    def variable_function_covariance(self, fun, x = None, data = None):
        """
            Receives a single dimensional function of independent and nn parameters and return its corresponding variance for all observations queried
        """
        nn_output = None
        if(x is not None):
            x = tf.cast(x, dtype = tf.float32)
            # If input is a vector, transform it into a column
            if(len(x.shape) == 1):
                x = tf.reshape( x, shape = (len(x), 1) )

            # Obtain the network raw output
            nn_output = self(x)
        
        # Obtain the covariance matrices for all inputs, x
        theta_cov = self.covariance_output(x)

        # Initialize gradient tracker for parameters
        self._delta_tape = tf.GradientTape(persistent=True)
        self._tracked_theta_tensors = {}

        data = [x] + data
        
        # Run the user's function with _delta_tape as context
        with self._delta_tape:
            f_theta = fun(self, nn_output, data)
            # Ensures the output from fun is atleast two dimensional
            if(len(f_theta.shape) == 1):
                f_theta = tf.expand_dims(f_theta, axis = -1)

        # print("f_theta shape", f_theta.shape)
        
        ordered_theta_var_names = []
        ordered_theta_tensors = []
        # Tracks which parameters from the model were used in fun and which were not
        theta_used = {}
        # Goes through all variables in the order they appear in the covariance matrix
        for i in range(self.independent_output_size + self.nn_output_size):
            if(i < self.independent_output_size):
                # Get only the name of the variable
                var_name = self.independent_index_to_vars[i][4:].split("[")[0]
            else:
                var_name = self.nn_index_to_vars[i-self.independent_output_size][4:].split("[")[0]

            # In case the variable has shape > 1, we ensure it gets added only once in this list
            if(var_name not in theta_used):
                # If variable was used in fun, add it on its correct order to the list
                if(var_name in self._tracked_theta_tensors):
                    ordered_theta_tensors.append( self._tracked_theta_tensors[var_name] )
                    theta_used[ var_name ] = True
                # If given variable was not used in fun, just include a None (the Jacobian will have a column full of zeros)
                else:
                    # ordered_theta_tensors.append( None )
                    theta_used[ var_name ] = False

        J_list = []
        
        used_counter = 0
        # Now that the Jacobians were obtained, we fix each one of them to match a proper matrix, J
        for i, parameter in enumerate(theta_used):
            # If parameter was used in fun
            if( theta_used[parameter] ):
                # If parameter is independent get the full jacobian, since it is the same for every observation
                if( parameter in self.independent_pars ):
                    parameter_jacobian = self._delta_tape.jacobian(f_theta, ordered_theta_tensors[ used_counter ], experimental_use_pfor = False)
                    # If first dimension of jacobian does not match data dimension and we know for sure there are input observations, x
                    if( (parameter_jacobian is not None) and (x is not None and self.neural_network_use) and (parameter_jacobian.shape[0] != x.shape[0]) ):
                        parameter_jacobian = tf.broadcast_to(parameter_jacobian, (x.shape[0], 1, 1))
                    
                # If parameter is output from the network, get the batch_jacobian instead
                elif( parameter in self.nn_pars ):
                    try:
                        parameter_jacobian = self._delta_tape.batch_jacobian(f_theta, ordered_theta_tensors[ used_counter ], experimental_use_pfor = False)
                    except ValueError:
                        parameter_jacobian = None
                # Increase the counter for the next used parameter in ordered_theta_tensors
                used_counter += 1
                # If get_variable was called, but parameter was not used, the jacobian still returns None
                if(parameter_jacobian is None):
                    if(x is not None):
                        jacobian_zeros = tf.zeros((x.shape[0], f_theta.shape[1], self.parameters[parameter]["shape"]))
                    else:
                        jacobian_zeros = tf.zeros((1, f_theta.shape[1], self.parameters[parameter]["shape"]))
                    parameter_jacobian = jacobian_zeros
                    
                J_list.append( parameter_jacobian )
            # If parameter was not used, we must impute its shape of zeros in the Jacobian matrix
            else:
                if(x is not None):
                    jacobian_zeros = tf.zeros((x.shape[0], f_theta.shape[1], self.parameters[parameter]["shape"]))
                else:
                    jacobian_zeros = tf.zeros((1, f_theta.shape[1], self.parameters[parameter]["shape"]))
                J_list.append( jacobian_zeros )

        
        # Concatenate all gradients into the jacobian matrix and virtually increase one dimension for following operation
        J = tf.concat(J_list, axis = -1)
        
        # 5. Clean up the state so it doesn't interfere with standard training
        self._delta_tape = None
        self._tracked_theta_tensors = None
        
        # Finally, with the Jacobian ordered and ready, the covariance matrix for function fun is J theta_cov J^T from the Delta method
        # This operation is simply expressed in terms of the Einstein summation convention given below
        fun_cov = tf.einsum("...il, ...lj, ...ju -> ...iu", J, theta_cov, tf.transpose(J, perm = [0,2,1]))
        
        return fun_cov
        

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
    
            
