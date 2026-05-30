import os
import warnings

import random

import time
import copy
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

import pickle

def set_global_determinism():
    # 1. Force TensorFlow to use deterministic C++ operations
    tf.config.experimental.enable_op_determinism()
    
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

        if(input_dim is None):
            raise ValueError("Please, provide an input dimension for the data.")
        self.input_dim = input_dim
        self.seed = seed
        # If seed was specified, fix the seed structure before initializing the model weights to ensure reproducibility
        if(self.seed is not None):
            if(self.gpu_use):
                set_global_determinism() 
            set_global_seed(seed = self.seed, verbose = False)
        
        self.configured = False
        self.training = False
        self.current_epoch = tf.Variable(0, dtype = tf.int32, trainable = False, name = "current_epoch")

        # Initialize raw data structures
        # If user passes tf.data.Datasets in training, these will continue being None
        # IF user passes raw Numpy arrays, these will track their sample sizes and handled data
        self.n_train, self.n_val = None, None
        self.x, self.data = None, None
        self.x_train, self.data_train = None, None
        self.x_val, self.data_val = None, None
        
        self.total_hessian = None
        self.weights_covariance = None
        
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

    def save_model(self, file_prefix):
        """
        Saves the trained model weights and all custom training metadata to disk.
        Generates two files: file_prefix.weights.h5 and file_prefix_meta.pkl
        """        
        if(not self.configured):
            warnings.warn("Model has not been configured/trained yet. Saving raw initialized weights.")

        # Save weights
        weights_path = f"{file_prefix}.weights.h5"
        self.save_weights(weights_path)

        # Collect metadata
        metadata = {
            "configured": self.configured,
            "training_completed": hasattr(self, 'loss_history'),
        }

        # If model has been trained, capture all the history metrics
        if hasattr(self, 'loss_history'):
            metrics_to_save = [
                "last_epoch", "loss_history", "val_loss_history", "convergence_reason",
                "nn_learning_rate_history", "best_metric_epoch", "best_metric",
                "last_epoch_finetune", "loss_history_finetune", "val_loss_history_finetune", "convergence_reason_finetune",
                "nn_learning_rate_history_finetune", "best_metric_epoch_finetune", "best_metric_finetune",
                "last_epoch_pretrain", "loss_history_pretrain", "val_loss_history_pretrain", "convergence_reason_pretrain",
                "best_metric_epoch_pretrain", "best_metric_pretrain",
                "nn_learning_rate_history_pretrain"
            ]
            
            for metric in metrics_to_save:
                if hasattr(self, metric):
                    metadata[metric] = getattr(self, metric)

        # Safely extract tf.Variable lists by converting them to Numpy arrays
        if hasattr(self, 'pre_finetuning_best_weights'):
            metadata["pre_finetuning_best_weights"] = [
                w.numpy() for w in self.pre_finetuning_best_weights
            ]

        if hasattr(self, 'best_weights'):
            metadata["best_weights"] = [
                w.numpy() for w in self.best_weights
            ]
                    
        # If covariances were calculated, save them too
        if self.total_hessian is not None:
            metadata["total_hessian"] = self.total_hessian.numpy() if hasattr(self.total_hessian, 'numpy') else self.total_hessian
            metadata["weights_covariance"] = self.weights_covariance.numpy() if hasattr(self.weights_covariance, 'numpy') else self.weights_covariance
            metadata["hessian_jitter"] = self.hessian_jitter

        # 5. Dump metadata
        meta_path = f"{file_prefix}_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
            
        print(f"Model successfully saved to {weights_path} and {meta_path}")

    def load_model(self, file_prefix):
        """
        Loads the model weights and training metadata from disk.
        The model must be instantiated with the exact same structure before calling this.
        """
        import pickle
        import os
        import tensorflow as tf
        
        weights_path = f"{file_prefix}.weights.h5"
        meta_path = f"{file_prefix}_meta.pkl"
        
        if not os.path.exists(weights_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Could not find {weights_path} or {meta_path}. Please check the file prefix.")

        # 1. Load weights
        self.load_weights(weights_path)

        # 2. Load metadata
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        # 3. Restore metadata attributes to the object
        for key, value in metadata.items():
            # Reconstruct tf.Variable tracking lists
            if key == "pre_finetuning_best_weights":
                self.pre_finetuning_best_weights = [tf.Variable(w, trainable=False) for w in value]
            elif key == "best_weights":
                self.best_weights = [tf.Variable(w, trainable=False) for w in value]
            # Restore matrix tensors
            elif key in ["total_hessian", "weights_covariance"]:
                setattr(self, key, tf.constant(value, dtype=tf.float32))
            # Restore standard metrics and configurations
            else:
                setattr(self, key, value)
                
        print(f"Model successfully loaded from {file_prefix}.")
    
    def call(self, x_input, training = True):
        if(self.neural_network_call is None):
            return None
        x = self.neural_network_call(self, x_input, training = training)
        return x

    @tf.function(reduce_retracing=True)
    def _compiled_predict_dataset(self, dataset):
        """
        Executes the forward pass over an entire dataset purely in C++.
        Uses tf.TensorArray to dynamically accumulate batches without memory explosions.
        """
        # Initialize a dynamic array to hold the batch outputs
        predictions_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        
        batch_index = tf.constant(0, dtype=tf.int32)
        
        for batch_data in dataset:
            # Safely extract X depending on whether the dataset yields (X, y) or just X
            # (e.g., your UTKFaces test_ds likely yields (filenames, ages))
            if isinstance(batch_data, tuple):
                x_batch = batch_data[0]
            else:
                x_batch = batch_data
                
            # Run the forward pass strictly in inference mode to keep BatchNorm/Dropout frozen
            nn_output_batch = self(x_batch, training=False)
            
            # Write the batch output into the array
            predictions_array = predictions_array.write(batch_index, nn_output_batch)
            batch_index += 1
            
        # Concat merges all batches along the 0th axis into a single continuous tensor!
        return predictions_array.concat()
        
    # def predict(self, var_input, get_raw_value = False, training = False):
    #     # If x_input is a string, the user want a
    #     if isinstance(var_input, str):
    #         return self.get_variable(var_input, get_raw_value = get_raw_value).numpy()
        
    #     x_input = tf.cast(var_input, dtype = tf.float32)
    #     # If input is a vector, transform it into a column
    #     if(len(x_input.shape) == 1):
    #         x_input = tf.reshape( x_input, shape = (len(x_input), 1) )

    #     nn_output = self.neural_network_call(self, x_input, training = training)

    #     nn_output_parameters = {}
    #     for par in self.nn_pars:
    #         par_values = self.get_variable(par, nn_output, get_raw_value = get_raw_value)
    #         nn_output_parameters[par] = par_values
    #     return nn_output_parameters

    def predict(self, var_input, get_raw_value=False, training=False):
        # Handle Independent Parameters
        if isinstance(var_input, str):
            return self.get_variable(var_input, get_raw_value=get_raw_value).numpy()
            
        # Handle tf.data.Dataset (C++ Batches)
        if isinstance(var_input, tf.data.Dataset):
            # Fire the compiled loop to compute everything at maximum GPU speed
            nn_output = self._compiled_predict_dataset(var_input)
            
        # Handle Raw Numpy Arrays / Tensors
        else:
            x_input = tf.cast(var_input, dtype=tf.float32)
            # If input is a vector, transform it into a column
            if len(x_input.shape) == 1:
                x_input = tf.reshape(x_input, shape=(len(x_input), 1))
                
            # Compute a single massive forward pass
            nn_output = self.neural_network_call(self, x_input, training=training)

        # Map the outputs to the dictionary
        nn_output_parameters = {}
        if self.neural_network_use:
            for par in self.nn_pars:
                # Slices the massive concatenated tensor perfectly into your defined parameters
                par_values = self.get_variable(par, nn_output, get_raw_value=get_raw_value)
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
            
    def compile_model(self, optimizer_independent, optimizer_nn):
        """
            Defines the configuration for the model, such as batch size, training mode, early stopping.
        """
        self.optimizer_independent = optimizer_independent
        self.optimizer_nn = optimizer_nn

    @tf.function(reduce_retracing=True)
    def evaluate_dataset_loss(self, dataset):
        """
        Evaluates the total log-likelihood loss iteratively over a tf.data.Dataset.
        Runs purely in C++ to prevent memory explosions on massive datasets.
        """
        total_loss = tf.constant(0.0, dtype=tf.float32)

        # Loop through the dataset natively in the C++ graph
        for batch_full_data in dataset:
            
            # 1. Unpack the batch
            if self.neural_network_use:
                x_batch = batch_full_data[0]
                batch_data_tuple = batch_full_data[1:]
            else:
                x_batch = None
                batch_data_tuple = batch_full_data

            # 2. Dimension guard to prevent broadcasting errors
            batch_data_tuple = tuple(
                [tf.expand_dims(d, axis=-1) if len(d.shape) == 1 else d for d in batch_data_tuple]
            )
            batch_full_data_reconstructed = (x_batch,) + batch_data_tuple
            
            # 3. Forward pass strictly in inference mode (training=False)
            if self.neural_network_use:
                nn_output_batch = self(x_batch, training=False)
            else:
                nn_output_batch = None

            # 4. Calculate loss for the current batch
            batch_loss = self.loglikelihood_loss(self, nn_output=nn_output_batch, data=batch_full_data_reconstructed)
            
            # 5. Add regularization penalties if any exist
            if self.losses:
                regularization_penalty = tf.math.add_n(self.losses)
                # Scale the penalty down by the dataset cardinality so it doesn't artificially inflate
                # Note: We divide by dataset cardinality here if penalty is per-sample, 
                # but standard Keras adds it directly. Adjust scaling if necessary for your formulation.
                batch_loss += regularization_penalty 

            # 6. Accumulate the loss
            total_loss += batch_loss

        return total_loss
    
    @tf.function(jit_compile = False, reduce_retracing = True)
    def _compiled_training_loop_dataset(self, train_dataset, epochs,
                                        shuffle = True,
                                        validation = False, val_dataset = None, force_training_validation = False,
                                        early_stopping = True,
                                        early_stopping_patience = tf.constant(10, dtype = tf.int32),
                                        early_stopping_warmup = tf.constant(0, dtype = tf.int32),
                                        reduce_lr = True,
                                        reduce_lr_warmup = tf.constant(0, dtype = tf.int32),
                                        reduce_lr_factor = tf.constant(0.5, dtype = tf.float32),
                                        reduce_lr_min_delta = tf.constant(0.0, dtype = tf.float32),
                                        reduce_lr_patience = tf.constant(5, dtype = tf.int32),
                                        reduce_lr_cooldown = tf.constant(20, dtype = tf.int32),
                                        reduce_lr_min_lr = tf.constant(5e-4, dtype = tf.float32),
                                        deterministic = True,
                                        pre_training = False,
                                        fine_tuning = False,
                                        verbose = True, print_freq = tf.constant(100, dtype = tf.int32)):
        """
            Executes the entire optimization loop purely in C++.
            Bypasses all Keras callbacks, progress bars, and Python overhead.
        """
        # Training variables
        final_epoch = tf.constant(0, dtype = tf.int32)
        current_loss = tf.constant(0.0, dtype = tf.float32)
        stop_training = False
        
        lr_independent = self.optimizer_independent.learning_rate
        lr_nn = self.optimizer_nn.learning_rate

        # Set up history variables to track convergence profile
        loss_history = tf.TensorArray(tf.float32, size = epochs, clear_after_read = False)
        val_loss_history = tf.TensorArray(tf.float32, size = epochs, clear_after_read = False)
        nn_learning_rate_history = tf.TensorArray(tf.float32, size = epochs, clear_after_read = False)
        
        start_time = tf.cast(tf.py_function(func=lambda: time.time(), inp=[], Tout=tf.float64), tf.float64)

        # ReduceLROnPlateau routine variables        
        # Reduce learning rate wait
        lr_wait = tf.constant(0, dtype = tf.int32)
        # Early stopping wait
        es_wait = tf.constant(0, dtype = tf.int32)

        current_loss_train = np.nan
        current_loss_val = 0.0
        
        lr_cooldown_counter = tf.constant(0, dtype = tf.int32)
        best_metric = tf.constant(float('inf'), dtype = tf.float32)
        best_metric_epoch = tf.constant(0, dtype = tf.int32)
        new_lr_ind = tf.constant(0.0, dtype = tf.float32)
        new_lr_nn = tf.constant(0.0, dtype = tf.float32)

        # Save the initial weights as the best
        # best_weights = [tf.identity(w) for w in self.trainable_variables]
        # self.best_weights = [tf.Variable(w, trainable=False) for w in self.trainable_variables]

        # By default, if convergence_reason does not change during training, that's because the algorithm reached the final
        # epoch without ever converging
        convergence_reason = "all_epochs"
        
        for epoch in tf.range(epochs):
            # At the start of each epoch, assign the current epoch to a global variable
            self.current_epoch.assign( tf.cast(epoch, tf.int32) )
            for batch_full_data in train_dataset:
                if(self.neural_network_use):
                    x_batch = batch_full_data[0]
                    batch_data_tuple = batch_full_data[1:]
                else:
                    x_batch = None
                    batch_data_tuple = batch_full_data

                # Ensure all target variables are at least 2D column vectors to prevent broadcasting errors
                batch_data_tuple = tuple(
                    [tf.expand_dims(d, axis=-1) if len(d.shape) == 1 else d for d in batch_data_tuple]
                )
                
                # Reconstruct full_data for the loss function
                batch_full_data_reconstructed = (x_batch,) + batch_data_tuple
                
                # 1. Forward Pass & Loss Computation
                with tf.GradientTape() as tape:
                    # If model finetuning, treat the neural network as a deterministic black box
                    # If there are dropout layers, the output will be random and break training
                    if(fine_tuning):
                        nn_output_batch = self(x_batch, training = False)
                    else:
                        nn_output_batch = self(x_batch, training = True)
                    
                    if(pre_training):
                        loss_value = self.loglikelihood_loss_pretrain(nn_output = nn_output_batch, data = batch_full_data_reconstructed)
                    else:
                        loss_value = self.loglikelihood_loss(self, nn_output = nn_output_batch, data = batch_full_data_reconstructed)
                                      
                    # Automatic regularization from layer definitions. Check if any layer in the model generated a regularization loss
                    if(self.losses):
                        # sums all tensors in the self.losses list
                        regularization_penalty = tf.math.add_n( self.losses )

                        current_batch_size = tf.cast(tf.shape(batch_data_tuple[0])[0], tf.float32)
                        batch_fraction = current_batch_size / tf.cast(self.n_train, tf.float32)
                        
                        # Add it to the base log-likelihood
                        loss_value = loss_value + regularization_penalty * batch_fraction
                        
                gradients = tape.gradient(loss_value, self.trainable_variables)
    
                # Gradient trap: Check if any gradient in the entire network became NaN or Inf
                has_nan_grad = tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None])
                has_inf_grad = tf.reduce_any([tf.reduce_any(tf.math.is_inf(g)) for g in gradients if g is not None])
                if tf.math.logical_or(has_nan_grad, has_inf_grad):
                    tf.print("\n[!] FATAL: Gradients exploded to NaN/Inf at Epoch:", epoch)
                    convergence_reason = "nan_gradients"
                    stop_training = True
                    break
                
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
                
                if(self.independent_pars_use):
                    for i in range( len(self.gradient_accumulation_independent_pars) ):
                        self.gradient_accumulation_independent_pars[i].assign_add( independent_gradients[i] )
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

            
            nn_learning_rate_history = nn_learning_rate_history.write(epoch, self.optimizer_nn.learning_rate)
            # --------------------------------------------------------------- Evaluate stop criteria ---------------------------------------------------------------
            # For comparisons we will be using the raw value in order to avoid potential link functions exponential explosions
            # ------------------------------------ ReduceLROnPlateau custom mechanism. Hard-coded implementation needed for performance issues ------------------------------------

            # The training loss is exactly the accumulated loss from the epoch's batch loop! - I GUESS THIS IS WRONG!!
            current_loss_train = tf.constant(0.0, dtype = tf.float32)
            # Iterate through the training dataset natively
            for batch_full_data_train in train_dataset:
                if(self.neural_network_use):
                    x_batch_train = batch_full_data_train[0]
                    batch_data_tuple_train = batch_full_data_train[1:] 
                else:
                    x_batch_train = None
                    batch_data_tuple_train = batch_full_data_train 
                
                batch_full_data_reconstructed_train = (x_batch_train,) + batch_data_tuple_train
                
                # Training data, but considering training = False strictly for metrics updates
                nn_train_batch = self(x_batch_train, training = False)
                
                if(pre_training):
                    train_batch_loss = self.loglikelihood_loss_pretrain(nn_output = nn_train_batch, data = batch_full_data_reconstructed_train)
                else:
                    train_batch_loss = self.loglikelihood_loss(self, nn_output = nn_train_batch, data = batch_full_data_reconstructed_train)
                
                if(self.losses):
                    regularization_penalty_train = tf.math.add_n(self.losses)
                    current_batch_size_train = tf.cast(tf.shape(batch_data_tuple_train[0])[0], tf.float32)
                    batch_fraction_train = current_batch_size_train / tf.cast(self.n_train, tf.float32)
                    train_batch_loss = train_batch_loss + regularization_penalty_train * batch_fraction_train
                    
                current_loss_train += train_batch_loss

            loss_history = loss_history.write(epoch, current_loss_train)
            
            # Compute the validation loss ONLY if requested
            if(validation):
                current_loss_val = tf.constant(0.0, dtype = tf.float32)
                # Iterate through the validation dataset natively
                for batch_full_data_val in val_dataset:
                    # Dynamic unpack for validation
                    if(self.neural_network_use):
                        x_batch_val = batch_full_data_val[0]
                        batch_data_tuple_val = batch_full_data_val[1:] 
                    else:
                        x_batch_val = None
                        batch_data_tuple_val = batch_full_data_val 
                    
                    batch_full_data_reconstructed_val = (x_batch_val,) + batch_data_tuple_val
                    
                    # Validation considering training = False strictly for metrics updates
                    nn_val_batch = self(x_batch_val, training = False)
                    
                    if(pre_training):
                        val_batch_loss = self.loglikelihood_loss_pretrain(nn_output = nn_val_batch, data = batch_full_data_reconstructed_val)
                    else:
                        val_batch_loss = self.loglikelihood_loss(self, nn_output = nn_val_batch, data = batch_full_data_reconstructed_val)
                    
                    if(self.losses):
                        regularization_penalty_val = tf.math.add_n(self.losses)
                        
                        current_batch_size_val = tf.cast(tf.shape(batch_data_tuple_val[0])[0], tf.float32)
                        # Use n_val so the penalty scales perfectly for the validation set sum
                        batch_fraction_val = current_batch_size_val / tf.cast(self.n_val, tf.float32)
                        
                        val_batch_loss = val_batch_loss + regularization_penalty_val * batch_fraction_val
                        
                    current_loss_val += val_batch_loss
                
                # Write to validation history
                val_loss_history = val_loss_history.write(epoch, current_loss_val)

            # If validation = True and force_training_validation = True, we are asking the model to save the validation data loss,
            # but do not use it for early stopping. Essentially, this variable controls whether we want to observe the loss
            # behaviour on the validation set when we ignore it in training (useful for didactic purposes. Showing how the model overfits)
            if(force_training_validation or not validation):
                current_loss = current_loss_train
            else:
                current_loss = current_loss_val
            
            # if(reduce_lr or early_stopping):

            # Only start tracking the best metric after the warmup period
            # that avoids the model from getting low loss values from an initial stage of training
            # where the model may had been in a degenerate, unstable state, yet with a pathological low loss value (burnin phase)
            if( epoch >= tf.math.minimum(early_stopping_warmup, reduce_lr_warmup) ):
                # Check if the loss improved by at least the min_delta
                if(current_loss < (best_metric - reduce_lr_min_delta)):
                    best_metric_epoch = epoch
                    best_metric = current_loss
                    
                    # Update the existing variables inside the self.best_weights object.
                    for i, w in enumerate(self.weights):
                        self.best_weights[i].assign(w)

                    lr_wait = tf.constant(0, dtype = tf.int32)
                    es_wait = tf.constant(0, dtype = tf.int32)
                else:
                    if(epoch >= reduce_lr_warmup):
                        lr_wait = lr_wait + 1
                    if(epoch >= early_stopping_warmup):
                        es_wait = es_wait + 1

                # If it has passed early_stopping_patience epochs with no improvement in the loss function, halts training
                if(early_stopping and (es_wait >= early_stopping_patience) and (epoch > early_stopping_warmup)):
                    if(verbose):
                        tf.print("\nConvergence criterion reached. Stopping.")
                        tf.print("Restoring best weights...")
                    
                    # Restoring best weights
                    for i, w in enumerate(self.weights):
                        self.weights[i].assign( self.best_weights[i] )
    
                    convergence_reason = "stopped_improving"
                    stop_training = True

                if(not stop_training):
                    if(lr_cooldown_counter > 0):
                        lr_cooldown_counter = lr_cooldown_counter - 1
                        lr_wait = tf.constant(0, dtype = tf.int32)
                    else:                                
                        if(reduce_lr and (lr_wait >= reduce_lr_patience) and (epoch > reduce_lr_warmup)):
                            # Decay the learning rates
                            if(self.independent_pars_use):
                                old_lr_ind = self.optimizer_independent.learning_rate
                                new_lr_ind = tf.maximum(old_lr_ind * reduce_lr_factor, reduce_lr_min_lr)
                                self.optimizer_independent.learning_rate.assign( new_lr_ind )
                                
                            if(self.neural_network_use):
                                old_lr_nn = self.optimizer_nn.learning_rate
                                new_lr_nn = tf.maximum(old_lr_nn * reduce_lr_factor, reduce_lr_min_lr)
                                self.optimizer_nn.learning_rate.assign( new_lr_nn )

                            # If minimum learning rate is reached, stop training
                            if( early_stopping and (tf.equal(new_lr_ind, reduce_lr_min_lr) or tf.equal(new_lr_nn, reduce_lr_min_lr)) and (epoch > early_stopping_warmup) ):
                                if(verbose):
                                    tf.print("\nConvergence criterion reached. Stopping.")
                                    tf.print("Restoring best weights...")
                                # Restoring best weights
                                for i, w in enumerate(self.weights):
                                    self.weights[i].assign(self.best_weights[i])

                                convergence_reason = "minimal_learning_rate"
                                stop_training = True
                            
                            # Right after reducing learning rate, set a cooldown for it to settle (For the next reduce_lr_cooldown epochs, it won't be reduced)
                            lr_cooldown_counter = reduce_lr_cooldown
                            # Redefine the learning rate counter to zero
                            lr_wait = tf.constant(0, dtype = tf.int32)
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
            # --------------------------------------------- Native progress tracker without great performance loss ---------------------------------------------
            if(verbose and epoch % print_freq == 0):
                if(epoch > 0):
                    current_time = tf.cast(tf.py_function(func=lambda: time.time(), inp=[], Tout=tf.float64), tf.float64)
                    elapsed_time = current_time - start_time
                    epochs_per_sec = tf.cast(epoch, tf.float64) / elapsed_time
                    
                    # If epochs_per_sec < 1, that means eack epoch takes longer than a second. We take its reciprocal to obtain sec_per_epoch
                    if(epochs_per_sec < 1):
                        tf.print(
                            "\rOptimizing... Epoch: [", epoch, "/", epochs, "] ",
                            "| Loss: ", current_loss, 
                            "| Best Loss: ", best_metric,
                            "| Speed: ", tf.cast(1 / epochs_per_sec, tf.float32), " s/epoch   ",
                            "| Elapsed Time: ", tf.cast(elapsed_time, tf.float32), " s   ",
                            end = ""
                        )
                    else:
                        tf.print(
                            "\rOptimizing... Epoch: [", epoch, "/", epochs, "] ",
                            "| Loss: ", current_loss, 
                            "| Best Loss: ", best_metric,
                            "| Speed: ", tf.cast(epochs_per_sec, tf.int32), " epoch/s   ",
                            "| Elapsed Time: ", tf.cast(elapsed_time, tf.float32), " s   ",
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

        if(validation):
            val_loss_history = val_loss_history.stack()
        else:
            val_loss_history = None
        
        loss_history = loss_history.stack()
        nn_learning_rate_history = nn_learning_rate_history.stack()
        
        return final_epoch, convergence_reason, loss_history, val_loss_history, nn_learning_rate_history, best_metric_epoch, best_metric

    @tf.function(jit_compile = False, reduce_retracing = True)
    def _compiled_training_loop_rawdata(self, x_train, data_train,
                                        epochs, batch_size,
                                        shuffle = True,
                                        validation = False, x_val = None, data_val = None, force_training_validation = False,
                                        early_stopping = True,
                                        early_stopping_patience = tf.constant(10, dtype = tf.int32),
                                        early_stopping_warmup = tf.constant(0, dtype = tf.int32),
                                        reduce_lr = True,
                                        reduce_lr_warmup = tf.constant(0, dtype = tf.int32),
                                        reduce_lr_factor = tf.constant(0.5, dtype = tf.float32),
                                        reduce_lr_min_delta = tf.constant(0.0, dtype = tf.float32),
                                        reduce_lr_patience = tf.constant(5, dtype = tf.int32),
                                        reduce_lr_cooldown = tf.constant(20, dtype = tf.int32),
                                        reduce_lr_min_lr = tf.constant(5e-4, dtype = tf.float32),
                                        deterministic = True,
                                        pre_training = False,
                                        fine_tuning = False,
                                        verbose = True, print_freq = tf.constant(100, dtype = tf.int32)):
        """
            Executes the entire optimization loop purely in C++.
            Bypasses all Keras callbacks, progress bars, and Python overhead.
        """
        # Training variables
        final_epoch = tf.constant(0, dtype = tf.int32)
        current_loss = tf.constant(0.0, dtype = tf.float32)
        stop_training = False
        
        lr_independent = self.optimizer_independent.learning_rate
        lr_nn = self.optimizer_nn.learning_rate

        # Set up history variables to track convergence profile
        loss_history = tf.TensorArray(tf.float32, size = epochs, clear_after_read = False)
        val_loss_history = tf.TensorArray(tf.float32, size = epochs, clear_after_read = False)
        
        nn_learning_rate_history = tf.TensorArray(tf.float32, size = epochs, clear_after_read = False)
        
        start_time = tf.cast(tf.py_function(func=lambda: time.time(), inp=[], Tout=tf.float64), tf.float64)

        n_samples = tf.shape(data_train[0])[0]

        # ReduceLROnPlateau routine variables
        
        # Reduce learning rate wait
        lr_wait = tf.constant(0, dtype = tf.int32)
        # Early stopping wait
        es_wait = tf.constant(0, dtype = tf.int32)

        current_loss_train = np.nan
        current_loss_val = 0.0
        
        lr_cooldown_counter = tf.constant(0, dtype = tf.int32)
        best_metric = tf.constant(float('inf'), dtype = tf.float32)
        best_metric_epoch = tf.constant(0, dtype = tf.int32)
        new_lr_ind = tf.constant(0.0, dtype = tf.float32)
        new_lr_nn = tf.constant(0.0, dtype = tf.float32)

        # Save the initial weights as the best
        # best_weights = [tf.identity(w) for w in self.trainable_variables]
        # self.best_weights = [tf.Variable(w, trainable=False) for w in self.trainable_variables]

        # By default, if convergence_reason does not change during training, that's because the algorithm reached the final
        # epoch without ever converging
        convergence_reason = "all_epochs"
        
        for epoch in tf.range(epochs):
            # At the start of each epoch, assign the current epoch to a global variable
            self.current_epoch.assign( tf.cast(epoch, tf.int32) )
            
            # Shuffle data at the start of each epoch, if desired
            if(shuffle):
                indices = tf.random.shuffle( tf.range(n_samples) )
                # If we are dealing with a purely statistical model (no regression in any parameter) x_train may be None
                x_epoch = None
                if(x_train is not None):
                    x_epoch = tf.gather(x_train, indices)
                data_epoch = tuple([tf.gather(d, indices) for d in data_train])
            else:
                x_epoch = x_train
                data_epoch = data_train

            batch_num = 0
            # Cycle through all batches
            for start_idx in tf.range(0, n_samples, batch_size):
                batch_num += 1
                
                # Ensure the last batch doesn't go out of bounds
                end_idx = tf.minimum(start_idx + batch_size, n_samples)

                # Slice the batch out of RAM instantly
                x_batch = None
                if(x_train is not None):
                    x_batch = x_epoch[start_idx : end_idx]
                batch_data_tuple = tuple( [d[start_idx : end_idx] for d in data_epoch] )

                # Reconstruct full_data for the loss function
                batch_full_data = (x_batch,) + batch_data_tuple
                
                # 1. Forward Pass & Loss Computation
                with tf.GradientTape() as tape:

                    # If model finetuning, treat the neural network as a deterministic black box
                    # If there are dropout layers, the output will be random and break training
                    if(fine_tuning):
                        nn_output_batch = self(x_batch, training = False)
                    else:
                        nn_output_batch = self(x_batch, training = True)
                    
                    if(pre_training):
                        loss_value = self.loglikelihood_loss_pretrain(nn_output = nn_output_batch, data = batch_full_data)
                    else:
                        loss_value = self.loglikelihood_loss(self, nn_output = nn_output_batch, data = batch_full_data)

                    # loss_history = loss_history.write(epoch, loss_value)
                                      
                    # Automatic regularization from layer definitions. Check if any layer in the model generated a regularization loss
                    if(self.losses):
                        # sums all tensors in the self.losses list
                        regularization_penalty = tf.math.add_n( self.losses )

                        batch_fraction = tf.cast(tf.shape(x_batch)[0], tf.float32) / tf.cast(n_samples, tf.float32)
                        # Add it to the base log-likelihood
                        loss_value = loss_value + regularization_penalty * batch_fraction
                
                gradients = tape.gradient(loss_value, self.trainable_variables)
    
                # Gradient trap: Check if any gradient in the entire network became NaN or Inf
                has_nan_grad = tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None])
                has_inf_grad = tf.reduce_any([tf.reduce_any(tf.math.is_inf(g)) for g in gradients if g is not None])
                if tf.math.logical_or(has_nan_grad, has_inf_grad):
                    tf.print("\n[!] FATAL: Gradients exploded to NaN/Inf at Epoch:", epoch)
                    convergence_reason = "nan_gradients"
                    stop_training = True
                    break
                
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
                
                if(self.independent_pars_use):
                    for i in range( len(self.gradient_accumulation_independent_pars) ):
                        self.gradient_accumulation_independent_pars[i].assign_add( independent_gradients[i] )
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

            
            nn_learning_rate_history = nn_learning_rate_history.write(epoch, self.optimizer_nn.learning_rate)
            # --------------------------------------------------------------- Evaluate stop criteria ---------------------------------------------------------------
            # For comparisons we will be using the raw value in order to avoid potential link functions exponential explosions
            # if(epoch >= 0):
            # ------------------------------------ ReduceLROnPlateau custom mechanism. Hard-coded implementation needed for performance issues ------------------------------------
            if(reduce_lr or early_stopping):
                # Always compute the true, intact training loss
                batch_train_full = (x_train,) + tuple(data_train)
                nn_train_full = self(x_train, training = False)
                
                if(pre_training):
                    current_loss_train = self.loglikelihood_loss_pretrain(nn_output = nn_train_full, data = batch_train_full)
                else:
                    current_loss_train = self.loglikelihood_loss(self, nn_output = nn_train_full, data = batch_train_full)
                
                # Write the true, full-dataset training loss to history!
                loss_history = loss_history.write(epoch, current_loss_train)

                # Compute the validation loss ONLY if requested
                if(validation):
                    batch_val_data = (x_val,) + tuple(data_val)
                    nn_val_batch = self(x_val, training = False)
                    
                    if(pre_training):
                        current_loss_val = self.loglikelihood_loss_pretrain(nn_output = nn_val_batch, data = batch_val_data)
                    else:
                        current_loss_val = self.loglikelihood_loss(self, nn_output = nn_val_batch, data = batch_val_data)
                    
                    # Write to validation history
                    val_loss_history = val_loss_history.write(epoch, current_loss_val)

                # If validation = True and force_training_validation = True, we are asking the model to save the validation data loss,
                # but do not use it for early stopping. Essentially, this variable controls whether we want to observe the loss
                # behaviour on the validation set when we ignore it in training (useful for didactic purposes. Showing how the model overfits)
                if(force_training_validation or not validation):
                    current_loss = current_loss_train
                else:
                    current_loss = current_loss_val

                # Only start tracking the best metric after the warmup period
                # that avoids the model from getting low loss values from an initial stage of training
                # where the model may had been in a degenerate, unstable state, yet with a pathological low loss value (burnin phase)
                if(epoch >= early_stopping_warmup):
                    # Check if the loss improved by at least the min_delta
                    if(current_loss < (best_metric - reduce_lr_min_delta)):
                        best_metric_epoch = epoch
                        best_metric = current_loss
                        # best_weights = [tf.identity(w) for w in self.trainable_variables]
                        # self.best_weights = [tf.Variable(w, trainable = False) for w in self.weights]
    
                        # Do NOT create a new list. Update the existing variables in-place.
                        for i, w in enumerate(self.weights):
                            self.best_weights[i].assign(w)
    
                        lr_wait = tf.constant(0, dtype = tf.int32)
                        es_wait = tf.constant(0, dtype = tf.int32)
                    else:
                        lr_wait = lr_wait + 1
                        es_wait = es_wait + 1
                
                # If it has passed early_stopping_patience epochs with no improvement in the loss function, halts training
                if(early_stopping and es_wait >= early_stopping_patience and epoch > early_stopping_warmup):
                    if(verbose):
                        tf.print("\nConvergence criterion reached. Stopping.")
                        tf.print("Restoring best weights...")
                    # Restoring best weights
                    for i, w in enumerate(self.weights):
                        self.weights[i].assign(self.best_weights[i])
                        # tf.print(self.best_weights[i])

                    convergence_reason = "stopped_improving"
                    stop_training = True
                
                if(not stop_training):
                    if(lr_cooldown_counter > 0):
                        lr_cooldown_counter = lr_cooldown_counter - 1
                        lr_wait = tf.constant(0, dtype = tf.int32)
                    else:                                
                        if(reduce_lr and lr_wait >= reduce_lr_patience and epoch > reduce_lr_warmup):
                            # Decay the learning rates
                            if(self.independent_pars_use):
                                old_lr_ind = self.optimizer_independent.learning_rate
                                new_lr_ind = tf.maximum(old_lr_ind * reduce_lr_factor, reduce_lr_min_lr)
                                self.optimizer_independent.learning_rate.assign( new_lr_ind )
                                
                            if(self.neural_network_use):
                                old_lr_nn = self.optimizer_nn.learning_rate
                                new_lr_nn = tf.maximum(old_lr_nn * reduce_lr_factor, reduce_lr_min_lr)
                                self.optimizer_nn.learning_rate.assign( new_lr_nn )

                            # If minimum learning rate is reached, stop training
                            if( early_stopping and (tf.equal(new_lr_ind, reduce_lr_min_lr) or tf.equal(new_lr_nn, reduce_lr_min_lr)) and (epoch > early_stopping_warmup) ):
                                if(verbose):
                                    tf.print("\nConvergence criterion reached. Stopping.")
                                    tf.print("Restoring best weights...")
                                # Restoring best weights
                                for i, w in enumerate(self.weights):
                                    self.weights[i].assign(self.best_weights[i])
                                    # tf.print(self.best_weights[i])

                                convergence_reason = "minimal_learning_rate"
                                stop_training = True
                            
                            # Right after reducing learning rate, set a cooldown for it to settle (For the next reduce_lr_cooldown epochs, it won't be reduced)
                            lr_cooldown_counter = reduce_lr_cooldown
                            # Redefine the learning rate counter to zero
                            lr_wait = tf.constant(0, dtype = tf.int32)    
                    
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
            # --------------------------------------------- Native progress tracker without great performance lose ---------------------------------------------
            if(verbose and epoch % print_freq == 0):
                if(epoch > 0):
                    current_time = tf.cast(tf.py_function(func=lambda: time.time(), inp=[], Tout=tf.float64), tf.float64)
                    elapsed_time = current_time - start_time
                    epochs_per_sec = tf.cast(epoch, tf.float64) / elapsed_time
                    
                    # If epochs_per_sec < 1, that means eack epoch takes longer than a second. We take its reciprocal to obtain sec_per_epoch
                    if(epochs_per_sec < 1):
                        tf.print(
                            "\rOptimizing... Epoch: [", epoch, "/", epochs, "] ",
                            "| Loss: ", current_loss, 
                            "| Best Loss: ", best_metric,
                            "| Speed: ", tf.cast(1 / epochs_per_sec, tf.float32), " s/epoch   ",
                            "| Elapsed Time: ", tf.cast(elapsed_time, tf.float32), " s   ",
                            end = ""
                        )
                    else:
                        tf.print(
                            "\rOptimizing... Epoch: [", epoch, "/", epochs, "] ",
                            "| Loss: ", current_loss, 
                            "| Best Loss: ", best_metric,
                            "| Speed: ", tf.cast(epochs_per_sec, tf.int32), " epoch/s   ",
                            "| Elapsed Time: ", tf.cast(elapsed_time, tf.float32), " s   ",
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

        if(validation):
            val_loss_history = val_loss_history.write(0, val_loss_history.read(1))
            val_loss_history = val_loss_history.stack()
        else:
            val_loss_history = None
        
        loss_history = loss_history.stack()

        nn_learning_rate_history = nn_learning_rate_history.stack()
        
        return final_epoch, convergence_reason, loss_history, val_loss_history, nn_learning_rate_history, best_metric_epoch, best_metric

    
    def train_model(self, epochs, x, data = None,
                    shuffle = True,
                    validation = False, n_train = None, n_val = None, x_val = None, data_val = None, val_prop = None, force_training_validation = False,
                    optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                    optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                    fine_tune_independent_lr = None, fine_tune_nn_lr = None,
                    train_batch_size = None, val_batch_size = None,
                    buffer_size = 4096, gradient_accumulation_steps = None,
                    early_stopping = True, early_stopping_patience = 10, early_stopping_warmup = 100,
                    reduce_lr = True, reduce_lr_warmup = 0, reduce_lr_factor = 0.5, reduce_lr_min_delta = 0.0, reduce_lr_patience = 5,
                    reduce_lr_cooldown = 0, reduce_lr_min_lr = 1e-5,
                    fine_tune = True,
                    get_covariances = True, covariance_jitter = 1.0e-6,
                    finetune_epochs = None,
                    finetune_early_stopping = None, finetune_early_stopping_patience = None,
                    finetune_early_stopping_warmup = None,
                    finetune_reduce_lr = None, finetune_reduce_lr_warmup = None, finetune_reduce_lr_factor = None,
                    finetune_reduce_lr_min_delta = None, finetune_reduce_lr_patience = None,
                    finetune_reduce_lr_cooldown = None, finetune_reduce_lr_min_lr = None,
                    deterministic = True,
                    verbose = True, print_freq = 25, track_time = True):
        
        # Format the input data accordingly and prepare training and validation datasets
        self.config_training(x, data = data,
                             shuffle = shuffle,
                             validation = validation, n_train = n_train, n_val = n_val,
                             x_val = x_val, data_val = data_val, val_prop = val_prop,
                             optimizer_independent = optimizer_independent,
                             optimizer_nn = optimizer_nn,
                             train_batch_size = train_batch_size, val_batch_size = val_batch_size,
                             buffer_size = buffer_size, gradient_accumulation_steps = gradient_accumulation_steps,
                             verbose = verbose)
        
        # Force the optimizers to build their state variables in Python so they don't try to create them inside the C++ when function is called a second time
        if(self.independent_pars_use):
            self.optimizer_independent.build( self.trainable_variables[:len(self.independent_pars)] )
        if(self.neural_network_use):
            self.optimizer_nn.build( self.trainable_variables[len(self.independent_pars):] )

        independent_learning_rate = tf.identity( optimizer_independent.learning_rate )
        nn_learning_rate = tf.identity( optimizer_nn.learning_rate )
        
        epochs = tf.constant(epochs, dtype = tf.int32)

        early_stopping_patience = tf.constant(early_stopping_patience, dtype = tf.int32)
        early_stopping_warmup = tf.constant(early_stopping_warmup, dtype = tf.int32)

        reduce_lr_warmup = tf.constant(reduce_lr_warmup, dtype = tf.int32)
        reduce_lr_factor = tf.constant(reduce_lr_factor, dtype = tf.float32)
        reduce_lr_min_delta = tf.constant(reduce_lr_min_delta, dtype = tf.float32)
        reduce_lr_patience = tf.constant(reduce_lr_patience, dtype = tf.int32)
        reduce_lr_cooldown = tf.constant(reduce_lr_cooldown, dtype = tf.int32)
        reduce_lr_min_lr = tf.constant(reduce_lr_min_lr, dtype = tf.float32)

        print_freq = tf.constant(print_freq, dtype = tf.int32)

        # If user need a deterministic outcome for reproducibility, set all seeds to the global defined seed before training
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

        # self.best_weights = [tf.Variable(w, trainable = False) for w in self.trainable_variables]
        self.best_weights = [tf.Variable(w, trainable = False) for w in self.weights]
        
        self.training = True
        # If self.data is not None, that means the user passed raw data to the model
        # tf.data.Datasets are not needed for training. That allows a faster training routine
        if(self.data is not None):
            # Compiled training routine
            last_epoch, convergence_reason, loss_history, val_loss_history, \
            nn_learning_rate_history, best_metric_epoch, best_metric = self._compiled_training_loop_rawdata(
                self.x_train, self.data_train, epochs,
                tf.constant(self.train_batch_size, dtype = tf.int32),
                shuffle = shuffle,
                validation = validation, x_val = self.x_val, data_val = self.data_val, force_training_validation = force_training_validation,
                early_stopping = early_stopping,
                early_stopping_patience = early_stopping_patience,
                early_stopping_warmup = early_stopping_warmup,
                reduce_lr = reduce_lr,
                reduce_lr_warmup = reduce_lr_warmup,
                reduce_lr_factor = reduce_lr_factor,
                reduce_lr_min_delta = reduce_lr_min_delta,
                reduce_lr_patience = reduce_lr_patience,
                reduce_lr_cooldown = reduce_lr_cooldown,
                reduce_lr_min_lr = reduce_lr_min_lr,
                deterministic = deterministic,
                pre_training = False,
                fine_tuning = False,
                verbose = verbose,
                print_freq = print_freq
            )
        else:
            # Compiled training routine
            last_epoch, convergence_reason, loss_history, val_loss_history, \
            nn_learning_rate_history, best_metric_epoch, best_metric = self._compiled_training_loop_dataset(
                self.train_dataset, epochs,
                shuffle = shuffle,
                validation = validation, val_dataset = self.val_dataset, force_training_validation = force_training_validation,
                early_stopping = early_stopping,
                early_stopping_patience = early_stopping_patience,
                early_stopping_warmup = early_stopping_warmup,
                reduce_lr = reduce_lr,
                reduce_lr_warmup = reduce_lr_warmup,
                reduce_lr_factor = reduce_lr_factor,
                reduce_lr_min_delta = reduce_lr_min_delta,
                reduce_lr_patience = reduce_lr_patience,
                reduce_lr_cooldown = reduce_lr_cooldown,
                reduce_lr_min_lr = reduce_lr_min_lr,
                deterministic = deterministic,
                pre_training = False,
                fine_tuning = False,
                verbose = verbose,
                print_freq = print_freq
            )
        self.training = False

        self.last_epoch = last_epoch
        self.loss_history = loss_history
        self.val_loss_history = val_loss_history
        self.convergence_reason = convergence_reason
        self.nn_learning_rate_history = nn_learning_rate_history
        self.best_metric_epoch = best_metric_epoch
        self.best_metric = best_metric

        # Save the best weights previous to finetuning so we can compare metrics later if neccessary
        self.pre_finetuning_best_weights = [ tf.Variable(w, trainable = False) for w in self.weights ]
        
        if(verbose):
            print("\nDone.")

        # If neural network is not being used, there is no need to fine-tune the model as it has already converged
        if(fine_tune and self.neural_network_use):
            if(fine_tune_independent_lr is None):
                fine_tune_independent_lr = independent_learning_rate
            if(fine_tune_nn_lr is None):
                fine_tune_nn_lr = nn_learning_rate
            # Resets the optimizers, fixing the learning rate to be the user defined ones
            if(self.independent_pars_use):
                self.optimizer_independent.learning_rate = fine_tune_independent_lr
                self.optimizer_independent.build( self.trainable_variables[:len(self.independent_pars)] )
            if(self.neural_network_use):
                self.optimizer_nn.learning_rate = fine_tune_nn_lr
                self.optimizer_nn.build( self.trainable_variables[len(self.independent_pars):] )
            
            # Before fine-tuning, since the model already learned the basis structure, there is no need to consider
            # parameters warmup_time anymore (That would only slow down training) Then, we remove those for fine-tuning
            original_parameters = copy.deepcopy(self.parameters)
            for parameter in self.parameters:
                if("warmup_time" in self.parameters[parameter]):
                    self.parameters[parameter]["warmup_time"] = 0
            
            if(verbose):
                print("Initializing model fine tuning (only independent parameters and last-layer)")
            # Format the input data accordingly and prepare training and validation datasets

            # Specifically for fine-tuning, we must ensure the model to converge to the maximum log-likelihood in the training data
            # To avoid user caused problems, we enforce a single batch the size of the data with train_batch_size = None and
            # gradient_accumulation_steps = None
            # We are essentially assuring the Full-Batch Gradient Descent mechanics, avoiding noisy mini-batch updates
            
            # If user passed raw data, training configuration follows differently
            if(self.data is not None):
                # Also, since the first call to config_training already defined the validation data (if x_val and data_val are originally None)
                # Here, we explicitly pass self.x_train, self.x_val, self.data_train and self.data_val
                # Otherwise, this function may completely shuffle the data and result in completely different loss values and data leakage!
                self.config_training(self.x_train, self.data_train,
                                     shuffle = shuffle,
                                     validation = validation, val_prop = val_prop, x_val = self.x_val, data_val = self.data_val,
                                     optimizer_independent = self.optimizer_independent,
                                     optimizer_nn = self.optimizer_nn,
                                     train_batch_size = None, val_batch_size = None,
                                     buffer_size = buffer_size, gradient_accumulation_steps = None,
                                     verbose = verbose)
            else:
                self.config_training(self.train_dataset, data = None,
                                     shuffle = shuffle,
                                     validation = validation, n_train = self.n_train, n_val = self.n_val,
                                     x_val = self.val_dataset, data_val = None, val_prop = None,
                                     optimizer_independent = self.optimizer_independent,
                                     optimizer_nn = self.optimizer_nn,
                                     train_batch_size = self.train_batch_size, val_batch_size = self.val_batch_size,
                                     buffer_size = buffer_size, gradient_accumulation_steps = None,
                                     verbose = verbose)
            # Set all but the last layers as non-trainable
            for i in range( len(self.layers)-1 ):
                self.layers[i].trainable = False
                
            # Redefine the gradients accumulation objects to match the current trainable_variables structure
            self.define_gradients()
            # Redefine the best weights object so they match the current trainable_variables structure
            # self.best_weights = [tf.Variable(w, trainable = False) for w in self.trainable_variables]
            self.best_weights = [tf.Variable(w, trainable = False) for w in self.weights]

            # During the fine-tuning phase, it is desired to obtain a local maxima for the log-likelihood.
            # Therefore, we necessarily fix the training data loss to be the observed metric for early stopping and learning rate reduction
            # We also remove the possibility of reducing the learning rate and to 

            # If hyperparameters for fine tuning were not provided, consider the training parameters instead
            if(finetune_epochs is None):
                finetune_epochs = epochs
            if(finetune_early_stopping is None):
                finetune_early_stopping = early_stopping
            if(finetune_early_stopping_patience is None):
                finetune_early_stopping_patience = early_stopping_patience
            if(finetune_early_stopping_warmup is None):
                finetune_early_stopping_warmup = early_stopping_warmup
            if(finetune_reduce_lr is None):
                finetune_reduce_lr = reduce_lr
            if(finetune_reduce_lr_warmup is None):
                finetune_reduce_lr_warmup = reduce_lr_warmup
            if(finetune_reduce_lr_factor is None):
                finetune_reduce_lr_factor = reduce_lr_factor
            if(finetune_reduce_lr_min_delta is None):
                finetune_reduce_lr_min_delta = reduce_lr_min_delta
            if(finetune_reduce_lr_patience is None):
                finetune_reduce_lr_patience = reduce_lr_patience
            if(finetune_reduce_lr_cooldown is None):
                finetune_reduce_lr_cooldown = reduce_lr_cooldown
            if(finetune_reduce_lr_min_lr is None):
                finetune_reduce_lr_min_lr = reduce_lr_min_lr
                
            finetune_epochs = tf.constant(finetune_epochs, dtype = tf.int32)
            
            finetune_early_stopping_patience = tf.constant(finetune_early_stopping_patience, dtype = tf.int32)
            finetune_early_stopping_warmup = tf.constant(finetune_early_stopping_warmup, dtype = tf.int32)
    
            finetune_reduce_lr_warmup = tf.constant(finetune_reduce_lr_warmup, dtype = tf.int32)
            finetune_reduce_lr_factor = tf.constant(finetune_reduce_lr_factor, dtype = tf.float32)
            finetune_reduce_lr_min_delta = tf.constant(finetune_reduce_lr_min_delta, dtype = tf.float32)
            finetune_reduce_lr_patience = tf.constant(finetune_reduce_lr_patience, dtype = tf.int32)
            finetune_reduce_lr_cooldown = tf.constant(finetune_reduce_lr_cooldown, dtype = tf.int32)
            finetune_reduce_lr_min_lr = tf.constant(finetune_reduce_lr_min_lr, dtype = tf.float32)

            # If self.data is not None, that means the user passed raw data to the model
            # tf.data.Datasets are not needed for training. That allows a faster training routine
            if(self.data is not None):
                # Compiled training routine
                last_epoch, convergence_reason, loss_history, val_loss_history, \
                nn_learning_rate_history, best_metric_epoch, best_metric = self._compiled_training_loop_rawdata(
                    self.x_train, self.data_train, finetune_epochs,
                    tf.constant(self.train_batch_size, dtype = tf.int32),
                    shuffle = shuffle,
                    validation = validation, x_val = self.x_val, data_val = self.data_val, force_training_validation = True,
                    early_stopping = finetune_early_stopping,
                    early_stopping_patience = finetune_early_stopping_patience,
                    early_stopping_warmup = finetune_early_stopping_warmup,
                    reduce_lr = finetune_reduce_lr,
                    reduce_lr_warmup = finetune_reduce_lr_warmup,
                    reduce_lr_factor = finetune_reduce_lr_factor,
                    reduce_lr_min_delta = finetune_reduce_lr_min_delta,
                    reduce_lr_patience = finetune_reduce_lr_patience,
                    reduce_lr_cooldown = finetune_reduce_lr_cooldown,
                    reduce_lr_min_lr = finetune_reduce_lr_min_lr,
                    deterministic = deterministic,
                    pre_training = False,
                    fine_tuning = True,
                    verbose = verbose,
                    print_freq = print_freq
                )
            else:
                last_epoch, convergence_reason, loss_history, val_loss_history, \
                nn_learning_rate_history, best_metric_epoch, best_metric = self._compiled_training_loop_dataset(
                    self.train_dataset, finetune_epochs,
                    shuffle = shuffle,
                    validation = validation, val_dataset = self.val_dataset, force_training_validation = True,
                    early_stopping = finetune_early_stopping,
                    early_stopping_patience = finetune_early_stopping_patience,
                    early_stopping_warmup = finetune_early_stopping_warmup,
                    reduce_lr = finetune_reduce_lr,
                    reduce_lr_warmup = finetune_reduce_lr_warmup,
                    reduce_lr_factor = finetune_reduce_lr_factor,
                    reduce_lr_min_delta = finetune_reduce_lr_min_delta,
                    reduce_lr_patience = finetune_reduce_lr_patience,
                    reduce_lr_cooldown = finetune_reduce_lr_cooldown,
                    reduce_lr_min_lr = finetune_reduce_lr_min_lr,
                    deterministic = deterministic,
                    pre_training = False,
                    fine_tuning = True,
                    verbose = verbose,
                    print_freq = print_freq
                )

            self.last_epoch_finetune = last_epoch
            self.loss_history_finetune = loss_history
            self.val_loss_history_finetune = val_loss_history
            self.convergence_reason_finetune = convergence_reason
            self.nn_learning_rate_history_finetune = nn_learning_rate_history
            self.best_metric_epoch_finetune = best_metric_epoch
            self.best_metric_finetune = best_metric
            
            self.optimizer_independent.learning_rate = independent_learning_rate
            self.optimizer_nn.learning_rate = nn_learning_rate

            # After training, return the user configuration for the parameters with warmup times
            self.parameters = original_parameters
            
            if(verbose):
                print("\nDone.")
        
        if(get_covariances):
            if(not fine_tune):
                warnings.simplefilter("always", UserWarning)
                warnings.warn(
                    "get_covariances = True, but fine_tune = False\n" + \
                    "To stabilize optimizer at a local maximum, it is highly" + \
                    "recommended to perform fine-tuning before obtaining covariance matrices.",
                    category = UserWarning,
                )
                warnings.simplefilter("default", UserWarning)
                
            if(verbose):
                print("Extracting covariance structure.")
            # Obtain covariance estimates for the neural network induced parameters
            # covariance_jitter stands for the diagonal added to the last-layer Hessian
            # In the Bayesian Neural Networks interpretation, the prior variance of the last-layer
            # is given by Var[\omega] = 1 / jitter
            # By setting it small, we account for a small, ridge term to avoid a singular hessian
            self.get_covariances(jitter = covariance_jitter)
            if(verbose):
                print("Done.")

        execution_time = time.time() - start_time
        if(verbose and track_time):
            print("Optimization finished in {:.3f} seconds.".format(execution_time))


    def pre_train_model(self, epochs, x, data = None,
                        shuffle = True,
                        validation = False, n_train = None, n_val = None, x_val = None, data_val = None, val_prop = None, force_training_validation = False,
                        optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                        optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                        train_batch_size = None, val_batch_size = None,
                        buffer_size = 4096, gradient_accumulation_steps = None,
                        early_stopping = True, early_stopping_patience = 10, early_stopping_warmup = 100,
                        reduce_lr = True, reduce_lr_warmup = 0, reduce_lr_factor = 0.5, reduce_lr_min_delta = 0.0, reduce_lr_patience = 5,
                        reduce_lr_cooldown = 0, reduce_lr_min_lr = 1e-5,
                        deterministic = True,
                        verbose = True, print_freq = 100, track_time = True):
        
        # Format the input data accordingly and prepare training and validation datasets
        self.config_training(x, data = data,
                             shuffle = shuffle,
                             validation = validation, n_train = n_train, n_val = n_val,
                             x_val = x_val, data_val = data_val, val_prop = val_prop,
                             optimizer_independent = optimizer_independent,
                             optimizer_nn = optimizer_nn,
                             train_batch_size = train_batch_size, val_batch_size = val_batch_size,
                             buffer_size = buffer_size, gradient_accumulation_steps = gradient_accumulation_steps,
                             verbose = verbose)

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
            independent_learning_rate = tf.identity( optimizer_independent.learning_rate )
            nn_learning_rate = tf.identity( optimizer_nn.learning_rate )
            
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

            # Initialize best weights object
            self.best_weights = [tf.Variable(w, trainable = False) for w in self.weights]
            
            epochs = tf.constant(epochs, dtype = tf.int32)

            early_stopping_patience = tf.constant(early_stopping_patience, dtype = tf.int32)
            early_stopping_warmup = tf.constant(early_stopping_warmup, dtype = tf.int32)
            reduce_lr_warmup = tf.constant(reduce_lr_warmup, dtype = tf.int32)
            reduce_lr_factor = tf.constant(reduce_lr_factor, dtype = tf.float32)
            reduce_lr_min_delta = tf.constant(reduce_lr_min_delta, dtype = tf.float32)
            reduce_lr_patience = tf.constant(reduce_lr_patience, dtype = tf.int32)
            reduce_lr_cooldown = tf.constant(reduce_lr_cooldown, dtype = tf.int32)
            reduce_lr_min_lr = tf.constant(reduce_lr_min_lr, dtype = tf.float32)
    
            print_freq = tf.constant(print_freq, dtype = tf.int32)

            # If self.data is not None, that means the user passed raw data to the model
            # tf.data.Datasets are not needed for training. That allows a faster training routine
            if(self.data is not None):
                # Compiled training routine
                last_epoch, convergence_reason, loss_history, val_loss_history, \
                nn_learning_rate_history, best_metric_epoch, best_metric = self._compiled_training_loop_rawdata(
                    self.x_train, self.data_train, epochs,
                    tf.constant(self.train_batch_size, dtype = tf.int32),
                    shuffle = shuffle,
                    validation = validation, x_val = self.x_val, data_val = self.data_val, force_training_validation = force_training_validation,
                    early_stopping = early_stopping,
                    early_stopping_patience = early_stopping_patience,
                    early_stopping_warmup = early_stopping_warmup,
                    reduce_lr = reduce_lr,
                    reduce_lr_warmup = reduce_lr_warmup,
                    reduce_lr_factor = reduce_lr_factor,
                    reduce_lr_min_delta = reduce_lr_min_delta,
                    reduce_lr_patience = reduce_lr_patience,
                    reduce_lr_cooldown = reduce_lr_cooldown,
                    reduce_lr_min_lr = reduce_lr_min_lr,
                    deterministic = deterministic,
                    pre_training = True,
                    fine_tuning = False,
                    verbose = verbose,
                    print_freq = print_freq
                )
            else:
                last_epoch, convergence_reason, loss_history, val_loss_history, \
                nn_learning_rate_history, best_metric_epoch, best_metric = self._compiled_training_loop_dataset(
                    self.train_dataset, epochs,
                    shuffle = shuffle,
                    validation = validation, val_dataset = self.val_dataset, force_training_validation = force_training_validation,
                    early_stopping = early_stopping,
                    early_stopping_patience = early_stopping_patience,
                    early_stopping_warmup = early_stopping_warmup,
                    reduce_lr = reduce_lr,
                    reduce_lr_warmup = reduce_lr_warmup,
                    reduce_lr_factor = reduce_lr_factor,
                    reduce_lr_min_delta = reduce_lr_min_delta,
                    reduce_lr_patience = reduce_lr_patience,
                    reduce_lr_cooldown = reduce_lr_cooldown,
                    reduce_lr_min_lr = reduce_lr_min_lr,
                    deterministic = deterministic,
                    pre_training = True,
                    fine_tuning = False,
                    verbose = verbose,
                    print_freq = print_freq
                )

            self.last_epoch_pretrain = last_epoch
            self.loss_history_pretrain = loss_history
            self.val_loss_history_pretrain = val_loss_history
            self.convergence_reason_pretrain = convergence_reason
            self.best_metric_epoch_pretrain = best_metric_epoch
            self.best_metric_pretrain = best_metric
            self.nn_learning_rate_history_pretrain = nn_learning_rate_history # ADD THIS
            self.optimizer_independent.learning_rate = independent_learning_rate
            self.optimizer_nn.learning_rate = nn_learning_rate

    def _detect_shuffle_in_dataset(self, dataset):
        """
        Recursively traverses a tf.data.Dataset execution graph to find any Shuffle operations.
        """
        # Base case: Check if the current node is a ShuffleDataset
        if("Shuffle" in dataset.__class__.__name__):
            return True
            
        # Recursive case 1: Handle datasets with multiple inputs (like tf.data.Dataset.zip)
        if(hasattr(dataset, '_inputs')):
            for input_ds in dataset._inputs():
                if self._detect_shuffle_in_dataset(input_ds):
                    return True
        # Recursive case 2: Handle standard linear transformations (map, batch, etc.)
        elif hasattr(dataset, '_input_dataset'):
            return self._detect_shuffle_in_dataset(dataset._input_dataset)
            
        return False
    
    def config_training(self, x, data = None,
                        shuffle = True,
                        validation = False, n_train = None, n_val = None, x_val = None, data_val = None, val_prop = None,
                        optimizer_independent = optimizers.Adam(learning_rate = 0.001),
                        optimizer_nn = optimizers.Adam(learning_rate = 0.001),
                        train_batch_size = None, val_batch_size = None,
                        buffer_size = 4096, gradient_accumulation_steps = None,
                        verbose = True):

        # Initialize raw numpy array data variables as None
        # self.x_train, self.data_train = None, None
        # self.x_val, self.data_val = None, None
        
        # If there are no trainable variables, there is no reason to train such a model
        if( len(self.trainable_variables) == 0 ):
            raise RuntimeError(
                "Training failed: the model does not contain any trainable variables. "
                "This model is fixed and cannot be trained."
            )
        self.validation = validation
        
        if(isinstance(x, tf.data.Dataset)):
            # Extract the shape of the dataset's features (ignoring labels if the dataset yields tuples)
            x_spec = x.element_spec[0] if isinstance(x.element_spec, tuple) else x.element_spec
            
            # If the dataset rank is greater than the input_dim rank,
            # And if by taking the first dimension of x_spec out
            # and that first dimension is in fact equal to None,
            # then the user has previously created a batched dataset. No need to handle that here
            # In fact, suppose the input data is given by a tensor with shape (200, 200, 3)
            # Then a batched tf.data.Dataset will return a x_spec with shape (None, 200, 200, 3),
            # while an unbatched Dataset returns a shape (200, 200, 3)
            is_batched = False
            if( (len(x_spec.shape) > len(self.input_dim)) and (x_spec.shape[0] is None) ):
                # Specifically, if when unbatching the Dataset, its shape continues to be different from the input_dim
                if x_spec.shape[1:] != self.input_dim:
                    raise ValueError(
                        f"The provided train dataset appears to be batched (shape: {x_spec.shape}), "
                        f"but its underlying element shape {x_spec.shape[1:]} does not match "
                        f"the expected `input_dim` of {self.input_dim}. "
                        f"Please ensure your dataset has the correct dimensions or wasn't accidentally batched twice."
                    )
                is_batched = True
            
            # -----------------------------
            
            # If dataset was batched by the user, this is the number of batches,
            # otherwise, this is the total sample size
            cardinality = tf.data.experimental.cardinality(x).numpy()
            is_cardinality_known = cardinality not in [tf.data.experimental.UNKNOWN_CARDINALITY, tf.data.experimental.INFINITE_CARDINALITY]

            if(not is_cardinality_known):
                raise ValueError("Cannot process tf.data.Dataset with unknown size. Ensure the dataset has a known cardinality.")


            # If data x given was already batched by the user, we must explore all the possitiblities of such case
            # The cardinality above here will represent the total number of BATCHES, not SAMPLES
            if(is_batched):
                if(n_train is None):
                    raise ValueError("When providing a batched tf.data.Dataset, you must explicitly provide `n_train` (the total number of training samples).")
                self.n_train = n_train

                # Infer batch size from the graph
                sample_batch = next(iter(x))
                if isinstance(sample_batch, tuple):
                    inferred_batch_size = int(sample_batch[0].shape[0])
                else:
                    inferred_batch_size = int(sample_batch.shape[0])
                if( inferred_batch_size != train_batch_size ):
                    if(verbose and train_batch_size is not None):
                        warnings.simplefilter("always", RuntimeWarning)
                        warnings.warn(
                            "train_batch_size ({}) does not match dataset actual batch size information ({}). Keeping {}.".format(train_batch_size, inferred_batch_size, inferred_batch_size),
                            category = RuntimeWarning
                        )
                        warnings.simplefilter("default", RuntimeWarning)
                self.train_batch_size = inferred_batch_size
                steps_per_epoch = int(cardinality)

                if(validation):
                    # If user provided both x and x_val as Datasets ready to use. They only have to provide n_train and n_val as extra parameters
                    if(isinstance(x_val, tf.data.Dataset)):
                        x_val_spec = x_val.element_spec[0] if isinstance(x_val.element_spec, tuple) else x_val.element_spec
                        is_batched_val = False
                        if( (len(x_val_spec.shape) > len(self.input_dim)) and (x_val_spec.shape[0] is None) ):
                            # Specifically, if when unbatching the Dataset, its shape continues to be different from the input_dim
                            if(x_val_spec.shape[1:] != self.input_dim):
                                raise ValueError(
                                    f"The provided validation dataset appears to be batched (shape: {x_val_spec.shape}), "
                                    f"but its underlying element shape {x_val_spec.shape[1:]} does not match "
                                    f"the expected `input_dim` of {self.input_dim}. "
                                    f"Please ensure your dataset has the correct dimensions or wasn't accidentally batched twice."
                                )
                            is_batched_val = True

                        # If validation data is also batched, require mandatory n_val value
                        self.val_batch_size = None
                        if(is_batched_val):
                            if n_val is None:
                                raise ValueError("When providing a separate batched tf.data.Dataset for validation (`x_val`), you must explicitly provide `n_val`.")
                            self.n_val = n_val
                            
                            # Infer validation batch size from the graph
                            sample_val_batch = next(iter(x_val))
                            if isinstance(sample_val_batch, tuple):
                                inferred_val_batch_size = int(sample_val_batch[0].shape[0])
                            else:
                                inferred_val_batch_size = int(sample_val_batch.shape[0])
                            self.val_batch_size = inferred_val_batch_size
                        else:
                            self.val_batch_size = val_batch_size
                            # Given x_val is not batched, n_val is simply the cardinality of the dataset
                            self.n_val = tf.data.experimental.cardinality(x_val).numpy()
                            
                        self.train_dataset = x
                        self.val_dataset = x_val

                    # If x_val is not a Dataset, yet the user provided a data_val, we assume it can be fully loaded into memory
                    elif(data_val is not None):
                        if(x_val is not None):
                            x_val = tf.cast(x_val, dtype = tf.float32)                            
                            # If input is a vector, transform it into a column
                            if(len(x_val.shape) == 1):
                                x_val = tf.reshape( x_val, shape = (len(x_val), 1) )
                        
                        # Cast all variables from data to tf.float32 and pass them to tf.arrays if neccessary
                        for i in range(len(data_val)):
                            data_val[i] = tf.cast(data_val[i], dtype = tf.float32)
                            if(len(data_val[i].shape) == 1):
                                data_val[i] = tf.reshape( data_val[i], shape = (len(data_val[i]), 1) )
                                
                        self.n_val = len(data_val[0])
                        
                        self.val_batch_size = val_batch_size
                        if(self.val_batch_size is None):
                            self.val_batch_size = self.n_val

                        val_tuple = (x_val, *data_val) if x_val is not None else tuple(data_val)
                        val_dataset = tf.data.Dataset.from_tensor_slices( val_tuple )
                        # Validation data never needs to be shuffled. Just batch and prefetch for speed.
                        val_dataset = val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)
                        
                        self.train_dataset = x
                        self.val_dataset = val_dataset
                        
                    # If data_val was never given, yet validation = True, we must sample the validation data from the training data
                    elif(val_prop is not None or n_val is not None):
                        if(self._detect_shuffle_in_dataset(x)):
                            raise ValueError(
                                "CRITICAL DATA LEAKAGE RISK: thetaflow detected a `.shuffle()` operation "
                                "in your tf.data.Dataset pipeline before validation splitting.\n"
                                "Because tf.data re-evaluates shuffles every epoch, applying `val_prop` "
                                "now will cause validation data to bleed into the training set.\n"
                                "Please, remove `.shuffle()` from your dataset pipeline. thetaflow will split "
                                "the deterministic data first, and apply shuffling safely afterward."
                            )

                        # --- THE UNBATCH TRICK ---
                        # We calculate exact samples instead of batch approximations
                        if(n_val is not None):
                            self.n_val = n_val
                        else:
                            self.n_val = int(self.n_train * val_prop)
                        self.n_train = self.n_train - self.n_val
                        
                        if(self.n_val == 0):
                            raise ValueError(f"Number of samples in validation set too low. Please, increase val_prop (or n_val directly).")

                        # Unbatch the dataset to sequence individual elements
                        unbatched_x = x.unbatch()
                        
                        # Precisely split by exact sample count
                        self.val_dataset = unbatched_x.take(self.n_val)
                        self.train_dataset = unbatched_x.skip(self.n_val)
                        
                        if(shuffle):
                            self.train_dataset = self.train_dataset.shuffle(buffer_size = buffer_size, reshuffle_each_iteration = True, seed = self.seed)

                        self.val_batch_size = val_batch_size
                        if(self.val_batch_size is None):
                            # If user did not specify a validation data batch size, consider the train batch size for safety in case of large datasets
                            self.val_batch_size = self.train_batch_size
                        
                        self.train_dataset = self.train_dataset.batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE)
                        self.val_dataset = self.val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)                        
                        
                        # Recalculate steps per epoch perfectly
                        steps_per_epoch = int( tf.math.ceil(self.n_train / self.train_batch_size) )
                    else:
                        raise ValueError("validation = True on a batched dataset requires either (`x_val`, `data_val`), `val_prop` or `n_val` to be specified.")
                        
                    self.n = self.n_train + self.n_val
                else:
                    self.train_dataset = x
                    self.val_dataset = None
                    self.val_batch_size = None
                    self.n_val = 0

            # If data x given was not batched by the user, we must explore all the possitiblities of such case
            # The cardinality above here will represent the total number of SAMPLES, not BATCHES
            else:
                self.n_train = int(cardinality)
                val_needs_batching = False

                if(validation):
                    # If user provided x_val as a tf.Dataset ready to use.
                    if(isinstance(x_val, tf.data.Dataset)):
                        x_val_spec = x_val.element_spec[0] if isinstance(x_val.element_spec, tuple) else x_val.element_spec
                        is_batched_val = False
                        if( (len(x_val_spec.shape) > len(self.input_dim)) and (x_val_spec.shape[0] is None) ):
                            # Specifically, if when unbatching the Dataset, its shape continues to be different from the input_dim
                            if(x_val_spec.shape[1:] != self.input_dim):
                                raise ValueError(
                                    f"The provided validation dataset appears to be batched (shape: {x_val_spec.shape}), "
                                    f"but its underlying element shape {x_val_spec.shape[1:]} does not match "
                                    f"the expected `input_dim` of {self.input_dim}. "
                                    f"Please ensure your dataset has the correct dimensions or wasn't accidentally batched twice."
                                )
                            is_batched_val = True

                        # If x_val is batched. We cannot get exact n_val without iteration.
                        if(is_batched_val):
                            if n_val is None:
                                raise ValueError("When providing a separate batched tf.data.Dataset for validation (`x_val`), you must explicitly provide `n_val`.")
                            self.n_val = n_val
                            
                            sample_val_batch = next(iter(x_val))
                            if isinstance(sample_val_batch, tuple):
                                inferred_val_batch_size = int(sample_val_batch[0].shape[0])
                            else:
                                inferred_val_batch_size = int(sample_val_batch.shape[0])
                            self.val_batch_size = inferred_val_batch_size
                            val_needs_batching = False
                            
                        # x_val is unbatched. We can just pull the cardinality.
                        else:
                            self.n_val = int(tf.data.experimental.cardinality(x_val).numpy())
                            val_needs_batching = True
                            self.val_batch_size = val_batch_size
                            
                        self.train_dataset = x
                        self.val_dataset = x_val

                    # User provided numpy arrays for validation
                    elif(data_val is not None):
                        if(x_val is not None):
                            x_val = tf.cast(x_val, dtype = tf.float32)
                            if(len(x_val.shape) == 1):
                                x_val = tf.reshape( x_val, shape = (len(x_val), 1) )
                        
                        for i in range(len(data_val)):
                            data_val[i] = tf.cast(data_val[i], dtype = tf.float32)
                            if(len(data_val[i].shape) == 1):
                                data_val[i] = tf.reshape( data_val[i], shape = (len(data_val[i]), 1) )
                                
                        self.n_val = len(data_val[0])
                        self.val_batch_size = val_batch_size

                        val_tuple = (x_val, *data_val) if x_val is not None else tuple(data_val)
                        val_dataset = tf.data.Dataset.from_tensor_slices( val_tuple )
                        
                        self.x_val = x_val
                        self.data_val = data_val

                        self.train_dataset = x
                        self.val_dataset = val_dataset
                        val_needs_batching = True
                        
                    # x is unbatched and split to obtain x_val
                    elif(val_prop is not None or n_val is not None):
                        if(self._detect_shuffle_in_dataset(x)):
                            raise ValueError(
                                "CRITICAL DATA LEAKAGE RISK: thetaflow detected a `.shuffle()` operation "
                                "in your tf.data.Dataset pipeline before validation splitting.\n"
                                "Because tf.data re-evaluates shuffles every epoch, applying `val_prop` "
                                "now will cause validation data to bleed into the training set.\n"
                                "Please, remove `.shuffle()` from your dataset pipeline. thetaflow will split "
                                "the deterministic data first, and apply shuffling safely afterward."
                            )

                        if(n_val is not None):
                            self.n_val = n_val
                        else:
                            self.n_val = int(self.n_train * val_prop)
                        self.n_train = self.n_train - self.n_val
                        
                        if(self.n_val == 0):
                            raise ValueError("Number of samples in validation set too low. Please, increase val_prop (or n_val directly).")

                        # No unbatching needed! Dataset is already unbatched.
                        self.val_dataset = x.take(self.n_val)
                        self.train_dataset = x.skip(self.n_val)
                        
                        if(shuffle):
                            self.train_dataset = self.train_dataset.shuffle(buffer_size = buffer_size, reshuffle_each_iteration = True, seed = self.seed)
                            
                        self.val_batch_size = val_batch_size
                        val_needs_batching = True
                        
                    else:
                        raise ValueError("validation = True on an unbatched dataset requires either (`x_val`, `data_val`), `val_prop` or `n_val` to be specified.")
                        
                    self.n = self.n_train + self.n_val
                else:
                    self.train_dataset = x
                    self.val_dataset = None
                    self.val_batch_size = None
                    self.n_val = 0
                    self.n = self.n_train

                # Apply batch execution both to train_dataset and val_dataset
                self.train_batch_size = train_batch_size
                if(self.train_batch_size is None):
                    self.train_batch_size = self.n_train
                
                self.train_dataset = self.train_dataset.batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE)
                
                if(self.val_dataset is not None):
                    if(self.val_batch_size is None):
                        self.val_batch_size = self.n_val
                        
                    if(val_needs_batching):
                        self.val_dataset = self.val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)
                
                # Calculate accumulation steps based on the dynamically created batches
                steps_per_epoch = int( tf.math.ceil(self.n_train / self.train_batch_size) )

    
            self.compile_model(optimizer_independent = optimizer_independent, optimizer_nn = optimizer_nn)
            
            # Enforce the accumulation steps for Full Batch Gradient Descent (or whatever the user defined)
            self.gradient_accumulation_steps = gradient_accumulation_steps
            if(self.gradient_accumulation_steps is None):
                self.gradient_accumulation_steps = steps_per_epoch
                
            self.configured = True
            return

        # data is optional only if the user provides all the information inside an already preprocessed (at least partially)
        # tf.data.Dataset object. If user want to pass raw data, they must provide the data information
        if(data is None):
            raise ValueError(
                "You must provide at least the `data` argument (a list of target tensors/arrays) "
                "when not using a pre-compiled tf.data.Dataset."
            )
        
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
        # data = tuple(data)
        
        # Save original processed data in object
        self.x = x
        self.data = data
        self.n = len(data[0]) # Sample size

        if(self.validation):
            # If all validation data was given
            if(data_val is not None):
                x_val = tf.cast(x_val, dtype = tf.float32)
                # If input is a vector, transform it into a column
                if(len(x_val.shape) == 1):
                    x_val = tf.reshape( x_val, shape = (len(x_val), 1) )
                
                # Cast all variables from data to tf.float32 and pass them to tf.arrays if neccessary
                for i in range(len(data_val)):
                    data_val[i] = tf.cast(data_val[i], dtype = tf.float32)
                    if(len(data_val[i].shape) == 1):
                        data_val[i] = tf.reshape( data_val[i], shape = (len(data_val[i]), 1) )
                
                self.x_val, self.data_val = x_val, data_val
                self.x_train, self.data_train = self.x, self.data
                self.n_train = self.n
                self.n_val = len(data_val[0])
            elif(val_prop is not None or n_val is not None):
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
                    
                # Dynamically set val_size based on what the user provided
                if(n_val is not None):
                    val_size = int(n_val)
                else:
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
                # For each variable in data, separate into train and validation
                for i in range(len(data)):
                    data_val.append( data_shuffled[i][:val_size] )
                    data_train.append( data_shuffled[i][val_size:] )

                self.data_train, self.data_val = data_train, data_val
            else:
                raise ValueError("validation = True on dataset requires either (`x_val`, `data_val`), `val_prop` or `n_val` to be specified.")
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

        train_tuple = (self.x_train, *self.data_train) if self.x_train is not None else tuple(self.data_train)
        train_dataset = tf.data.Dataset.from_tensor_slices( train_tuple )
        # Shuffles the dataset on every call
        if(shuffle):
            train_dataset = train_dataset.cache().shuffle(buffer_size = self.buffer_size, reshuffle_each_iteration = True, seed = self.seed)
        self.train_dataset = train_dataset.batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE)
        self.train_dataset = [ tf.data.Dataset.get_single_element(self.train_dataset) ]

        val_dataset = None
        if(validation):
            val_tuple = (self.x_val, *self.data_val) if self.x_val is not None else tuple(self.data_val)
            val_dataset = tf.data.Dataset.from_tensor_slices( val_tuple )
            # Validation data never needs to be shuffled. Just batch and prefetch for speed.
            val_dataset = val_dataset.batch(self.val_batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_dataset = val_dataset
        
        self.configured = True

    # --- NEW OPTIMIZED GRAPH COMPILATION ---
    
    # ---------------------------------------
    
    def get_covariances(self, jitter = 1.0e-6):
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
        # So we can obtain covariance matrices for all independent estimators (independent on data x)
        if(self.independent_pars_use):
            for i in range( len(self.independent_pars) ):
                vars_to_differentiate.append( self.trainable_variables[i] )
        
            # Number of weights associated to independent parameters
            num_independent_params = sum([tf.size(v).numpy() for v in vars_to_differentiate])

        num_nn_params = 0
        # So we can obtain confidence intervals for all outputs from the network
        if(self.neural_network_use):
            nn_vars = [ v for v in self.layers[-1].trainable_variables ]
            # Append the list of vars to differentiate with all weights on the last layer (linear predictor and bias weights)
            vars_to_differentiate += nn_vars
            # Number of weights associated to the neural network component
            num_nn_params = sum( tf.size(v) for v in nn_vars )
        
        # Total number of real weights we consider as statistical parameters
        num_params = num_independent_params + num_nn_params
        total_hessian = tf.zeros((num_params, num_params))

        # --- KERAS 3 C++ COMPATIBILITY FIX ---
        # tf.GradientTape.watch() strictly expects native C++ tf.Variable objects and rejects Keras 3 Variables.
        # We extract the underlying backend tensors to safely pass them into the AutoDiff graph.
        native_vars = []
        for v in vars_to_differentiate:
            if not isinstance(v, tf.Variable) and hasattr(v, '_value'):
                native_vars.append(v._value)
            else:
                native_vars.append(v)
        # -------------------------------------
        
        # ----------------------------------------------------------------------------------------------------------------------------------
        # PRE-FLIGHT HESSIAN CHECK
        # Runs in compiled mode on a single batch to check for trivial misspecification before compiling the massive Jacobian graph
        # ----------------------------------------------------------------------------------------------------------------------------------
        sample_batch = next(iter(self.train_dataset))
        
        if(self.neural_network_use):
            x_batch_check = sample_batch[0]
            batch_data_tuple_check = sample_batch[1:]
        else:
            x_batch_check = tf.zeros((1,)) # Safe dummy tensor for the compiled graph
            batch_data_tuple_check = sample_batch
            
        # Apply the same dimension guard
        batch_data_tuple_check = tuple(
            [tf.expand_dims(d, axis=-1) if len(d.shape) == 1 else d for d in batch_data_tuple_check]
        )

        @tf.function(reduce_retracing=True)
        def _compiled_preflight_step(x_batch, batch_data_tuple):
            if(self.neural_network_use):
                batch_full_data_check = (x_batch,) + batch_data_tuple
            else:
                # If no neural network is used, it still must pass the x predictor as None
                batch_full_data_check = (None,) + batch_data_tuple

            with tf.GradientTape(watch_accessed_variables = False) as tape1_check:
                tape1_check.watch(native_vars)
                
                if(self.neural_network_use):
                    nn_output_check = self(x_batch, training = False)
                else:
                    nn_output_check = None

                loss_value_check = self.loglikelihood_loss(self, nn_output = nn_output_check, data = batch_full_data_check)
                
            return tape1_check.gradient(loss_value_check, native_vars)

        # Run the compiled first-derivative check
        grads_check = _compiled_preflight_step(x_batch_check, batch_data_tuple_check)

        # ----------------------------------------------------------------------------------------------------------------------------------
        # This routine is designed to identify singular hessian problems and which parameters they may correspond to before all calculations
        # ----------------------------------------------------------------------------------------------------------------------------------
        # List of parameters that are not used in the loss function. That results in a non-invertible hessian matrix
        lack_independent_pars = []
        lack_nn_pars = []
        halt_hessian = False
        # Check if any grad value is None
        # If there is a None grad, it means the loss function does not depend on that parameter, and therefore, can not obtain covariance matrix
        for i, grad in enumerate(grads_check):
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
        
        # --- NEW OPTIMIZED GRAPH COMPILATION ---
        @tf.function(reduce_retracing=True)
        def _compiled_hessian_step(x_batch, batch_data_tuple):
            if(self.neural_network_use):
                batch_full_data_reconstructed = (x_batch,) + batch_data_tuple
            else:
                batch_full_data_reconstructed = (None,) + batch_data_tuple
                
            # Set watch_accessed_variables = False to stop TF from tracking the CNN convolutions and only track final layer
            with tf.GradientTape(persistent = True, watch_accessed_variables = False) as tape2:
                # Explicitly watch only statistical parameters
                tape2.watch(native_vars)
                with tf.GradientTape(watch_accessed_variables = False) as tape1:
                    tape1.watch(native_vars)
                    
                    # Ensured training = False for determinism and function smoothness
                    if(self.neural_network_use):
                        nn_output = self(x_batch, training = False)
                    else:
                        nn_output = None
                        
                    loss_value = self.loglikelihood_loss(self, nn_output = nn_output, data = batch_full_data_reconstructed)
                
                # First Derivative
                grads = tape1.gradient(loss_value, native_vars)
                
                # Flatten gradients to a single vector for easier Jacobian computation
                # Suppose we have k neurons on the last linear layer and d outputs. Then:
                # - The first group of k weights will correspond to the weights to the first output
                # - The second group of k weights will correspond to the weights to the second output
                grads_flat = tf.concat([tf.reshape(tf.transpose(g), [-1]) for g in grads], axis = 0)
                
            hessian_batch = tape2.jacobian(grads_flat, native_vars, experimental_use_pfor = False)
            
            # TF may return a tuple; convert to list so we can modify the None elements
            hessian_batch = list(hessian_batch)
            
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
            
            return hessian_final_batch
        # ---------------------------------------

        for batch_full_data in self.train_dataset:
            if(self.neural_network_use):
                x_batch = batch_full_data[0]
                batch_data_tuple = batch_full_data[1:] 
            else:
                x_batch = None
                batch_data_tuple = batch_full_data 
            
            # batch_full_data_reconstructed = (x_batch,) + batch_data_tuple
            
            # Dimension guard to prevent broadcasting errors in the dataset
            batch_data_tuple = tuple(
                [tf.expand_dims(d, axis=-1) if len(d.shape) == 1 else d for d in batch_data_tuple]
            )

            hessian_final_batch = _compiled_hessian_step(x_batch, batch_data_tuple)
            total_hessian += hessian_final_batch
            
        self.total_hessian = total_hessian
            
        try:
            # Try to invert with current jitter
            self.weights_covariance = tf.linalg.inv( self.total_hessian + jitter * tf.eye( num_params ) )
            self.hessian_jitter = jitter
            return
        except tf.errors.InvalidArgumentError:
            # Avoid code crash. Instead, prints a warning that the Hessian is nearly singular
            pass
                
        # If for all retries the hessian could not be inverted, return a warning that the covariance structure could not be obtained
        warnings.simplefilter("always", RuntimeWarning)
        warnings.warn(
            "Covariance matrix could not be computed because the log-likelihood Hessian is singular (or near singular).\n" + \
            "The model may not be identified.\n",
            category = RuntimeWarning,
        )
        warnings.simplefilter("default", RuntimeWarning)

    @tf.function(reduce_retracing=True)
    def _compiled_covariance_dataset(self, dataset):
        """
        Executes the covariance extraction over an entire dataset purely in C++.
        Uses tf.TensorArray to dynamically accumulate batches without memory explosions.
        """
        raw_cov_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        theta_cov_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        batch_index = tf.constant(0, dtype=tf.int32)

        for batch_data in dataset:
            if isinstance(batch_data, tuple):
                x_batch = batch_data[0]
            else:
                x_batch = batch_data

            # Compute the covariance structures for this specific batch
            raw_cov_batch, theta_cov_batch = self._compute_covariance_math(x_batch)

            raw_cov_array = raw_cov_array.write(batch_index, raw_cov_batch)
            theta_cov_array = theta_cov_array.write(batch_index, theta_cov_batch)
            batch_index += 1

        return raw_cov_array.concat(), theta_cov_array.concat()

    def _compute_covariance_math(self, x):
        """
        The core mathematical engine for calculating LLLA and Delta Method covariances.
        Strictly utilizes TensorFlow native operations for AutoGraph compatibility.
        """
        b = self.independent_output_size
        d = self.nn_output_size
        H_tilde = None
        Ib = None

        if(self.neural_network_use and x is not None):
            Y_2 = self.neural_network_call_nolast(self, x)

            # Replaced x.shape[0] with tf.shape(x)[0] for dynamic graph compatibility
            m = tf.shape(x)[0] 
            k = Y_2.shape[-1] 

            H_tilde = tf.einsum("ij, ...kl -> ...ijkl", tf.eye(d), Y_2[:,:,None])
            H_tilde = tf.reshape(H_tilde, (m, d, k*d))

            if(self.bias_use):
                Id = tf.tile(tf.eye(d)[None,:,:], (m, 1, 1))
                H_tilde = tf.concat([H_tilde, Id], axis=-1)

        if(self.independent_pars_use):
            Ib = tf.eye(b)
            if self.neural_network_use and x is not None:
                Ib = tf.reshape(Ib, (1, b, b))
                Ib = tf.tile(Ib, [m, 1, 1])

        if(self.independent_pars_use and H_tilde is not None):
            Ib_op = tf.linalg.LinearOperatorFullMatrix(Ib)
            H_tilde_op = tf.linalg.LinearOperatorFullMatrix(H_tilde)
            H = tf.linalg.LinearOperatorBlockDiag([Ib_op, H_tilde_op]).to_dense()

            # Replaced self.get_weights() with self.weights for C++ graph tracking
            independent_pars = tf.concat([tf.reshape(v, [-1]) for v in self.weights[:len(self.independent_pars)]], axis=0)
            independent_pars = tf.reshape(independent_pars, (1, self.independent_output_size))
            independent_pars = tf.tile(independent_pars, [m, 1])

            nn_pars = self.layers[-1](Y_2)

            raw_pars = tf.concat([independent_pars, nn_pars], axis=1)
            raw_cov = tf.einsum("...il, lj, ...ju -> ...iu", H, self.weights_covariance, tf.transpose(H, perm=[0,2,1]))

        elif(self.independent_pars_use and H_tilde is None):
            independent_pars = tf.concat([tf.reshape(v, [-1]) for v in self.weights[:len(self.independent_pars)]], axis=0)
            raw_pars = tf.reshape(independent_pars, (1, self.independent_output_size))
            raw_cov = self.weights_covariance[:self.independent_output_size, :self.independent_output_size]
            # Enforce batch dimension so the Delta einsum broadcasts perfectly in C++
            raw_cov = tf.expand_dims(raw_cov, axis=0)

        else:
            raw_pars = self.layers[-1](Y_2)
            raw_cov = tf.einsum("...il, lj, ...ju -> ...iu", H_tilde, self.weights_covariance, tf.transpose(H_tilde, perm=[0,2,1]))

        with tf.GradientTape() as tape:
            tape.watch(raw_pars)
            theta_pars = self.apply_link(raw_pars)

        J = tape.batch_jacobian(theta_pars, raw_pars)
        theta_cov = tf.einsum("...il, ...lj, ...ju -> ...iu", J, raw_cov, tf.transpose(J, perm=[0,2,1]))

        return raw_cov, theta_cov

    def covariance_output(self, x = None):
        """
            Given an input, x (raw arrays or tf.data.Dataset), obtains the asymptotic 
            covariance matrices for the model weights estimators.
            If x is not given, returns only the covariance matrix from the independent parameters.
        """
        # 1. Handle tf.data.Dataset (The Massive C++ Batch Route)
        if(isinstance(x, tf.data.Dataset)):
            return self._compiled_covariance_dataset(x)

        # 2. Handle missing x (Independent parameters only)
        if(x is None):
            if(not self.independent_pars_use):
                raise TypeError("Please, provide a list of input values, x.")
            elif(self.neural_network_use):
                warnings.simplefilter("always", UserWarning)
                warnings.warn(
                    "Model supports both neural network modeled parameters and independent parameters.\n" + \
                    "As a list of input values, x, was not provided, obtaining the covariances only for {}.".format(self.independent_pars),
                    category = UserWarning,
                )
                warnings.simplefilter("default", UserWarning)
            # Route to math with x=None
            return self._compute_covariance_math(None)

        # 3. Handle Raw Numpy Arrays / Tensors (Single massive batch)
        x_input = tf.cast(x, dtype=tf.float32)
        if(len(x_input.shape) == 1):
            x_input = tf.reshape(x_input, shape=(len(x_input), 1))

        return self._compute_covariance_math(x_input)
                

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
        
    # def covariance_output(self, x = None):
    #     """
    #         Given an input, x, obtain the asymptotic covariance matrices for the model weights estimators.
    #         If x is not given, return only the covariance matrix from the independent parameters, that are constant for every input.
    #     """
    #     if(x is not None):
    #         x = tf.cast(x, dtype = tf.float32)
    #         # If input is a vector, transform it into a column
    #         if(len(x.shape) == 1):
    #             x = tf.reshape( x, shape = (len(x), 1) )
        
    #     # Number of independent parameter values as outputs (may be different from len(self.independent_pars), if vectors are considered)
    #     b = self.independent_output_size
    #     # Number of parameters as outputs to the neural network (may be different from len(self.nn_pars), if vectors are considered)
    #     d = self.nn_output_size

    #     # I_d \otimes Y^{(-2)} matrix for neural network weights
    #     H_tilde = None
    #     # I_b identity matrix for independent components covariance
    #     Ib = None
        
    #     if(self.neural_network_use):
    #         # If there are no independent parameters and also no input x was given, raise an Error
    #         if(not self.independent_pars_use and x is None):
    #             raise TypeError("Please, provide a list of input values, x.")
    #         elif(x is None):
    #             warnings.simplefilter("always", UserWarning)
    #             warnings.warn(
    #                 "Model supports both neural network modeled parameters and independent parameters.\n" + \
    #                 "As a list of input values, x, was not provided, obtaining the covariances only for {}.".format(self.independent_pars),
    #                 category = UserWarning,
    #             )
    #             warnings.simplefilter("default", UserWarning)
    #         # If there are independent pars and x was given, simply obtain tilde{H} = I_d \otimes Y^{(-2)}
    #         else:
    #             x = tf.cast(x, dtype = tf.float32)
    #             # Let m be the number of entries in x
    #             # Y^{(-2)} dimension: (m, n_neurons_last_layer)
    #             Y_2 = self.neural_network_call_nolast(self, x)
                
    #             # Take the final layer weights and flatten then column-wise (each column stacked on top of the other) -> IMPORTANT! MUST MATCH HESSIAN CALCULATIONS!
    #             W = np.transpose( self.get_weights()[-1] ).flatten()
        
    #             m = x.shape[0] # Number of inputs
    #             k = Y_2.shape[-1] # Number of neurons on the penultimate layer
                
    #             # For each entry, x_i, we need to obtain I_d \otimes Y^{(-2)}(x_i)
    #             # To do that, we must consider the Einstein summation formula, since np.kron always suppose 2d matrices
    #             # \tilde{H} = I_d \otimes Y^{(-2)}(x_i)
    #             # Therefore, H must have dimensions (m, d, kd) as it represents the transformation from the weights (normally distributed)
    #             # to the neural network output, considering multiplication with the penultimate layer, Y_2
    #             H_tilde = tf.einsum("ij, ...kl -> ...ijkl", tf.eye(d), Y_2[:,:,None]) # (m, d, k, d, 1) tensor
    #             H_tilde = tf.reshape(H_tilde, (m, d, k*d))
                
    #             # If there is a bias on the last layer, concatenate a I_d matrix to H_tilde
    #             if(self.bias_use):
    #                 # Create an (m,d,d) tensor with I_d in each m index
    #                 Id = tf.tile(tf.eye(d)[None,:,:], (m, 1, 1))
    #                 H_tilde = tf.concat([H_tilde, Id], axis = -1)

    #     if(self.independent_pars_use):
    #         Ib = tf.eye(b)
    #         if(self.neural_network_use and x is not None):
    #             Ib = tf.reshape(Ib, (1, b, b))
    #             Ib = tf.tile(Ib, [m, 1, 1])
        
    #     # Ib exists and H_tilde exists
    #     if(self.independent_pars_use and H_tilde is not None):
    #         Ib = tf.linalg.LinearOperatorFullMatrix(Ib)
    #         H_tilde = tf.linalg.LinearOperatorFullMatrix(H_tilde)
    #         H = tf.linalg.LinearOperatorBlockDiag([Ib, H_tilde]).to_dense()
            
    #         # Cycle through all independent parameters and flatten their values into a single vector of real values
    #         independent_pars = tf.concat([ tf.reshape(v, [-1]) for v in self.get_weights()[:len(self.independent_pars)] ], axis = 0)
    #         independent_pars = tf.reshape(independent_pars, (1, self.independent_output_size))
    #         independent_pars = tf.tile(independent_pars, [m, 1])
    #         # Obtain the raw expression for each parameter modeled as a nn output
    #         nn_pars = self.layers[-1](Y_2)

    #         # Concatenate all parameters into a single vector. It will be used to get the gradients to the link functions
    #         raw_pars = tf.concat([independent_pars, nn_pars], axis = 1)
    #         raw_cov = tf.einsum("...il, lj, ...ju -> ...iu", H, self.weights_covariance, tf.transpose(H, perm = [0,2,1]))
    #     # Ib exists and H_tilde do not
    #     elif(self.independent_pars_use and H_tilde is None):
    #         # Cycle through all independent parameters and flatten their values into a single vector of real values
    #         independent_pars = tf.concat([ tf.reshape(v, [-1]) for v in self.get_weights()[:len(self.independent_pars)] ], axis = 0)
    #         raw_pars = tf.reshape(independent_pars, (1, self.independent_output_size))
    #         raw_cov = self.weights_covariance[:self.independent_output_size, :self.independent_output_size]
    #     # Ib do not exist and H_tilde does (consequently, x was given)
    #     else:
    #         raw_pars = self.layers[-1](Y_2)
    #         raw_cov = tf.einsum("...il, lj, ...ju -> ...iu", H_tilde, self.weights_covariance, tf.transpose(H_tilde, perm = [0,2,1]))
        
    #     # Compute the Jacobian J for link functions over each individual
    #     with tf.GradientTape() as tape:
    #         tape.watch(raw_pars)
    #         theta_pars = self.apply_link( raw_pars )

    #     # Delta method implementation for all parameters
    #     # (m, b+d, b+d)
    #     J = tape.batch_jacobian(theta_pars, raw_pars)
        
    #     # Obtain the covariance matrices for the transformed estimators according to the delta method
    #     theta_cov = tf.einsum("...il, ...lj, ...ju -> ...iu", J, raw_cov, tf.transpose(J, perm = [0,2,1]))
        
    #     return raw_cov, theta_cov

    # def summary(self, x = None, alpha = 0.05):
    #     pars_summary = {"index": [1]}
    #     nn_output = None
    #     if(x is not None):
    #         x = tf.cast(x, dtype = tf.float32)
    #         # If input is a vector, transform it into a column
    #         if(len(x.shape) == 1):
    #             x = tf.reshape( x, shape = (len(x), 1) )
    #         pars_summary = {"index": np.arange(len(x))+1}

    #         # Evaluate the neural network for all x values
    #         if(self.neural_network_use):
    #             nn_output = self(x, training = False)
            
    #     # Obtain the covariance matrices for all inputs, x
    #     raw_cov, theta_cov = self.covariance_output(x)
    #     z_norm = norm.ppf(1-alpha/2)
        
    #     for i in range(theta_cov.shape[1]):
    #         if(i < self.independent_output_size):
    #             # Take the name of the parameter in this respective position
    #             par_index_var = self.independent_index_to_vars[i][4:] # Remove the raw_ prefix
    #         else:
    #             j = i - self.independent_output_size
    #             par_index_var = self.nn_index_to_vars[j][4:]
    #             # nn_output = self(x, training = False)
        
    #         par_index_var_split = par_index_var.split("[")
    #         par_name = par_index_var_split[0]
    #         # If name matches the index_to_vars result, parameter is a single number (not a vector)
    #         if(par_name == par_index_var):
    #             par_index = 0
    #         else:
    #             par_index = int( par_index_var_split[-1].split("]")[0] )

    #         if(i < self.independent_output_size):
    #             raw_par_value = np.repeat( self.get_variable(par_name, nn_output, get_raw_value = True, force_true = True)[par_index], theta_cov.shape[0] )
    #         else:
    #             raw_par_value = self.get_variable(par_name, nn_output, get_raw_value = True, force_true = True)[:,par_index]
            
    #         par_value = self.parameters[par_name]["link"]( raw_par_value )
    #         # Raw parameter variance (Last-Layer Laplace Approximations - LLLA)
    #         raw_par_se = np.sqrt(raw_cov[:,i,i])
    #         # Final parameter variance (Delta method)
    #         par_se = np.sqrt(theta_cov[:,i,i])
    #         raw_par_lower = raw_par_value - z_norm * raw_par_se
    #         raw_par_upper = raw_par_value + z_norm * raw_par_se

    #         par_lower = self.parameters[par_name]["link"]( raw_par_lower )
    #         par_upper = self.parameters[par_name]["link"]( raw_par_upper )

    #         pars_summary[par_index_var] = par_value
    #         pars_summary[par_index_var + "_se"] = par_se
    #         pars_summary[par_index_var + "_lower"] = par_lower
    #         pars_summary[par_index_var + "_upper"] = par_upper
            
    #     return pd.DataFrame(pars_summary)

    def summary(self, x=None, alpha=0.05):
        """
        Generates a Pandas DataFrame containing the parameter estimates, standard errors, 
        and confidence intervals (via Delta Method). 
        Accepts raw tensors, numpy arrays, or tf.data.Dataset.
        """
        pars_summary = {}
        nn_output = None
        
        # 1. Handle tf.data.Dataset
        if isinstance(x, tf.data.Dataset):
            # Fire the compiled loops to process the massive dataset in C++
            if self.neural_network_use:
                nn_output = self._compiled_predict_dataset(x)
                
            raw_cov, theta_cov = self.covariance_output(x)
            n_samples = raw_cov.shape[0]
            pars_summary["index"] = np.arange(n_samples) + 1
            
        # 2. Handle missing x (Independent parameters only)
        elif x is None:
            raw_cov, theta_cov = self.covariance_output(None)
            n_samples = 1
            pars_summary["index"] = [1]
            
        # 3. Handle Raw Numpy Arrays / Tensors
        else:
            x_input = tf.cast(x, dtype=tf.float32)
            if len(x_input.shape) == 1:
                x_input = tf.reshape(x_input, shape=(len(x_input), 1))
                
            if self.neural_network_use:
                nn_output = self(x_input, training = False)
                
            raw_cov, theta_cov = self.covariance_output(x_input)
            n_samples = x_input.shape[0]
            pars_summary["index"] = np.arange(n_samples) + 1

        # Calculate the normal multiplier for the confidence bounds
        z_norm = norm.ppf(1 - alpha / 2)
        
        # 4. Vectorized parameter extraction and Delta method boundaries
        for i in range(theta_cov.shape[1]):
            # Identify the parameter name and its vector index
            if(i < self.independent_output_size):
                par_index_var = self.independent_index_to_vars[i][4:] # Remove raw_ prefix
            else:
                j = i - self.independent_output_size
                par_index_var = self.nn_index_to_vars[j][4:]
        
            par_index_var_split = par_index_var.split("[")
            par_name = par_index_var_split[0]
            
            if(par_name == par_index_var):
                par_index = 0
            else:
                par_index = int(par_index_var_split[-1].split("]")[0])

            # Extract the raw parameter (Before the link function)
            if(i < self.independent_output_size):
                # Independent parameters are constant, so we repeat them for the sample size
                raw_scalar = self.get_variable(par_name, nn_output = None, get_raw_value = True, force_true = True)[par_index]
                raw_par_value = tf.repeat(raw_scalar, n_samples)
            else:
                raw_par_value = self.get_variable(par_name, nn_output, get_raw_value = True, force_true = True)[:, par_index]
            
            # 1. Transform raw parameter to natural scale
            par_value = self.parameters[par_name]["link"](raw_par_value)
            
            # 2. Extract standard errors
            raw_par_se = tf.sqrt(raw_cov[:, i, i])
            par_se = tf.sqrt(theta_cov[:, i, i])
            
            # 3. Calculate Confidence Intervals in the unrestricted Raw (LLLA) space
            raw_par_lower = raw_par_value - z_norm * raw_par_se
            raw_par_upper = raw_par_value + z_norm * raw_par_se

            # 4. Apply inverse link to boundaries to map them back to the constrained Natural space
            par_lower = self.parameters[par_name]["link"](raw_par_lower)
            par_upper = self.parameters[par_name]["link"](raw_par_upper)

            # Extract to numpy arrays safely to prevent tf.Tensor memory leaks in Pandas
            pars_summary[par_index_var] = par_value.numpy() if hasattr(par_value, 'numpy') else par_value
            pars_summary[par_index_var + "_se"] = par_se.numpy() if hasattr(par_se, 'numpy') else par_se
            pars_summary[par_index_var + "_lower"] = par_lower.numpy() if hasattr(par_lower, 'numpy') else par_lower
            pars_summary[par_index_var + "_upper"] = par_upper.numpy() if hasattr(par_upper, 'numpy') else par_upper
            
        return pd.DataFrame(pars_summary)
    
    # def variable_function_covariance(self, fun, data, x = None):
    #     """
    #         Receives a single dimensional function of independent and nn parameters and return its corresponding variance for all observations queried
    #     """
    #     nn_output = None
    #     if(x is not None):
    #         x = tf.cast(x, dtype = tf.float32)
    #         # If input is a vector, transform it into a column
    #         if(len(x.shape) == 1):
    #             x = tf.reshape( x, shape = (len(x), 1) )

    #         # Obtain the network raw output
    #         nn_output = self(x)
        
    #     # Obtain the covariance matrices for all inputs, x
    #     raw_cov, theta_cov = self.covariance_output(x)
        
    #     # Initialize gradient tracker for parameters
    #     self._delta_tape = tf.GradientTape(persistent=True)
    #     self._tracked_theta_tensors = {}

    #     data = [x] + data
        
    #     # Run the user's function with _delta_tape as context
    #     with self._delta_tape:
    #         f_theta = fun(self, nn_output, data)
    #         # Ensures the output from fun is atleast two dimensional
    #         if(len(f_theta.shape) == 1):
    #             f_theta = tf.expand_dims(f_theta, axis = -1)
        
    #     ordered_theta_var_names = []
    #     ordered_theta_tensors = []
    #     # Tracks which parameters from the model were used in fun and which were not
    #     theta_used = {}
    #     # Goes through all variables in the order they appear in the covariance matrix
    #     for i in range(self.independent_output_size + self.nn_output_size):
    #         if(i < self.independent_output_size):
    #             # Get only the name of the variable
    #             var_name = self.independent_index_to_vars[i][4:].split("[")[0]
    #         else:
    #             var_name = self.nn_index_to_vars[i-self.independent_output_size][4:].split("[")[0]

    #         # In case the variable has shape > 1, we ensure it gets added only once in this list
    #         if(var_name not in theta_used):
    #             # If variable was used in fun, add it on its correct order to the list
    #             if(var_name in self._tracked_theta_tensors):
    #                 ordered_theta_tensors.append( self._tracked_theta_tensors[var_name] )
    #                 theta_used[ var_name ] = True
    #             # If given variable was not used in fun, just include a None (the Jacobian will have a column full of zeros)
    #             else:
    #                 # ordered_theta_tensors.append( None )
    #                 theta_used[ var_name ] = False
                    
    #     J_list = []
        
    #     used_counter = 0
    #     # Now that the Jacobians were obtained, we fix each one of them to match a proper matrix, J
    #     for i, parameter in enumerate(theta_used):
    #         # If parameter was used in fun
    #         if( theta_used[parameter] ):
    #             # If parameter is independent get the full jacobian, since it is the same for every observation
    #             if( parameter in self.independent_pars ):
    #                 parameter_jacobian = self._delta_tape.jacobian(f_theta, ordered_theta_tensors[ used_counter ], experimental_use_pfor = False)
    #                 # print("JACOBIAN", parameter, ":", parameter_jacobian)
    #                 # If first dimension of jacobian does not match data dimension and we know for sure there are input observations, x
    #                 if( (parameter_jacobian is not None) and (x is not None and self.neural_network_use) and (parameter_jacobian.shape[0] != x.shape[0]) ):
    #                     parameter_jacobian = tf.broadcast_to(parameter_jacobian, (x.shape[0], 1, 1))
                    
    #             # If parameter is output from the network, get the batch_jacobian instead
    #             elif( parameter in self.nn_pars ):
    #                 try:
    #                     parameter_jacobian = self._delta_tape.batch_jacobian(f_theta, ordered_theta_tensors[ used_counter ], experimental_use_pfor = False)
    #                 except ValueError:
    #                     parameter_jacobian = None
    #             # Increase the counter for the next used parameter in ordered_theta_tensors
    #             used_counter += 1
    #             # If get_variable was called, but parameter was not used, the jacobian still returns None
    #             if(parameter_jacobian is None):
    #                 if(x is not None):
    #                     jacobian_zeros = tf.zeros((x.shape[0], f_theta.shape[1], self.parameters[parameter]["shape"]))
    #                 else:
    #                     jacobian_zeros = tf.zeros((1, f_theta.shape[1], self.parameters[parameter]["shape"]))
    #                 parameter_jacobian = jacobian_zeros
                    
    #             J_list.append( parameter_jacobian )
    #         # If parameter was not used, we must impute its shape of zeros in the Jacobian matrix
    #         else:
    #             if(x is not None):
    #                 jacobian_zeros = tf.zeros((x.shape[0], f_theta.shape[1], self.parameters[parameter]["shape"]))
    #             else:
    #                 jacobian_zeros = tf.zeros((1, f_theta.shape[1], self.parameters[parameter]["shape"]))
    #             J_list.append( jacobian_zeros )

        
    #     # Concatenate all gradients into the jacobian matrix and virtually increase one dimension for following operation
    #     J = tf.concat(J_list, axis = -1)
        
    #     # 5. Clean up the state so it doesn't interfere with standard training
    #     self._delta_tape = None
    #     self._tracked_theta_tensors = None
        
    #     # Finally, with the Jacobian ordered and ready, the covariance matrix for function fun is J theta_cov J^T from the Delta method
    #     # This operation is simply expressed in terms of the Einstein summation convention given below
    #     fun_cov = tf.einsum("...il, ...lj, ...ju -> ...iu", J, theta_cov, tf.transpose(J, perm = [0,2,1]))
        
    #     return fun_cov
        
    @tf.function(reduce_retracing=True)
    def _compiled_variable_function_covariance_dataset(self, fun, dataset, extra_data):
        """
        Executes the function covariance extraction over an entire dataset purely in C++.
        Uses tf.TensorArray to dynamically accumulate the Delta method variances.
        """
        fun_cov_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        batch_index = tf.constant(0, dtype=tf.int32)

        for batch_data in dataset:
            # Unpack the batch safely
            if isinstance(batch_data, tuple):
                x_batch = batch_data[0]
                batch_extra = list(batch_data[1:])
            else:
                x_batch = batch_data
                batch_extra = []
            
            # Combine any dataset-yielded extra data with the python-level extra_data
            current_data = batch_extra + extra_data

            # Compute the covariance structures for this specific batch
            fun_cov_batch = self._compute_variable_function_covariance_math(fun, current_data, x = x_batch)
            fun_cov_array = fun_cov_array.write(batch_index, fun_cov_batch)
            batch_index += 1

        return fun_cov_array.concat()

    def _compute_variable_function_covariance_math(self, fun, data, x=None):
        """
        The core mathematical engine for calculating the Jacobian and Delta Method variance 
        for an arbitrary user-defined function.
        """
        nn_output = None
        if(x is not None):
            x = tf.cast(x, dtype=tf.float32)
            if len(x.shape) == 1:
                x = tf.reshape(x, shape=(tf.shape(x)[0], 1))
            nn_output = self(x, training = False)
            batch_size = tf.shape(x)[0] # Dynamic shape tracking for C++
        else:
            batch_size = 1
            
        # Route directly to the math engine to avoid wrapper overhead inside the graph
        if(x is not None):
            raw_cov, theta_cov = self._compute_covariance_math(x)
        else:
            raw_cov, theta_cov = self._compute_covariance_math(None)

        self._delta_tape = tf.GradientTape(persistent=True)
        self._tracked_theta_tensors = {}

        data_with_x = [x] + data
        
        with self._delta_tape:
            f_theta = fun(self, nn_output, data_with_x)
            if len(f_theta.shape) == 1:
                f_theta = tf.expand_dims(f_theta, axis=-1)

        ordered_theta_tensors = []
        theta_used = {}
        
        for i in range(self.independent_output_size + self.nn_output_size):
            if i < self.independent_output_size:
                var_name = self.independent_index_to_vars[i][4:].split("[")[0]
            else:
                var_name = self.nn_index_to_vars[i-self.independent_output_size][4:].split("[")[0]

            if var_name not in theta_used:
                if var_name in self._tracked_theta_tensors:
                    ordered_theta_tensors.append(self._tracked_theta_tensors[var_name])
                    theta_used[var_name] = True
                else:
                    theta_used[var_name] = False
                    
        J_list = []
        used_counter = 0
        
        for i, parameter in enumerate(theta_used):
            if theta_used[parameter]:
                if parameter in self.independent_pars:
                    parameter_jacobian = self._delta_tape.jacobian(f_theta, ordered_theta_tensors[used_counter], experimental_use_pfor=False)
                    
                    # Use AutoGraph compatible tf.shape() instead of .shape to prevent NoneType crashes on dynamic batches
                    if parameter_jacobian is not None and x is not None and self.neural_network_use:
                        if tf.shape(parameter_jacobian)[0] != batch_size:
                            parameter_jacobian = tf.broadcast_to(parameter_jacobian, (batch_size, tf.shape(parameter_jacobian)[1], tf.shape(parameter_jacobian)[2]))
                            
                elif parameter in self.nn_pars:
                    try:
                        parameter_jacobian = self._delta_tape.batch_jacobian(f_theta, ordered_theta_tensors[used_counter], experimental_use_pfor=False)
                    except ValueError:
                        parameter_jacobian = None
                        
                used_counter += 1
                
                if parameter_jacobian is None:
                    jacobian_zeros = tf.zeros((batch_size, tf.shape(f_theta)[1], self.parameters[parameter]["shape"]))
                    parameter_jacobian = jacobian_zeros
                    
                J_list.append(parameter_jacobian)
            else:
                jacobian_zeros = tf.zeros((batch_size, tf.shape(f_theta)[1], self.parameters[parameter]["shape"]))
                J_list.append(jacobian_zeros)

        J = tf.concat(J_list, axis=-1)
        
        self._delta_tape = None
        self._tracked_theta_tensors = None
        
        fun_cov = tf.einsum("...il, ...lj, ...ju -> ...iu", J, theta_cov, tf.transpose(J, perm=[0,2,1]))
        return fun_cov

    def variable_function_covariance(self, fun, data, x = None):
        """
        Receives a single dimensional function of independent and nn parameters and 
        returns its corresponding variance for all observations queried.
        Supports raw tensors, numpy arrays, or tf.data.Dataset.
        """
        # Handle tf.data.Dataset (The Massive C++ Batch Route)
        if(isinstance(x, tf.data.Dataset)):
            # Passes `data` as extra auxiliary data to be appended to each batch
            return self._compiled_variable_function_covariance_dataset(fun, dataset = x, extra_data = data)
            
        # Handle missing x (Independent parameters only)
        if(x is None):
            return self._compute_variable_function_covariance_math(fun, data, x = None)
            
        # Handle Raw Numpy Arrays / Tensors (Your Original Route)
        x_input = tf.cast(x, dtype = tf.float32)
        if(len(x_input.shape) == 1):
            x_input = tf.reshape(x_input, shape=(tf.shape(x_input)[0], 1))
            
        return self._compute_variable_function_covariance_math(fun, data, x=x_input)
    
            

