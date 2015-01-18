from enum import Enum
import fann2.libfann

class net_type(Enum):
    '''
    Definition of network types used by fann_get_network_type
    '''
    
    '''
    Each layer only has connections to the next layer
    '''
    LAYER = fann2.libfann.LAYER

    '''
    Each layer has connections to all following layers
    '''
    SHORTCUT = fann2.libfann.SHORTCUT

class error_func(Enum):
    '''
    Error function used during training.
    '''

    '''
    Standard linear error function.
    '''
    LINEAR = fann2.libfann.ERRORFUNC_LINEAR

    '''
    Tanh error function, usually better but can require a lower learning rate.
    
    This error function agressively targets outputs that differ much from the
    desired, while not targetting outputs that only differ a little that much.
    This activation function is not recommended for cascade training and
    incremental training.
    '''
    TANH = fann2.libfann.ERRORFUNC_TANH

class stop_func(Enum):
    '''
    Stop criteria used during training.
    '''
    
    '''
    Stop criteria is Mean Square Error (MSE) value.
    '''
    MSE = fann2.libfann.STOPFUNC_MSE
    
    
    '''
    Stop criteria is number of bits that fail.
    
    The number of bits; means the number of output neurons which differ more than
    the bit fail limit (see fann_get_bit_fail_limit, fann_set_bit_fail_limit).
    The bits are counted in all of the training data, so this number can be higher
    than the number of training data.
    '''
    BIT = fann2.libfann.STOPFUNC_BIT

class activation_func(Enum):
    '''
    The activation functions used for the neurons during training.
    
    The activation functions can either be defined for a group of neurons by
    fann_set_activation_function_hidden and fann_set_activation_function_output
    or it can be defined for a single neuron by fann_set_activation_function.
    
    The steepness of an activation function is defined in the same way by
    fann_set_activation_steepness_hidden, fann_set_activation_steepness_output
    and fann_set_activation_steepness.
    '''
    
    LINEAR = fann2.libfann.LINEAR
    THRESHOLD = fann2.libfann.THRESHOLD
    THRESHOLD_SYMMETRIC = fann2.libfann.THRESHOLD_SYMMETRIC
    SIGMOID = fann2.libfann.SIGMOID
    SIGMOID_STEPWISE = fann2.libfann.SIGMOID_STEPWISE
    SIGMOID_SYMMETRIC = fann2.libfann.SIGMOID_SYMMETRIC
    SIGMOID_SYMMETRIC_STEPWISE = fann2.libfann.SIGMOID_SYMMETRIC_STEPWISE
    GAUSSIAN = fann2.libfann.GAUSSIAN
    GAUSSIAN_SYMMETRIC = fann2.libfann.GAUSSIAN_SYMMETRIC
    GAUSSIAN_STEPWISE = fann2.libfann.GAUSSIAN_STEPWISE
    ELLIOT = fann2.libfann.ELLIOT
    ELLIOT_SYMMETRIC = fann2.libfann.ELLIOT_SYMMETRIC
    LINEAR_PIECE = fann2.libfann.LINEAR_PIECE
    LINEAR_PIECE_SYMMETRIC = fann2.libfann.LINEAR_PIECE_SYMMETRIC
    SIN_SYMMETRIC = fann2.libfann.SIN_SYMMETRIC
    COS_SYMMETRIC = fann2.libfann.COS_SYMMETRIC

class train_algorithm(Enum):
    '''
    The Training algorithms used when training on struct fann_train_data with
    functions like fann_train_on_data or fann_train_on_file.
    
    The incremental training looks alters the weights after each time
    it is presented an input pattern, while batch only alters the weights 
    once after it has been presented to all the patterns.
    '''
    
    '''
    Standard backpropagation algorithm, where the weights are updated after each
    training pattern.
    
    This means that the weights are updated many times during a single epoch.
    For this reason some problems, will train very fast with this algorithm,
    while other more advanced problems will not train very well.
    '''
    INCREMENTAL = fann2.libfann.TRAIN_INCREMENTAL
    
    '''
    Standard backpropagation algorithm, where the weights are updated after
    calculating the mean square error for the whole training set.
    
    This means that the weights are only updated once during a epoch.
    For this reason some problems, will train slower with this algorithm.
    But since the mean square error is calculated more correctly than in
    incremental training, some problems will reach a better solutions with this
    algorithm.
    '''
    BATCH = fann2.libfann.TRAIN_BATCH
    
    '''
    A more advanced batch training algorithm which achieves good results for many
    problems.
    
    The RPROP training algorithm is adaptive, and does therefore not use the
    learning_rate.
    Some other parameters can however be set to change the way the RPROP algorithm
    works, but it is only recommended for users with insight in how the RPROP
    training algorithm works.  The RPROP training algorithm is described by
    [Riedmiller and Braun, 1993], but the actual learning algorithm used here is
    the iRPROP- training algorithm which is described by [Igel and Husken, 2000]
    which is an variety of the standard RPROP training algorithm.
    
    '''
    RPROP = fann2.libfann.TRAIN_RPROP
    
    '''
    A more advanced batch training algorithm which achieves good results for many
    problems.
    
    The quickprop training algorithm uses the learning_rate parameter along with
    other more advanced parameters, but it is only recommended to change these
    advanced parameters, for users with insight in how the quickprop training
    algorithm works.  The quickprop training algorithm is described by
    [Fahlman, 1988].
    '''
    QUICKPROP = fann2.libfann.TRAIN_QUICKPROP
    
    SARPROP = fann2.libfann.TRAIN_SARPROP