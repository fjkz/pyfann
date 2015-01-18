import fann2.libfann
from enums import net_type, activation_func

class NeuralNet(object):
    '''
    The fast artificial neural network(fann) structure.
    '''

    # TODO need this?
    def __init__(self):
        '''
        Constructor
        
        Do not call this directly
        '''
        self._fann = None

    def __del__(self):
        '''
        Destroys the entire network and properly freeing all the associated
        memmory.
        '''
        self._fann.destroy()

    def randomize_weights(self, min_weight=-0.1, max_weight=0.1):
        '''
        Give each connection a random weight between min_weight and max_weight

        From the beginning the weights are random between -0.1 and 0.1.
        '''
        self._fann.randomize_weights(min_weight, max_weight)

    def init_weights(self, train_data):
        '''
        Initialize the weights using Widrow + Nguyen’s algorithm.

        This function behaves similarly to fann_randomize_weights.
        It will use the algorithm developed by Derrick Nguyen and
        Bernard Widrow to set the weights in such a way as to speed
        up training.
        This technique is not always successful, and in some cases
        can be less efficient than a purely random initialization.

        The algorithm requires access to the range of the input data
        (ie, largest and smallest input), and therefore accepts
        a second argument, data, which is the training data
        that will be used to train the network.
        '''
    
        self._fann.init_weights(train_data) # TODO
    
    def run(self, input_data):
        '''
        Will run input through the neural network, returning an array of
        outputs, the number of which being equal to the number of neurons
        in the output layer.
        '''
        return self._fann.run(input_data)

    def save(self, filename):
        '''
        Save the entire network to a configuration file.

        The configuration file contains all information about the neural
        network and enables create_from_file to create an exact copy of
        the neural network and all of the parameters associated with the
        neural network.

        These two parameters (set_callback, set_error_log) are NOT saved
        to the file because they cannot safely be ported to a different
        location.  Also temporary parameters generated during training
        like get_MSE is not saved.
        '''
        # TODO

    def copy(self):
        '''
        Creates a copy of a fann structure.

        Data in the user data fann_set_user_data is not copied,
        but the user data pointer is copied.
        '''
        return NeuralNet(self._fann.copy())

    def get_num_input(self):
        '''
        Get the number of input neurons.
        '''
        return self._fann.get_num_input()

    def get_num_output(self):
        '''
        Get the number of output neurons.
        '''
        return self._fann.get_num_output()

    def get_total_neurons(self):
        '''
        Get the total number of neurons in the entire network.

        This number does also include the bias neurons,
        so a 2-4-2 network has 2+4+2 +2(bias) = 10 neurons.
        '''
        return self._fann.get_total_neurons()

    def get_total_connections(self):
        '''
        Get the total number of connections in the entire network.
        '''
        return self._fann.get_total_connections()

    def get_network_type(self):
        '''
        Get the type of neural network it was created as.
        '''
        return net_type(self._fann.get_network_type())

    def get_activation_function(self, layer, neuron):
        '''
        Get the activation function for neuron number neuron in layer number
        layer, counting the input layer as layer 0.

        It is not possible to get activation functions for the neurons in the
        input layer.
        
        Information about the individual activation functions is available at
        activationfunc
        
        @return: 
            The activation function for the neuron
            or -1 if the neuron is not defined in the neural network.
        '''
        ret = self._fann.get_activation_function(layer, neuron)
        if ret < 0:
            raise IndexError('the neuron (' + str(layer) + ', ' + str(neuron) +
                             'is not defined in the neural network')
        return activation_func(ret)

    def set_activation_function(self, activation_function, layer, neuron=-1):
        '''
        Set the activation function for neuron number neuron in layer number
        layer, counting the input layer as layer 0.

        It is not possible to set activation functions for the neurons in the
        input layer.
        
        When choosing an activation function it is important to note that the
        activation functions have different range.
        SIGMOID is e.g. in the 0 - 1 range while SIGMOID_SYMMETRIC
        is in the -1 - 1 range and LINEAR is unbound.
        
        Information about the individual activation functions is available at
        fann_activationfunc_enum.
        
        The default activation function is SIGMOID_STEPWISE.
        '''
        if neuron < 0:
            self._fann.set_activation_function(activation_function.value())
            return
        # TODO
    
    def get_activation_steepness(self, layer, neuron):
        '''
        Get the activation steepness for neuron number neuron in layer number
        layer, counting the input layer as layer 0.

        It is not possible to get activation steepness for the neurons in the
        input layer.

        The steepness of an activation function says something about how fast
        the activation function goes from the minimum to the maximum.
        A high value for the activation function will also give a more
        agressive training.
        
        When training neural networks where the output values should be at the
        extremes (usually 0 and 1, depending on the activation function),
        a steep activation function can be used (e.g.  1.0).
        
        The default activation steepness is 0.5.
        '''
        return self._fann.get_activation_steepness()
    
    def set_activation_steepness(self, steepness, layer, neuron=-1):
        '''
        Set the activation steepness for neuron number neuron in layer number
        layer, counting the input layer as layer 0.
    
        It is not possible to set activation steepness for the neurons in the
        input layer.
        
        The steepness of an activation function says something about how fast
        the activation function goes from the minimum to the maximum.
        A high value for the activation function will also give a more
        agressive training.
        
        When training neural networks where the output values should be at the
        extremes (usually 0 and 1, depending on the activation function),
        a steep activation function can be used (e.g.  1.0).
        
        The default activation steepness is 0.5.
        '''
        self._fann.get_activation_steepness(steepness, layer)

def create_standard_network(nums_neurons):
    '''
    Creates a standard fully connected backpropagation neural network.
    '''
    fann = fann2.libfann.neural_net()
    fann.create_standard_array(nums_neurons)
    return NeuralNet(fann)

def create_sparse_network(connection_rate, nums_neurons):
    '''
    Creates a standard backpropagation neural network,
    which is not fully connected.
    '''
    fann = fann2.libfann.neural_net()
    fann.create_sparse_array(connection_rate, nums_neurons)
    nn = NeuralNet()
    nn._fann = fann
    return nn

def create_shortcut_network(nums_neurons):
    '''
    Creates a standard backpropagation neural network,
    which is not fully connected and which also has shortcut connections.

    Shortcut connections are connections that skip layers.
    A fully connected network with shortcut connections, is a network
    where all neurons are connected to all neurons in later layers.
    Including direct connections from the input layer to the output layer.
    '''
    fann = fann2.libfann.neural_net()
    fann.create_standard_array(nums_neurons)
    nn = NeuralNet()
    nn._fann = fann
    return nn

def create_network_from_file(filename):
    '''
    Constructs a backpropagation neural network from a configuration file,
    which have been saved.
    '''
    fann = fann2.libfann.neural_net()
    fann.create_from_file(filename)
    nn = NeuralNet()
    nn._fann = fann
    return nn
