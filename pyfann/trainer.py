from enums import train_algorithm, error_func, stop_func

class Trainer():
    
    def __init__(self, neural_net, train_datas):
        self.neural_net = neural_net
        self.train_datas = train_datas
    
    def train(self):
        '''
        Train one epoch with a set of training data.

        Train one epoch with the training data stored in data.
        One epoch is where all of the training data is considered exactly once.

        This function returns the MSE error as it is calculated either before
        or during the actual training. This is not the actual MSE after the
        training epoch, but since calculating this will require to go through
        the entire training set once more, it is more than adequate to use
        this value during training.
        '''
        return self.neural_net.train_epoch(self.train_datas)

    def train_for(self, max_epochs, epochs_between_reports=0, disired_error=0.0):
        '''
        Trains on an entire dataset, for a period of time.

        This training uses the training algorithm chosen by
        set_training_algorithm, and the parameters set for these training
        algorithms.

        @param max_epochs:The maximum number of epochs the training should continue
        @param epochs_between_reports: The number of epochs between printing
            a status report to stdout. A value of zero means no reports should
            be printed
        @param desired_error: The desired MSE or bit fail, depending on
            which stop function is chosen by set_train_stop_function.
        '''
        return self.neural_net.train_on_data(self.train_datas, max_epochs,
                                         epochs_between_reports, disired_error)
    
    def test(self):
        '''
        Test a set of training data and calculates the MSE for the training data.

        This function updates the MSE and the bit fail values.
        '''
        return self.neural_net.test_data()

    _prop_funcs = [
    ('training_algorithm', None, None),
    ('learning_late', None, None),
    ('learning_momentum', None, None),
    ('train_error_function', None,None),
    ('train_stop_function', None,None),
    ('bit_fail_limit', None,None),
    ('quickprop_decay', None,None),
    ('rprop_increase_factor', None,None),
    ('rprop_decrease_factor', None,None),
    ('rprop_delta_min',None,None),
    ('rprop_delta_max',None,None),
    ('rprop_delta_zero',None,None),
    ('sarprop_weight_decay_shift',None,None),
    ('sarprop_step_error_threshold_factor',None,None),
    ('sarprop_step_error_shift',None,None),
    ('sarprop_temperature',None,None)
    ]
    
    def get_training_propaties(self):
        '''
        Get the training propaties including berow:
        '''
        propaties = {}
        for pf in self._prop_funcs:
            propaties[pf[0]] = pf[1]()

    def set_training_propaties(self, **kwargs):
        '''
        Set training propaties
        '''
        
        for pf in self._prop_funcs:
            if pf[0] in kwargs:
                pf[2](kwargs[pf[0]],)

    def get_training_algorithm(self):
        '''
        Return the training algorithm as described by train_algorithm.
        
        This training algorithm is used by train and associated methods.

        Note that this algorithm is also used during def cascadetrain_on_data,
        although only def RPROP and def QUICKPROP is allowed
        during cascade training.
        
        The default training algorithm is RPROP.
        '''
        return train_algorithm.RPROP
    
    def set_training_algorithm(self, algorithm):
        '''
        Set the training algorithm.
        '''
    
    def get_learning_rate(self):
        '''
        Return the learning rate.

        The learning rate is used to determine how aggressive training should be
        for some of the training algorithms (INCREMENTAL, BATCH, QUICKPROP).
    
        Do however note that it is not used in RPROP.
        
        The default learning rate is 0.7.
        '''
        return 0.7

    def set_learning_rate(self, lerning_rate):
        '''
        Set the learning rate.
        
        More info available in get_learning_rate
        '''

    def get_learning_momentum(self):
        '''
        Get the learning momentum.

        The learning momentum can be used to speed up NCREMENTAL training.
        A too high momentum will however not benefit training.
        Setting momentum to 0 will be the same as not using the momentum
        parameter.
        The recommended value of this parameter is between 0.0 and 1.0.

        The default momentum is 0.
        '''
        return 0.0

    def set_learning_momentum(self, learning_momentum):
        '''
        Set the learning momentum.

        More info available in get_learning_momentum
        '''

    def get_train_error_function(self):
        '''
        Returns the error function used during training.

        The error functions is described further in def errorfunc_enum
        
        The default error function is TANH
        '''
        return error_func.TANH

    def set_train_error_function(self, error_function):
        '''
        Set the error function used during training.
        '''

    def get_train_stop_function(self):
        '''
        Returns the the stop function used during training.

        The stop function is described further in def stop_func
        
        The default stop function is MSE
        '''
        return stop_func.MSE
    
    def set_train_stop_function(self, stop_function):
        '''
        Set the stop function used during training.
        '''

    def get_bit_fail_limit(self):
        '''
        Returns the bit fail limit used during training.

        The bit fail limit is used during training where the stop_func is set to
        BIT.
        
        The limit is the maximum accepted difference between the desired output
        and the actual output during training. Each output that diverges more
        than this limit is counted as an error bit. This difference is divided
        by two when dealing with symmetric activation functions, so that
        symmetric and not symmetric activation functions can use the same limit.
        
        The default bit fail limit is 0.35.
        '''
        return 0.35

    def set_bit_fail_limit(self, bit_fail_limit):
        '''
        Set the bit fail limit used during training.
        '''

    def get_quickprop_decay(self):
        '''
        The decay is a small negative valued number which is the factor that
        the weights should become smaller in each iteration during quickprop
        training.
        '''

    def set_quickprop_decay(self):
        '''
        Sets the quickprop decay factor.
        '''

    def get_quickprop_mu(self):
        '''
        The mu factor is used to increase and decrease the step-size during
        quickprop training.
        '''

    def set_quickprop_mu(self):
        '''
        Sets the quickprop mu factor.
        '''

    def get_rprop_increase_factor(self):
        '''
        The increase factor is a value larger than 1, which is used to
        increase the step-size during RPROP training.
        '''

    def set_rprop_increase_factor(self):
        '''
        The increase factor used during RPROP training.
        '''
    def get_rprop_decrease_factor(self):
        '''
        The decrease factor is a value smaller than 1, which is used to
        decrease the step-size during RPROP training.
        '''

    def set_rprop_decrease_factor(self):
        '''
        The decrease factor is a value smaller than 1, which is used to
        decrease the step-size during RPROP training.
        '''

    def get_rprop_delta_min(self):
        '''
        The minimum step-size is a small positive number determining how small
        the minimum step-size may be.
        '''

    def set_rprop_delta_min(self):
        '''
        The minimum step-size is a small positive number determining how small
        the minimum step-size may be.
        '''

    def get_rprop_delta_max(self):
        '''
        The maximum step-size is a positive number determining how large the
        maximum step-size may be.
        '''

    def set_rprop_delta_max(self):
        '''
        The maximum step-size is a positive number determining how large the
        maximum step-size may be.
        '''

    def get_rprop_delta_zero(self):
        '''
        The initial step-size is a positive number determining the initial step
        size.
        '''

    def set_rprop_delta_zero(self):
        '''
        The initial step-size is a positive number determining the initial step
        size.
        '''

    def get_sarprop_weight_decay_shift(self):
        '''
        The sarprop weight decay shift.
        '''

    def set_sarprop_weight_decay_shift(self):
        '''
        Set the sarprop weight decay shift.
        '''

    def get_sarprop_step_error_threshold_factor(self):
        '''
        The sarprop step error threshold factor.
        '''

    def set_sarprop_step_error_threshold_factor(self):
        '''
        Set the sarprop step error threshold factor.
        '''

    def get_sarprop_step_error_shift(self):
        '''
        The get sarprop step error shift.
        '''

    def set_sarprop_step_error_shift(self):
        '''
        Set the sarprop step error shift.
        '''

    def get_sarprop_temperature(self):
        '''
        The sarprop weight decay shift.
        '''

    def set_sarprop_temperature(self):
        '''
        Set the sarprop_temperature.
        '''
