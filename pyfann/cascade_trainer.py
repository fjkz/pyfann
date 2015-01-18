from trainer import Trainer

class CascadeTrainer(Trainer):
    '''
    Cascade training differs from ordinary training in the sense that it starts
    with an empty neural network and then adds neurons one by one, while it
    trains the neural network.  The main benefit of this approach, is that you
    do not have to guess the number of hidden layers and neurons prior to
    training, but cascade training have also proved better at solving some
    problems.

    The basic idea of cascade training is that a number of candidate neurons
    are trained separate from the real network, then the most promissing of
    these candidate neurons is inserted into the neural network. Then the
    output connections are trained and new candidate neurons is prepared. The
    candidate neurons are created as shorcut connected neurons in a new hidden
    layer, which means that the final neural network will consist of a number
    of hidden layers with one shorcut connected neuron in each.
    '''


    def __init__(self, params):
        '''
        Constructor
        '''

    def get_cascade_output_change_fraction(self):
        """
        """

    def set_cascade_output_change_fraction(self):
        """
        """

    def get_cascade_output_stagnation_epochs(self):
        """
        """

    def set_cascade_output_stagnation_epochs(self):
        """
        """

    def get_cascade_candidate_change_fraction(self):
        """
        """

    def set_cascade_candidate_change_fraction(self):
        """
        """

    def get_cascade_candidate_stagnation_epochs(self):
        """
        """

    def set_cascade_candidate_stagnation_epochs(self):
        """
        """

    def get_cascade_weight_multiplier(self):
        """
        """

    def set_cascade_weight_multiplier(self):
        """
        """

    def get_cascade_candidate_limit(self):
        """
        """

    def set_cascade_candidate_limit(self):
        """
        """

    def get_cascade_max_out_epochs(self):
        """
        """

    def set_cascade_max_out_epochs(self):
        """
        """

    def get_cascade_min_out_epochs(self):
        """
        """

    def set_cascade_min_out_epochs(self):
        """
        """

    def get_cascade_max_cand_epochs(self):
        """
        """

    def set_cascade_max_cand_epochs(self):
        """
        """

    def get_cascade_min_cand_epochs(self):
        """
        """

    def set_cascade_min_cand_epochs(self):
        """
        """

    def get_cascade_num_candidates(self):
        """
        """

    def get_cascade_activation_functions_count(self):
        """
        """

    def get_cascade_activation_functions(self):
        """
        """

    def set_cascade_activation_functions(self):
        """
        """

    def get_cascade_activation_steepnesses_count(self):
        """
        """

    def get_cascade_activation_steepnesses(self):
        """
        """

    def set_cascade_activation_steepnesses(self):
        """
        """

    def get_cascade_num_candidate_groups(self):
        """
        """

    def set_cascade_num_candidate_groups(self):
        """
        """

