import fann2.libfann

class TrainData(object):
    

    def __init__(self):
        self._training_data = None
        self._addDone = True
    
    def __del__(self):
        self._training_data.destroy_train()
    
    def add(self, input_data, ouput_data):
        self._addDone = False

    def _add_commit(self):
        self._addDone = True

    def num_input(self):
        '''
        Returns the number of inputs in each of the training patterns.
        '''
        return self._training_data.num_input()

    def num_output(self):
        '''
        Returns the number of outputs in each of the training patterns
        '''
        return self._training_data.num_output()

    def save(self, filename):
        '''
        Save the training structure to a file, with the format
        as specified in read_train_from_file
        '''
        if self._training_data.save_train(filename) < 0:
            raise IOError("Failed to save.")

def read_train_data_from_file(filename):
    '''
    Reads a file that stores training data.
    
    The file must be formatted like:

    num_train_data num_input num_output
    inputdata seperated by space
    outputdata seperated by space
    ...
    inputdata seperated by space
    outputdata seperated by space
    '''
    training_data = fann2.libfann.training_data()
    training_data.read_train_from_file(filename)
    train_data = TrainData()
    train_data._training_data = training_data
    return train_data
