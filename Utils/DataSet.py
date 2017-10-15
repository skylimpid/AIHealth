class DataSet(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    __len__, that provides the size of the dataset,
    __getitem__, supporting integer indexing in range from 0 to len(self) exclusive.
    getNextBatch, to get datas based on the input batch_size
    hasNextBatch, to check whether we have iterator all the data
    reset, to reset the index of data so we can fetch the data from the beginning
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def getNextBatch(self, batch_size):
        raise NotImplementedError

    def hasNextBatch(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError