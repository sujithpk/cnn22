import numpy as np
import csv
from scipy.fftpack import fft
from google.cloud import storage

class DataSet(object):
    def __init__(self, acdata, labels):
        assert acdata.shape[0] == labels.shape[0], (
                "acdata.shape: %s labels.shape: %s" % (acdata.shape,
                                                       labels.shape))
        assert acdata.shape[3] == 1
        acdata = acdata.reshape(acdata.shape[0],
                                    acdata.shape[1] * acdata.shape[2])
        acdata = acdata.astype(np.float32)
        self._num_examples = acdata.shape[0]
        self._acdata = acdata
        self._labels = labels
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def acdata(self):
        return self._acdata

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_done += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._acdata = self._acdata[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._acdata[start:end], self._labels[start:end]

def bubble_sort():
    with open("data.csv") as file:
        reader=csv.reader(file)
        da=list(reader) # d is list format of data.csv
    
    data=np.array(da)
    shp=data.shape
    rw=shp[0]
    cl=shp[1]
    print('  ..no of data points=',rw)
    
    #converting to float values
    global data_sorted
    data_sorted=np.zeros(shp)
    for i in range(rw):
        for j in range(cl):
            data_sorted[i][j]=float(data[i][j])

    #bubble sort based on time
    for i in range(rw):
    #Last i elements are already in place   
        for j in range(rw-1-i): 
            if (data_sorted[j][0]+0.0 > data_sorted[j+1][0]+0.0):
                tmp=data_sorted[j][0]
                data_sorted[j][0]=data_sorted[j+1][0]
                data_sorted[j+1][0] =tmp

def getData(colno):
    # colno = column to be read
    ac_sig = np.zeros(3139)
    for i in range(3139):
        ac_sig[i] = float(data_sorted[i][colno+1]) / 2.38
    
    #sliding window 5 long, step size 2
    ac_smpld = np.zeros(1568)

    for m in range(1568):
        adn = 0.0
        for n in range(5):
            adn = adn + float(ac_sig[m*2 + n]) # sum 
            ac_smpld[m] = adn / 5 #average

    han_wind=np.hanning(1568)
    ac_han=np.multiply(ac_smpld,han_wind)

    #get fft of ac_han
    ac_fft = abs(fft(ac_han))
    ac_data = np.zeros(784) # final result : the training data

    #finding rms of bands
    for i in range(784):
        sq_sum = 0.0
        for j in range(2):
            sq_sum = sq_sum + ac_fft[i*2 + j] * ac_fft[i*2 + j] #squared sum 
            sq_sum = sq_sum /2  #mean of squared sum
            ac_data[i] = np.sqrt(sq_sum) #root of mean of squared sum = rms
    return ac_data

def read_inp(n_pred,num_classes,one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    print('..Sorting..')
    bubble_sort()
    print('..Reading inputs and predicting..')

    pred_acdata = np.zeros((n_pred,28,28,1))
    for i in range(n_pred):
        count=0
        acdat=getData(i)
        for j in range(28):
            for k in range(28):
                pred_acdata[i,j,k,0]=acdat[count] 
                count+=1
  
    ext_lab = np.zeros((n_pred,))
    data_sets.pred = DataSet(pred_acdata, ext_lab)
    return data_sets
