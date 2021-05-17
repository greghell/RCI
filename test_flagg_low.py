####### low channels version ############
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from scipy import stats
import scipy.signal
import time

nAnts = 2048;
nChans = 8;
nPols = 2;

# spec = np.random.normal(0,5.,int(nChans*nAnts*(nAnts+1)/2*nPols*2));

# file = open("xcorr_in.bin", "wb");
# file.write(spec.astype(np.int32));
# file.close()

###### read files

datain = np.fromfile('xcorr_in.bin',dtype=np.int32);
dataout = np.fromfile('output.bin',dtype=np.float32);
datafl = np.fromfile('flags.bin',dtype=np.int32);
madmed = np.fromfile('madmed.bin',dtype=np.float32);

nChanLen = (nAnts*(nAnts-1)/2+nAnts) * nPols * 2;
tmp = np.zeros((nAnts*nChans*nPols));
allflags = np.zeros((nAnts*nChans*nPols));

for nc in range(nChans):
    for na in range(nAnts):
        for pol in range(nPols):
            idx = int((nAnts*(nAnts+1)/2) - ((nAnts-na)*((nAnts-na)+1)/2));
            tmp[int(na*(nChans*nPols)+nc*nPols+pol)] = datain[int(nc*nChanLen + idx*(nPols*2)+pol*2)];

def mad (arr):
    m = np.median(arr);
    return np.median(np.abs(arr-m));

def flagger (arr):
    lenarr = len(arr);
    fl = np.zeros((lenarr));
    ma = mad(arr);
    me = np.median(arr);
    #print(str(ma)+','+str(me));
    fl[np.where(np.abs(0.6745*(arr-me)/ma) > 6.0)[0]] = 1;
    return fl, ma, me;

def swap(p,q):
    t=p; 
    p=q; 
    q=t;

def mymedian(arr, cont):
    NCHANS = len(arr);
    tmp = np.zeros((NCHANS));
    if (cont == 0):
        for i in range(NCHANS):
            tmp[i] = arr[2*i];
    else:
        for i in range(NCHANS):
            tmp[i] = arr[i];

    for i in range(NCHANS-1):
        for j in range(NCHANS-i-1):
            if(tmp[j] > tmp[j+1]):
                swap(tmp[j],tmp[j+1]);

    if (NCHANS%2 == 0):
        return (tmp[(NCHANS/2-1)]+tmp[(NCHANS/2)])/2.;
    else:
        return tmp[(NCHANS+1)/2-1];




lenchan = nChans*nPols;
flagscpu = np.zeros((nAnts*lenchan));
pymadmed = np.zeros((nAnts*nPols*2));
for na in range(nAnts):
    dat = tmp[na*lenchan:(na+1)*lenchan];
    allflags[na*lenchan+np.arange(0,lenchan,2)], pymadmed[na*nPols*2], pymadmed[na*nPols*2+1] = flagger(dat[::2]);
    allflags[na*lenchan+np.arange(1,lenchan,2)], pymadmed[na*nPols*2 + 2], pymadmed[na*nPols*2+3]= flagger(dat[1::2]);
