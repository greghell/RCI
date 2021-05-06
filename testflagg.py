import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib

## simulate bandpass 2000 antennas

nAnts = 2000;
nChans = 1000;

bline = 100.*np.ones((nChans));
bline[:200] = np.linspace(0,100,200);
bline[-200:] = np.linspace(100,0,200);
spec = np.matlib.repmat(bline,1,nAnts);
spec = spec[0];
spec = spec + np.random.normal(0,5.,len(spec));

for k in range(nAnts):
    ch = np.random.randint(nChans);
    spec[k*nChans + ch] = spec[k*nChans + ch] + 20.*np.random.chisquare(2);

file = open("noise.bin", "wb");
file.write(spec.astype(np.float32));
file.close()



###### read files

datain = np.fromfile('noise.bin',dtype=np.float32);
dataout = np.fromfile('output.bin',dtype=np.float32);
datafl = np.fromfile('flags.bin',dtype=np.float32);
databl = np.fromfile('bline.bin',dtype=np.float32);

fl = np.where(datafl==1)[0];

plt.figure();
plt.plot(datain,label='data');
plt.plot(databl,label='baseline');
plt.xlim([875000,905000]);
plt.legend();
plt.grid();
plt.show();

plt.figure();
plt.plot(datain,label='data');
plt.plot(fl,datain[fl],'*r',label='flags');
plt.xlim([875000,905000]);
plt.legend();
plt.grid();
plt.show();
