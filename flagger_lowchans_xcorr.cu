/*
This flagger processes all cross correlations, no autocorrelations
it is assumed that autocorrelations, accessing all channels, can be flagged earlier in the signal flow (e.g. FPGAs)

file = open("noise.bin", "wb");
nChan = 8.;
nPols = 2.;
nAnts = 2048;
file.write(np.random.normal(0.,10.,int(nChan*nAnts*(nAnts-1)/2.*nPols*2.)).astype(np.int32));
file.close()
*/

#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#define NCHANS 8	// # of channels -- assume no more than 2048 channels for now, see blinest call in main{}
#define NANTS 2048	// # of antennas
#define NPOLS 2	// # of pols

/* work in progress */
/*
__global__
void reorder(int32_t *a, int32_t *b) {	// reorders array with baselines and pols varying slower than chans
	int ch, bl, po;
	int nNumBL = (NANTS*(NANTS-1)/2);
	for (ch=0; ch<NCHANS; ch++){
		for (bl=0; bl<nNumBL; bl++){
			for (po=0; po<NPOLS; po++){
				b[bl*NCHANS*NPOLS + po*NCHANS]
			}
		}
	}
}
*/

__device__
void swap(float *p, float *q) {
   float t;
   
   t=*p; 
   *p=*q; 
   *q=t;
}

/*median value for NCHANS-long array*/
__device__
float medchans(float *a, int cont) {	// cont controls wether the array is contiguous or interleaved
	int i,j;
	float tmp[NCHANS] = {0};
	if (cont == 0){
		for (i = 0; i < NCHANS; i++)
			tmp[i] = a[2*i];	// polarizations are interleaved
	} else {
		for (i = 0; i < NCHANS; i++)
			tmp[i] = a[i];	// samples are not interleaved
	}
	
	for(i = 0; i < NCHANS-1; i++) {
		for(j = 0; j < NCHANS-i-1; j++) {
			if(tmp[j] > tmp[j+1])
				swap(&tmp[j],&tmp[j+1]);
		}
	}
	/*CHECK IF NCHANS IS EVEN ; IF EVEN: TAKE AVERAGE BETWEEN TWO MIDDLE VALUES*/
	
	if (NCHANS%2 == 0){
		return (tmp[(int)(NCHANS/2-1)]+tmp[(int)(NCHANS/2)])/2.;
	} else {
		return tmp[(int)((NCHANS+1)/2-1)];
	}
}

/*computes MAD for one spectrum*/
__device__
void mad(float *a, float *madval, float *medval) {
	int i;
	float dev[NCHANS];
	float me = medchans(a,1);	// interleaved array
	*medval = me;
	for (i = 0; i < NCHANS; i++)
		dev[i] = abs(a[i]-me);	// polarizations are interleaved
	*madval = medchans(dev,1);		// contiguous array
}

__global__
void flagg(int32_t *d_data, uint8_t *d_flags, float dThres) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nChanLen = NCHANS*2;
	if (tid < 512){
		float sub[NCHANS];
		int i;
		for (i=0; i<NCHANS; i++)
			sub[i] = sqrtf((float)(d_data[(bid*512+tid)*nChanLen+i*2]*d_data[(bid*512+tid)*nChanLen+i*2]+d_data[(bid*512+tid)*nChanLen+i*2+1]*d_data[(bid*512+tid)*nChanLen+i*2+1]));
		float madval = 0;
		float medval = 0;
		float M;
		mad(sub, &madval, &medval);
//		printf("ran into thread");
		for(i = 0; i < NCHANS; i++){
			M = 0.6745*(sub[i]-medval) / madval;
			if (abs(M) > dThres){
				d_flags[(bid*512+tid)*nChanLen+i*2] = 1;
				d_flags[(bid*512+tid)*nChanLen+i*2+1] = 1;
			}
		}
	}
}

void usage()
{
  fprintf (stdout,
	   "flagg_low [options]\n"
	   " -t threshold   flagger threshold in # of sigma [default : 3.5]\n"
	   " -h print usage\n");
}


int main(int argc, char**argv)
{
	int arg = 0;
	int nXcorr = NCHANS*NANTS*(NANTS-1)/2*NPOLS*2;	
	float dThres = 3.5;	// modified z-score threshold recommended by Iglewicz and Hoaglin
	
	int32_t *readdata = (int32_t *)malloc(nXcorr*sizeof(int32_t));	// data in (all xcorr, no autocorr)
	int32_t *d_data;	// input data on device
	cudaMalloc((void **)&d_data, nXcorr*sizeof(int32_t));

// reorder data for fast baselines
//	int32_t *d_reor;	// input data on device
//	cudaMalloc((void **)&d_reor, nXcorr*sizeof(int32_t));
	
	uint8_t *h_flags = (uint8_t *)malloc(nXcorr*sizeof(uint8_t));	// data out (flags)
	uint8_t *d_flags;	// flags on device
	cudaMalloc((void **)&d_flags, nXcorr*sizeof(uint8_t));
	cudaMemset(d_flags, 0., nXcorr*sizeof(uint8_t));
	
	while ((arg=getopt(argc,argv,"t:h")) != -1)
    {
		switch (arg)
		{
			case 't':
			if (optarg)
			{
			  dThres = atof(optarg);
			  break;
			}
			else
			{
			  printf("-t flag requires argument");
			  usage();
			}
			case 'h':
				usage();
				return EXIT_SUCCESS;
		}
    }

	/*disk files management*/
	FILE *ptr;
	FILE *write_flg;
	ptr = fopen("noise.bin","rb");	// simulates 1 time sample of all xcorr (no auto corr)
	write_flg = fopen("flags.bin","wb");
	int rd;
	rd = fread(readdata,nXcorr,sizeof(int32_t),ptr);

	/*copy data onto GPU*/
	cudaMemcpy(d_data, readdata, nXcorr*sizeof(int32_t), cudaMemcpyHostToDevice);
	
	/*FLAG DATA*/
	flagg<<<NANTS*(NANTS-1)/2*NPOLS/512, 512>>>(d_data, d_flags, dThres);
	cudaDeviceSynchronize();
	
	/*copy back to CPU and write to disk*/
	cudaMemcpy(h_flags, d_flags, nXcorr*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	fwrite(h_flags,nXcorr,sizeof(uint8_t),write_flg);

	/*Free memory*/
	free(readdata);
	free(h_flags);

	cudaFree(d_data);
	cudaFree(d_flags);
	
	fclose(ptr);
	fclose(write_flg);
	return 0;
}
