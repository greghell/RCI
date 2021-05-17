/*
nvcc flagg_low.cu -o flagg_low
./flagg_low -h
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

void extract_autocorrs(int32_t *readdata, float *h_data) {
	int nc, na, np;
	int nChanLen = (NANTS*(NANTS-1)/2+NANTS) * NPOLS * 2;
	int idx;
	for(nc = 0; nc < NCHANS; nc++){
		for(na = 0; na < NANTS; na++){
			for(np = 0; np < NPOLS; np++){
				idx = (int)((NANTS*(NANTS+1)/2) - ((NANTS-na)*((NANTS-na)+1)/2));
				h_data[na*(NCHANS*NPOLS)+nc*NPOLS+np] = (float)readdata[nc*nChanLen + idx*(NPOLS*2)+np*2];
			}
		}
	}
}

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
	float me = medchans(a,0);	// interleaved array
	*medval = me;
	for (i = 0; i < NCHANS; i++)
		dev[i] = abs(a[2*i]-me);	// polarizations are interleaved
	*madval = medchans(dev,1);		// contiguous array
}

__global__
void flagg(float *d_data, int32_t *d_flags, float dThres, float *d_madmed) {
	int nAnt = blockIdx.x;
	int nPol = threadIdx.x;
	int nChanLen = NCHANS*NPOLS;
	if (nPol < 2){
		int i;
		float madval = 0;
		float medval = 0;
		float M;
		mad(&d_data[nAnt*nChanLen+nPol], &madval, &medval);
		d_madmed[nAnt*NPOLS*2+nPol*2] = madval;	// TO TEST
		d_madmed[nAnt*NPOLS*2+nPol*2+1] = medval;	// TO TEST
		for(i = 0; i < NCHANS; i++){
			M = 0.6745*((float)d_data[nAnt*nChanLen + nPol + 2*i]-medval) / madval;
			if (abs(M) > dThres)
				d_flags[nAnt*nChanLen + nPol + 2*i] = 1;
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
	int nXcorr = NCHANS*(NANTS*(NANTS-1)/2 + NANTS)*NPOLS*2;	
	int N = NANTS*NCHANS*NPOLS;	// size of 1 time sample, autocorrelations only
	float dThres = 3.5;	// modified z-score threshold recommended by Iglewicz and Hoaglin
	
	int32_t *readdata = (int32_t *)malloc(nXcorr*sizeof(int32_t));	// data in (all xcorr incl auto corr)
	float *h_data = (float *)malloc(N*sizeof(float));	// data in (autocorrelations)
	float *fl_data = (float *)malloc(N*sizeof(float));	// data out (corrected data)
	float *d_data;	// input data on device
	cudaMalloc((void **)&d_data, N*sizeof(float));
	
	int32_t *h_flags = (int32_t *)malloc(N*sizeof(int32_t));	// data out (corrected data)
	int32_t *d_flags;	// flags on device
	cudaMalloc((void **)&d_flags, N*sizeof(int32_t));
	cudaMemset(d_flags, 0., N*sizeof(int32_t));
	
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
	
	/*TO TEST*/
	float *h_madmed = (float *)malloc(NANTS*NPOLS*2*sizeof(float));
	float *d_madmed;	// flags on device
	cudaMalloc((void **)&d_madmed, NANTS*NPOLS*2*sizeof(float));
	/*TO TEST*/

	/*disk files management*/
	FILE *ptr;
	FILE *write_ptr;
	FILE *write_flg;
	FILE *write_madmed;	// TO TEST
	ptr = fopen("xcorr_in.bin","rb");	// simulates 1 time sample of all xcorr (incl auto corr)
	write_ptr = fopen("output.bin","wb");
	write_flg = fopen("flags.bin","wb");
	write_madmed = fopen("madmed.bin","wb");	// TO TEST
	int rd;
	rd = fread(readdata,nXcorr,sizeof(int32_t),ptr);
	
	/*extract autocorr and write them to h_data*/
	extract_autocorrs(readdata, h_data);

	/*copy data onto GPU*/
	cudaMemcpy(d_data, h_data, N*sizeof(float), cudaMemcpyHostToDevice);
	
	/*FLAG DATA*/
	flagg<<<NANTS, 2>>>(d_data, d_flags, dThres, d_madmed);
	cudaDeviceSynchronize();
	
	/*copy back to CPU and write to disk*/
	cudaMemcpy(fl_data, d_data, N*sizeof(float), cudaMemcpyDeviceToHost);
	fwrite(fl_data,N,sizeof(float),write_ptr);
	cudaMemcpy(h_flags, d_flags, N*sizeof(int32_t), cudaMemcpyDeviceToHost);
	fwrite(h_flags,N,sizeof(int32_t),write_flg);
	
	cudaMemcpy(h_madmed, d_madmed, NANTS*NPOLS*2*sizeof(float), cudaMemcpyDeviceToHost);	// TO TEST
	fwrite(h_madmed,NANTS*NPOLS*2,sizeof(float),write_madmed);				// TO TEST

	/*Free memory*/
	free(readdata);
	free(h_data);
	free(fl_data);
	free(h_flags);
	
	/*TO TEST*/
	free(h_madmed);
	cudaFree(d_madmed);
	fclose(write_madmed);
	/*TO TEST*/
	
	cudaFree(d_data);
	cudaFree(d_flags);
	fclose(ptr);
	fclose(write_ptr);
	fclose(write_flg);
	return 0;
}
