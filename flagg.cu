/*
nvcc flagg.cu -o flagg

file = open("noise.bin", "wb");
file.write(np.random.normal(0.,10.,2000000,dtype=np.float32));
file.close()
*/

#include <iostream>
#include <math.h>

#define NCHANS 1000	// # of channels -- assume no more than 2048 channels for now, see blinest call in main{}
#define NANTS 2000	// # of antennas
#define NFILT 101	// filter size -- need to be odd

__device__
void swap(float *p,float *q) {
   float t;
   
   t=*p; 
   *p=*q; 
   *q=t;
}

/*median value for NCHANS-long array*/
__device__
float medchans(float *a) {
	int i,j;
	float tmp[NCHANS] = {0};
	for (i = 0; i < NCHANS; i++)
		tmp[i] = a[i];
	
	for(i = 0; i < NCHANS-1; i++) {
		for(j = 0; j < NCHANS-i-1; j++) {
			if(tmp[j] > tmp[j+1])
				swap(&tmp[j],&tmp[j+1]);
		}
	}
	return tmp[(int)((NCHANS+1)/2-1)];
}

/*computes MAD for one spectrum*/
__device__
float mad(float *a) {
	int i;
	float med;
	float dev[NCHANS];
	med = medchans(a);
	for (i = 0; i < NCHANS; i++)
		dev[i] = abs(a[i]-med);
	return 1.4826*medchans(dev);
}

/*median value for NFILT-long array + manages edge effects*/
__device__
float medval(float *a, int idx, int n) {
	int i,j;
	float tmp[NFILT] = {0};
	if (idx < (NFILT-1)/2+1) {
		for (i=idx+(NFILT-1)/2+1; i<NFILT; i++)
			tmp[i] = a[0];
		for (i = 0; i<idx+(NFILT-1)/2+1; i++)
			tmp[i] = a[i];
	}
	else if(idx > n - ((NFILT-1)/2+1)) {
		for (i=n-(idx-(NFILT-1)/2); i<NFILT; i++)
			tmp[i] = a[n-1];
		for (i = idx-(NFILT-1)/2; i<n; i++)
			tmp[i-(idx-(NFILT-1)/2)] = a[i];
	}
	else{
		for (i = idx-(NFILT-1)/2; i<idx+(NFILT-1)/2+1; i++)
			tmp[i-(idx-(NFILT-1)/2)] = a[i];
	}
	
	for(i = 0; i<NFILT-1;i++) {
		for(j = 0;j < NFILT-i-1;j++) {
			if(tmp[j] > tmp[j+1])
				swap(&tmp[j],&tmp[j+1]);
		}
	}
	return tmp[(NFILT+1)/2-1];
}

__global__
void blinest(float *d_data, float *d_bline) {
	if (threadIdx.x < NCHANS/4){
		int idx = blockIdx.x * NCHANS/4 + threadIdx.x;
		int nAnt = (int)(blockIdx.x/4);
		int nSam = (blockIdx.x % 4) * NCHANS/4 + threadIdx.x;
		d_bline[idx] = medval(&d_data[nAnt*NCHANS], nSam, NCHANS);
	}
	__syncthreads();
}

__global__
void blincorr(float *d_data, float *d_bline) {
	if (threadIdx.x < NCHANS/4){
		int idx = blockIdx.x * NCHANS/4 + threadIdx.x;
		d_data[idx] = d_data[idx] - d_bline[idx];
	}
	__syncthreads();
}

__global__
void flagg(float *d_data, float *d_flags, float dThres)
{
	int i;
	int nAnt = blockIdx.x;
	float mv;
	mv = mad(&d_data[nAnt*NCHANS]);
	//printf("antenns %d : sigma = %f\n", nAnt, mv);
	for(i = nAnt*NCHANS; i < (nAnt+1)*NCHANS; i++)	// possible to write kernel to compute flags over blocks and threads
		if (d_data[i] > dThres*mv || d_data[i] < -dThres*mv)
			d_flags[i] = 1;
}

int main(void)
{

	int N = NANTS*NCHANS;	// size of 1 time sample, autocorrelations only
	float dThres = 6.;
	
	float *x = (float *)malloc(N*sizeof(float));	// data in (autocorrelations)
	float *fl_data = (float *)malloc(N*sizeof(float));	// data out (corrected data)
	float *d_data;	// input data on device
	cudaMalloc((void **)&d_data, N*sizeof(float));
	
	float *h_bline = (float *)malloc(N*sizeof(float));	// baseline
	float *d_bline;	// baseline
	cudaMalloc((void **)&d_bline, N*sizeof(float));
	
	float *h_flags = (float *)malloc(N*sizeof(float));	// data out (corrected data)
	float *d_flags;	// flags on device
	cudaMalloc((void **)&d_flags, N*sizeof(float));
	cudaMemset(d_flags, 0., N*sizeof(float));

	/*disk files management*/
	FILE *ptr;
	FILE *write_ptr;
	FILE *write_flg;
	FILE *write_bl;
	ptr = fopen("noise.bin","rb");
	write_ptr = fopen("output.bin","wb");
	write_flg = fopen("flags.bin","wb");
	write_bl = fopen("bline.bin","wb");
	int rd;
	rd = fread(x,N,sizeof(float),ptr);

	/*copy data onto GPU*/
	cudaMemcpy(d_data, x, N*sizeof(float), cudaMemcpyHostToDevice);
	
	/*FLAG DATA*/
	blinest<<<NANTS*4, NCHANS/4>>>(d_data, d_bline);
	cudaDeviceSynchronize();
	blincorr<<<NANTS*4, NCHANS/4>>>(d_data, d_bline);
	cudaDeviceSynchronize();
	flagg<<<NANTS, 1>>>(d_data, d_flags, dThres);
	cudaDeviceSynchronize();
	
	/*copy back to CPU and write to disk*/
	cudaMemcpy(fl_data, d_data, N*sizeof(float), cudaMemcpyDeviceToHost);
	fwrite(fl_data,N,sizeof(float),write_ptr);
	cudaMemcpy(h_flags, d_flags, N*sizeof(float), cudaMemcpyDeviceToHost);
	fwrite(h_flags,N,sizeof(float),write_flg);
	cudaMemcpy(h_bline, d_bline, N*sizeof(float), cudaMemcpyDeviceToHost);
	fwrite(h_bline,N,sizeof(float),write_bl);

	/*Free memory*/
	free(x);
	free(fl_data);
	free(h_bline);
	free(h_flags);
	cudaFree(d_data);
	cudaFree(d_bline);
	cudaFree(d_flags);
	fclose(ptr);
	fclose(write_ptr);
	fclose(write_flg);
	fclose(write_bl);
	return 0;
}
