#include <ATen/ATen.h>
// #include <torch/torch.h>
// #include <THC/THC.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

#include <iostream>

#include "math.h"
#include <cstdio>

#define IDX2(X, n1, n2, i1, i2) (X[(i2)*(n1) + (i1)])
#define IDX3(X, n1, n2, n3, i1, i2, i3) (X[(i3)*((n1)*(n2)) + (i2)*(n1) + (i1)])
#define IDX4(X, n1, n2, n3, n4, i1, i2, i3, i4) (X[(i4)*((n1)*(n2)*(n3)) + (i3)*((n1)*(n2)) + (i2)*(n1) + (i1)])
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


void __global__ conv_cls_ker_mt_for(float* res,
                                float* images,
                                float* kmag,
                                float* kori,
                                int imw,
                                int imh,
                                int nch,
                                int ptw,
                                int pth)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int idpx = blockIdx.x;

    int PH = (2 * pth + 1);
    // int PW = (2 * ptw + 1);

    int idth = threadIdx.x; //  nfs
    int idco = idth % PH; // index for column
    int idft = (idth - idco) / PH; // index for channel

    // int total_threads = gridDim.x * blockDim.x;
    int T = imw * imh * nch;
    int Reso = imw * imh;
    // int dfs = PH * PW * nch;

    __shared__ float res_tmp[1024];
    __shared__ float res_sum[1024];

    if (id < T * PH) 
    {
        res_tmp[idth] = 0;
        res_sum[idth] = 0;

        float ori = kori[idpx];
        float mag = kmag[idpx];
        float cosphi = cos(ori);
        float sinphi = sin(ori);
        float half = (mag - 1) / 2.0f;
        float linewdt = 1;
        int x = idpx % imw;
        int y = (idpx - x) / imw;
        float eps = 2.2204e-16f;

        // perform convolution 
        float val = 0;
        float sum_val = 0;
        //for (int d = 0; d < nch; d++)
        //{
            int q = idco - pth;
            for(int p = -ptw; p <= ptw; p++)
            {
                int px = x + p;
                int py = y + q;

                px = (px < 0) ? 0 : (px >= imw ? (imw - 1) : px);
                py = (py < 0) ? 0 : (py >= imh ? (imh - 1) : py);

                // compute filter value
                float dist2line = q * cosphi + p * sinphi;
                float dist2cent = sqrtf(p * p + q * q);

                if (abs(dist2line) <= linewdt & dist2cent <= half + linewdt + eps)
                {
                    if (dist2cent >= half) // if it is the end point
                    {
                      float x2lastpix = half - abs((p + dist2line*sinphi)/cosphi);
                      dist2line = sqrt(dist2line * dist2line + x2lastpix * x2lastpix);
                    }

                   dist2line = linewdt + eps - abs(dist2line);

                   if (dist2line<0) dist2line = 0;

                   val += images[idft * Reso + py * imw + px] * dist2line; //* filters[lb * dfs * nfs + dfs * idft + d *  PW * PH + idco * PW + p + ptw];
                   sum_val += dist2line;
                 }
            }
        res_tmp[idth] = val;
        res_sum[idth] = sum_val;

        __syncthreads();

    if(1)
    {
        if (idth < nch)
        {
            float sum = 0;
            for(int q = 0; q < PH; q++)
            {
                sum += res_sum[idth * PH + q];
            }

            float val = 0;
            for(int q = 0; q < PH; q++)
            {
                val += res_tmp[idth * PH + q];
            }
            res[Reso * idth + idpx] = val / sum;
        }
    }

    }
}


void __global__ compCosSin(float *mat_cos, 
                            float * mat_sin, 
                            float *kmag,
                            float* kori,
                            int imw, 
                            int imh)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < imw * imh)
    {
        float ori = kori[id];
        mat_cos[id] = cos(ori);
        mat_sin[id] = sin(ori);
    }
}


void __global__ conv_cls_ker_mt_back(float *res, 
                                    float *images, 
                                    float* mat_cos,
                                    float* mat_sin, 
                                    float *kmag,
                                    int imw, 
                                    int imh, 
                                    int nch, 
                                    int ptw, 
                                    int pth)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int idpx = blockIdx.x;

    int PH = (2 * pth + 1);
    // int PW = (2 * ptw + 1);

    int idth = threadIdx.x; //  nfs
    int idco = idth % PH; // index for column
    int idft = (idth - idco) / PH; // index for channel

    // int total_threads = gridDim.x * blockDim.x;
    int T = imw * imh * nch;
    int Reso = imw * imh;
    // int dfs = PH * PW * nch;

    __shared__ float res_tmp[1024];
    __shared__ float res_sum[1024];

    if (id < T * PH)
    {
        res_tmp[idth] = 0;
        res_sum[idth] = 0;


        float linewdt = 1;
        int x = idpx % imw;
        int y = (idpx - x) / imw;
        float eps = 2.2204e-16f;

        // perform convolution
        float val = 0;
        float sum_val = 0;

        //for (int d = 0; d < nch; d++)
        //{
            int q = idco - pth;
            for(int p = -ptw; p <= ptw; p++)
            {
                int px = x + p;
                int py = y + q;

                if (px >= 0 & px < imw & py >= 0 & py < imh)
                {
                    int idpx_curr = py * imw + px;
                    float mag = kmag[idpx_curr];
                    float cosphi = mat_cos[idpx_curr];
                    float sinphi = mat_sin[idpx_curr];
                    float half = (mag - 1) / 2.0f;

                    // compute filter value
                    float dist2line = -q * cosphi - p * sinphi;
                    float dist2cent = sqrtf(p * p + q * q);

                    if (abs(dist2line) <= linewdt & dist2cent <= half + linewdt + eps)
                    {
                        if (dist2cent >= half) // if it is the end point
                        {
                            float x2lastpix = half - abs((-p + dist2line*sinphi)/cosphi);
                            dist2line = sqrt(dist2line * dist2line + x2lastpix * x2lastpix);
                        }
                        dist2line = linewdt + eps - abs(dist2line);

                        if (dist2line<0) dist2line = 0;

                        val += images[idft * Reso + py * imw + px] * dist2line; //* filters[lb * dfs * nfs + dfs * idft + d *  PW * PH + idco * PW + p + ptw];
                        sum_val += dist2line;
                    }
                }
            }

            res_tmp[idth] = val;
            res_sum[idth] = sum_val;

            __syncthreads();

           if (idth < nch)
            {
                float sum = 0;
                for(int q = 0; q < PH; q++)
                {
                    sum += res_sum[idth * PH + q];
                }

                float val = 0;
                for(int q = 0; q < PH; q++)
                {
                    val += res_tmp[idth * PH + q];
                }
                if (sum > 1e-10)
                    res[Reso * idth + idpx] = val / sum;
                else
                    res[Reso * idth + idpx] = sum;
            }

    }
}


void __global__ inv_conv_motion_ker(float *res,
                            float *images,
                            float* indSet,
                            float* filters,
                            float* biases,
                            int imw,
                            int imh,
                            int nch,
                            int ptw,
                            int pth,
                            int ncs,
                            int nfs)
{
    int idpx = blockIdx.x;
    int lb = indSet[idpx];
    int x = idpx % imw;
    int y = (idpx - x) / imw;

    int PH = (2 * pth + 1);
    int PW = (2 * ptw + 1);

    int idth = threadIdx.x;

    int Reso = imw * imh;
    int dfs = PH * PW * nch;

    int PW2 = 32;
    int PH2 = 32;
    int dfsSpat = PW2 * PH2;

    __shared__ float res_tmp[3072];  //  storage is ok for 3 filters, otherwise, modulo !
    __shared__ float filt_tmp[3072];

    // load filter
    for (int d = idth; d < PW2 * PH2 * nch; d=d+blockDim.x)
    {
        int pos = d % dfsSpat;
        int ic = (d - pos) / dfsSpat;

        int px = pos % PW2;
        int py = (pos - px) / PW2;

        if (px < PW & py < PH)
        {
            filt_tmp[d] = filters[lb * dfs + ic * PW * PH + py * PW + px];
            // filt_tmp[d] = filters[lb * dfs * nfs + dfs * idft + ic * PW * PH + py * PW + px];
        }
        else
        {
            filt_tmp[d] = 0;
        }

    }
    __syncthreads();

    // load image data
    for (int d = idth; d < PW2 * PH2 * nch; d=d+blockDim.x) // iteration over channels
    {
        int pos = d % dfsSpat;
        int ic = (d - pos) / dfsSpat;

        int dx = pos % PW2;
        int dy = (pos - dx) / PW2;

        int px = x + dx - ptw;
        int py = y + dy - pth;

        px = (px < 0) ? 0 : (px >= imw ? (imw - 1) : px);
        py = (py < 0) ? 0 : (py >= imh ? (imh - 1) : py);

        res_tmp[d] = images[ic * Reso + py * imw + px];
     }
     __syncthreads();

     // compute the inner product
      if (idth < 512)
      {
            float val = 0;
            for(int l = idth; l < dfsSpat * nch; l = l + 512)
            {
                val = val +  res_tmp[l] * filt_tmp[l];
            }
            res_tmp[idth] = val;
        }

        __syncthreads();

        if(idth < 64)
        {
            float val = 0;
            for(int l = idth; l < 512; l = l + 64)
            {
                val = val + res_tmp[l];
            }
            res_tmp[idth] = val;
        }

        __syncthreads();
        if(idth < 8)
        {
            float val = 0;
            for(int l = idth; l < 64; l = l + 8)
            {
                val = val + res_tmp[l];
            }
            res_tmp[idth] = val;
        }

        __syncthreads();
       if(idth < 1)
        {
            float val = 0;
            for(int l = 0; l < 8; l++)
            {
                val = val + res_tmp[l];
            }
            res[idpx] = val;
        }
}


void __global__ conv_cls_ker(float *res, 
                            float *images, 
                            float* indSet,
                            float* filters, 
                            float* biases,
                            int imw, 
                            int imh, 
                            int nch, 
                            int ptw, 
                            int pth, 
                            int ncs, 
                            int nfs)
{
    // int id = blockIdx.x * blockDim.x + threadIdx.x;
    int idpx = blockIdx.x;
    int lb = indSet[idpx];
    int x = idpx % imw;
    int y = (idpx - x) / imw;

    int PH = (2 * pth + 1);
    int PW = (2 * ptw + 1);

    int idth = threadIdx.x;
    // int idco = idth % PH;
    // int idft = 0; //(idth - idco) / PH;

    // int T = imw * imh * nfs;
    int Reso = imw * imh;
    int dfs = PH * PW * nch;

    int PW2 = 32; 
    int PH2 = 32;
    int dfsSpat = PW2 * PH2;

    __shared__ float res_tmp[3072];  //  storage is ok for 3 filters, otherwise, modulo !
    __shared__ float filt_tmp[3072];    
        
    // load filter
    for (int d = idth; d < PW2 * PH2 * nch; d=d+blockDim.x) 
    {
        int pos = d % dfsSpat;
        int ic = (d - pos) / dfsSpat;
        
        int px = pos % PW2;
        int py = (pos - px) / PW2;
        
        if (px < PW & py < PH)
        {
            filt_tmp[d] = filters[lb * dfs + ic * PW * PH + py * PW + px]; 
            // filt_tmp[d] = filters[lb * dfs * nfs + dfs * idft + ic * PW * PH + py * PW + px]; 
        }
        else
        {
            filt_tmp[d] = 0;
        }

    }
    __syncthreads(); 
    
    
    // load image data
    for (int d = idth; d < PW2 * PH2 * nch; d=d+blockDim.x) // iteration over channels
    {
        int pos = d % dfsSpat;
        int ic = (d - pos) / dfsSpat;

        int dx = pos % PW2;
        int dy = (pos - dx) / PW2;
        
        int px = x + dx - ptw;
        int py = y + dy - pth;

        px = (px < 0) ? 0 : (px >= imw ? (imw - 1) : px);
        py = (py < 0) ? 0 : (py >= imh ? (imh - 1) : py);

        res_tmp[d] = images[ic * Reso + py * imw + px]; 
     }
     __syncthreads();   

     // compute the inner product


      if (idth < 512)
      {   
            float val = 0;
            for(int l = idth; l < dfsSpat * nch; l = l + 512)
            {
                val = val +  res_tmp[l] * filt_tmp[l];
            }
            res_tmp[idth] = val;
        }
        
        __syncthreads(); 
    
        if(idth < 64)
        {
            float val = 0;
            for(int l = idth; l < 512; l = l + 64)
            {
                val = val + res_tmp[l];
            }
            res_tmp[idth] = val;
        }
     
        __syncthreads(); 
        if(idth < 8)
        {
            float val = 0;
            for(int l = idth; l < 64; l = l + 8)
            {
                val = val + res_tmp[l];
            }
            res_tmp[idth] = val;
        }
        
        __syncthreads(); 
       if(idth < 1)
        {
            float val = 0;
            for(int l = 0; l < 8; l++)
            {
                val = val + res_tmp[l];
            }
            // res[idpx] = val + biases[lb];
            res[idpx] = val;
        }    

}

void __global__ conv_cls_kerv2(float *res, 
                            float *images, 
                            float* indSet,
                            float* filters, 
                            float* biases,
                            int imw, 
                            int imh, 
                            int nch, 
                            int ptw, 
                            int pth, 
                            int ncs, 
                            int nfs)
{
    int idpx = blockIdx.x;
    int lb = indSet[idpx];
    int x = idpx % imw;
    int y = (idpx - x) / imw;

    int PH = (2 * pth + 1);
    int PW = (2 * ptw + 1);

    int idth = threadIdx.x;
    int Reso = imw * imh;
    int dfs = PH * PW * nch;

    int PW2 = 32; 
    int PH2 = 32;
    int dfsSpat = PW2 * PH2;

    __shared__ float res_tmp[3072];
    __shared__ float filt_tmp[3072];  //  storage is ok for 3 filters, otherwise, modulo !

    res[idpx] = 0.0;
    
    // load filter 3 by 3
    for (int k = 0; k < nch; k+=3) {
        for (int d = idth + k * PW2 * PH2; d < PW2 * PH2 * (k+3); d=d+blockDim.x) {
            int dd = d % 3072;  // put d in the previous range of values
            if (d < PW2 * PH2 * nch) {
                int pos = dd % dfsSpat;
                int ic = (dd - pos) / dfsSpat + k;  // must take into acount the k previous channels
                
                int px = pos % PW2;
                int py = (pos - px) / PW2;
                if (px < PW & py < PH) {
                    filt_tmp[dd] = filters[lb * dfs + ic * PW * PH + py * PW + px]; 
                }
                else {
                    filt_tmp[dd] = 0;
                }
            }
            else {
                filt_tmp[dd] = 0;
            }
        }
        __syncthreads(); 
        
        
        // load image data
        for (int d = idth + k * PW2 * PH2; d < PW2 * PH2 * (k+3); d=d+blockDim.x) // iteration over pixels and channels
        {
            // if (idth == 1023 && idpx == 0) {
            //     printf("In load image\n");    
            // }
            int dd = d % 3072;
            if (d < PW2 * PH2 * nch) {
                int pos = dd % dfsSpat;
                int ic = (dd - pos) / dfsSpat + k;

                int dx = pos % PW2;
                int dy = (pos - dx) / PW2;
                
                int px = x + dx - ptw;
                int py = y + dy - pth;

                px = (px < 0) ? 0 : (px >= imw ? (imw - 1) : px);
                py = (py < 0) ? 0 : (py >= imh ? (imh - 1) : py);
                
                res_tmp[dd] = images[ic * Reso + py * imw + px]; 
            }
        }
        __syncthreads();   

        // compute the inner product
        if (idth < 512)
        {   
            float val = 0;
            for(int l = idth; l < dfsSpat * 3; l = l + 512)
            {
                val = val + res_tmp[l] * filt_tmp[l];
            }
            res_tmp[idth] = val;
        }
        __syncthreads(); 
        if(idth < 64)
        {
            float val = 0;
            for(int l = idth; l < 512; l = l + 64)
            {
                val = val + res_tmp[l];
            }
            res_tmp[idth] = val;
        }
        __syncthreads(); 
        if(idth < 8)
        {
            float val = 0;
            for(int l = idth; l < 64; l = l + 8)
            {
                val = val + res_tmp[l];
            }
            res_tmp[idth] = val;
        }
        __syncthreads(); 
        if(idth < 1)
        {
            float val = 0;
            for(int l = 0; l < 8; l++)
            {
                val = val + res_tmp[l];
            }
            res[idpx] += val;
            // res[idpx] += val + biases[lb];
        }    
        __syncthreads(); 
        
        // reset buffer
        // if (idth < 512)
        // {   
        //     for(int l = idth; l < 3072; l = l + 512)
        //     res_tmp[idth] = 0;
        //     filt_tmp[idth] = 0;
        // }
        // __syncthreads(); 
    }
}


void __global__ inv_conv_motion_bg_ker(float *res_dz,
                            float* dydz,
                            float* indSet,
                            float* filters,
                            int imw,
                            int imh,
                            int pth,
                            int ptw,
                            int dfs,
                            int nfs,
                            int ncs,
                            int nch,
                            int nimgs,
                            int npatches)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int idpx = blockIdx.x;

    int PH = (2 * pth + 1);
    int PW = (2 * ptw + 1);

    int idth = threadIdx.x; // * nfs
    int idco = idth % PH;  // index of column
    int idch = (idth - idco) / PH; // index of channel

    int T = imw * imh * nch;
    int Reso = imw * imh;

    __shared__ float res_tmp[2048];
    if (id < T * PH)
    {
        res_tmp[idth] = 0;
        int lb = indSet[idpx];
        int x = idpx % imw;
        int y = (idpx - x) / imw;

        // perform convolution
        float val = 0;
        for (int d = 0; d < nfs; d++) // d = 0
        {
            int q = idco - pth;
            int py = y + q;
            for(int p = -ptw; p <= ptw; p++)
            {
                int px = x + p;

                if (px >= 0 & px < imw & py >= 0 & py < imh)
                {
                   val += filters[lb * dfs + dfs * d + idch *  PW * PH + (-q + pth) * PW - p + ptw] * dydz[d * Reso + py * imw + px]; // convolution using mirror filter
                }
            }
        }
        res_tmp[idth] = val;

        __syncthreads();

        if (idth < nch)
        {
            float val = 0;
            for(int q = 0; q < PH; q++)
            {
                val += res_tmp[idth * PH + q];
            }
            res_dz[Reso * idth + idpx] = val;
        }

    }
}


void __global__ conv_cls_bg_ker(float *res_dz, 
                            float* dydz, 
                            float* indSet,
                            float* filters, 
                            int imw, 
                            int imh, 
                            int pth, 
                            int ptw,
                            int dfs, 
                            int nfs, 
                            int ncs, 
                            int nch, 
                            int nimgs,
                            int npatches)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int idpx = blockIdx.x;

    int PH = (2 * pth + 1);
    int PW = (2 * ptw + 1);

    int idth = threadIdx.x; // * nfs
    int idco = idth % PH;  // index of column
    int idch = (idth - idco) / PH; // index of channel

    // int total_threads = gridDim.x * blockDim.x;
    int T = imw * imh * nch;
    int Reso = imw * imh;
    //int dfs = PH * PW * nch;

    __shared__ float res_tmp[2048];
    if (id < T * PH)
    {
        res_tmp[idth] = 0;
        int lb = indSet[idpx];
        int x = idpx % imw;
        int y = (idpx - x) / imw;

        // perform convolution
        float val = 0;
        for (int d = 0; d < nfs; d++) // d = 0
        {
            int q = idco - pth;
            int py = y + q;
            for(int p = -ptw; p <= ptw; p++)
            {
                int px = x + p;
                // px = (px < 0) ? 0 : (px >= imw ? (imw - 1) : px);
                // py = (py < 0) ? 0 : (py >= imh ? (imh - 1) : py);

                if (px >= 0 & px < imw & py >= 0 & py < imh)
                {
                   val += filters[lb * dfs + dfs * d + idch *  PW * PH + (-q + pth) * PW - p + ptw] * dydz[d * Reso + py * imw + px]; // convolution using mirror filter
                }
            }
        }
        res_tmp[idth] = val;


        __syncthreads();

        if (idth < nch)
        {
            float val = 0;
            for(int q = 0; q < PH; q++)
            {
                val += res_tmp[idth * PH + q];
            }
            res_dz[Reso * idth + idpx] = val;
        }

    }
}


void __global__ conv_cls_bg_filt_ker2(float *res_df,  
                                    float* dydz, 
                                    float* images, 
                                    float* indSet,
                                    float *ind2indSet, 
                                    int imw, 
                                    int imh, 
                                    int pth, 
                                    int ptw,
                                    int dfs, 
                                    int nfs, 
                                    int ncs, 
                                    int nch, 
                                    int nimgs,
                                    int npatches)
{
    __shared__ float res_tmp[1024];

    // int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    // int total_threads = gridDim.x * blockDim.x;

    int id = blockIdx.x ;
    int id_th = threadIdx.x;
    int id_tmp = id % (dfs * nfs);
    int id_c = (id - id_tmp) / (dfs * nfs);
    int id_d = id_tmp % dfs;
    int id_f = (id_tmp - id_d) / dfs;

    //for(; pixel < nfs * ncs * dfs; pixel += total_threads)
    {
        int HPW = (ptw - 1) / 2;
        int HPH = (pth - 1) / 2;

        int id_po = id_d % (ptw * pth);
        int id_ch = (id_d - id_po) / (ptw * pth);
        int id_p = id_po % ptw - HPW;
        int id_q = (id_po - id_p) / ptw - HPH;

        int cur_c = ind2indSet[id_c];

        float val = 0;
        for(int q = id_th; q < npatches; q = q + blockDim.x) // choose a pixel
        {
            if (indSet[q] == cur_c)
            {
               int px = q % imw;
               int x = px + id_p;
               int y = (q - px) / imw + id_q;

               if(x >= 0 && x < imh && y >= 0 && y < imw)
               {
                  val += dydz[npatches * id_f + q] * images[npatches * id_ch + y * imw + x]; //[q * dfs + id_d]; //val
               }
            }
        }

        res_tmp[id_th] = val;

         __syncthreads();

        if (id_th < 32)
        {
            float val = 0;
            for(int l = id_th; l < blockDim.x; l = l + 32)
            {
                val = val + res_tmp[l];
            }
            res_tmp[id_th] = val;
        }

        __syncthreads();

        if(id_th < 1)
        {
            float val = 0;
            for(int l = 0; l < 32; l++)
            {
                val = val + res_tmp[l];
            }
            res_df[cur_c * nfs * dfs + dfs * id_f + id_d] = val;
        }

    }
}


/* nnline_ker evaluate xvar with non-linear functions with control points */
void __global__ nnline_ker(
        const float *xlab, const float *ylab, const float *xvar,
        float *yvar, const float *lmap, int N_ft, int N_ps, int N_cl, int N_cp)
{
    // int k, l, p, q;
    int k;

    //int total_number = M * N * D;
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float margin = xlab[1] - xlab[0];
    float margin_inv = 1 / margin;
    for(; n < N_ps * N_ft; n += total_threads)
    {
        int idx_px = n % N_ps;        
        int idx_ft = (n - idx_px) / N_ps;
        int idx_cl = lmap[idx_px]; 
      
        k = floor((xvar[n] - xlab[0]) * margin_inv);
        if(k < 0)
        {
            // yvar[n] = xvar[n]- xlab[0] + IDX3(ylab, N_cp, N_ft, N_cl, 0, idx_ft, idx_cl); //IDX2(ylab, P, D, 0, idz);
            yvar[n] = xvar[n] - xlab[0] + ylab[idx_cl * N_ft * N_cp + idx_ft * N_cp + 0]; //IDX2(ylab, P, D, 0, idz);
        }
        else if(k >= N_cp-1)
        {
            // yvar[n] = xvar[n]- xlab[N_cp-1] + IDX3(ylab, N_cp, N_ft, N_cl, N_cp-1, idx_ft, idx_cl); //IDX2(ylab, P, D, P-1, idz);
            yvar[n] = xvar[n] - xlab[0] + ylab[idx_cl * N_ft * N_cp + idx_ft * N_cp + N_cp - 1]; //IDX2(ylab, P, D, 0, idz);
        }
        else
        {
            // float dlf = IDX3(ylab, N_cp, N_ft, N_cl, k, idx_ft, idx_cl);
            // float drt = IDX3(ylab, N_cp, N_ft, N_cl, k + 1, idx_ft, idx_cl);
            float dlf = ylab[idx_cl * N_ft * N_cp + idx_ft * N_cp + k];
            float drt = ylab[idx_cl * N_ft * N_cp + idx_ft * N_cp + k + 1];
            yvar[n] = (drt - dlf) * (xvar[n] - xlab[k]) * margin_inv + dlf;  //IDX2(ylab, P, D, k, idz);
        }
    }
}

/**/
void __global__ nngetp_ker(
        const float *xlab,
        const float *xvar, float *pind, const float *lmap,
        int N_ft, int N_ps, int N_cl, int N_cp)
{
    int total_number = N_ft * N_ps;
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float margin = xlab[1] - xlab[0];
    float margin_inv = 1 / margin;

    for(; n<total_number; n += total_threads)
    {
        int idx_px = n % N_ps;        
        int idx_ft = (n - idx_px) / N_ps;
        
        // IDX2(pind, N_ps, N_ft, idx_px, idx_ft) = floor((xvar[n] - xlab[0]) * margin_inv);
        pind[N_ps * idx_ft + idx_px] = floor((xvar[n] - xlab[0]) * margin_inv);
    }
}


/* nnback_ker back propagation computing gradients */
void __global__ nnbackx_ker(
        const float *xlab, const float *ylab,
        const float *xvar, const float *yvar,
        float *grad, const float *lmap, int N_ft, int N_ps, int N_cl, int N_cp)
{
    int k;

    int total_number = N_ft * N_ps;

    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    float margin = xlab[1] - xlab[0];
    float margin_inv = 1 / margin;

    for(; n<total_number; n += total_threads)
    {
        int idx_px = n % N_ps;        
        int idx_ft = (n - idx_px) / N_ps;
        int idx_cl = lmap[idx_px]; 
     
        k = floor((xvar[n] - xlab[0]) / margin);
        if(k<0 || k>=N_cp-1)
        {
            grad[n] = 1 * yvar[n];
        }
        else
        {
            float dx2 = ylab[idx_cl * N_ft * N_cp + idx_ft * N_cp + k + 1];
            float dx1 = ylab[idx_cl * N_ft * N_cp + idx_ft * N_cp + k];
            // grad[n] = ((IDX3(ylab, N_cp, N_ft, N_cl, k+1, idx_ft, idx_cl) - IDX3(ylab, N_cp, N_ft, N_cl, k, idx_ft, idx_cl)) * margin_inv) * yvar[n];
            grad[n] = ((dx2 - dx1) * margin_inv) * yvar[n];
        }
    }
}

void __global__ nnbackw_ker(
        const float *xlab, const float *ylab,
        const float *xvar, const float *yvar, const float *pind,
        float *grad, const float *lmap, int N_ft, int N_ps, int N_cl, int N_cp)
{
  //  __shared__ float INDP[128][128];
  //  __shared__ float L[41];
  //  __shared__ float Y[128][128];
  //  __shared__ float X[128][128];
    
    // int m, n, p, q;
    int m, p;
    int total_number = N_cp * N_ft * N_cl;

    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    float margin = xlab[1] - xlab[0];
    float margin_inv = 1 / (margin);
    
    // do computation
    for(; k<total_number; k+=total_threads)
    {
        int idx = k % (N_cp * N_ft);
        int idx_cp = idx % N_cp;
        int idx_ft = (idx-idx_cp) / N_cp;
        int idx_cl = (k - idx) / (N_cp * N_ft);

        float sum = 0;

        for(m=0; m<N_ps; m++)
        {
           // p = IDX2(pind, N_ft, N_ps, idx_ft, m);
           p = pind[idx_ft * N_ps + m];
           int idx_lb = lmap[m];
           
           if (idx_lb == idx_cl)
           {
               float dx = xvar[idx_ft * N_ps + m];
               float dy = yvar[idx_ft * N_ps + m];
               if(p == idx_cp-1 && p>=0)
               {
                   sum += (dx - xlab[p]) * margin_inv * dy;
                   // sum += (IDX2(xvar, N_ps, N_ft, m, idx_ft)- xlab[p]) * margin_inv * IDX2(yvar, N_ps,N_ft, m,  idx_ft);
               }
               else if(p == idx_cp && p<N_cp-1)
               {
                   sum += (1 - (dx - xlab[p]) * margin_inv) * dy;
                   // sum += (1 - (IDX2(xvar,  N_ps, N_ft, m, idx_ft) - xlab[p]) * margin_inv) * IDX2(yvar, N_ps, N_ft, m, idx_ft);
               }
           }
        }

        // IDX3(grad, N_cp, N_ft, N_cl, idx_cp, idx_ft, idx_cl) = sum;
        grad[idx_cl * N_ft * N_cp + idx_ft * N_cp + idx_cp] = sum;
    } 
}

// CUDA forward declarations

at::Tensor conv_motion_cuda_forward(
    at::Tensor input,
    at::Tensor mag,
    at::Tensor ori) {

    int nch = (int)input.size(1);
    int imw = (int)input.size(3);
    int imh = (int)input.size(2);
    int ptw = 12;
    int pth = 12;

    auto output = at::ones_like(input);

    int const threadsPerBlock = (2 * pth + 1) * nch; //nfs;
    // int const blocksPerGrid = 150*150;
    // int const blocksPerGrid = 50000;
    int const blocksPerGrid = imw * imh ;

    // std::cout << "block " << blocksPerGrid << std::endl;
    // std::cout << "threads " << threadsPerBlock << std::endl;
    // std::cout << "here" << std::endl;
    conv_cls_ker_mt_for<<<blocksPerGrid, threadsPerBlock>>>(output.data<float>(), input.data<float>(), mag.data<float>(), ori.data<float>(), imw, imh, nch, ptw, pth);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return output;
}

at::Tensor inv_conv_motion_cuda_forward(
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor bias) {
    int nch = (int)input.size(1);
    int imw = (int)input.size(3);
    int imh = (int)input.size(2);
    int pth = (int)weight.size(2);
    int ptw = (int)weight.size(3);
    int ncs = (int)weight.size(0);  // 361
    int nfs = 1;  // 1

    auto output = at::zeros_like(input).contiguous();

    int const threadsPerBlock = 1024;
    int const blocksPerGrid = imw * imh;

    inv_conv_motion_ker<<<blocksPerGrid, threadsPerBlock>>>(output.data<float>(), input.data<float>(), labels.data<float>(), weight.data<float>(), bias.data<float>(), imw, imh, nch, (ptw - 1) / 2, (pth - 1) / 2, ncs, nfs);

    gpuErrchk(cudaPeekAtLastError());

    return output;
}

at::Tensor conv_cls_cuda_forward(
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor bias) {
    int nch = (int)input.size(1);
    int imw = (int)input.size(3);
    int imh = (int)input.size(2);
    int pth = (int)weight.size(2);
    int ptw = (int)weight.size(3);
    int ncs = (int)weight.size(0);  // 361
    int nfs = 1;  // 1

    auto output = at::zeros_like(input).sum(1, true).contiguous();

    int const threadsPerBlock = 1024;
    int const blocksPerGrid = imw * imh;

    conv_cls_ker<<<blocksPerGrid, threadsPerBlock>>>(output.data<float>(), input.data<float>(), labels.data<float>(), weight.data<float>(), bias.data<float>(), imw, imh, nch, (ptw - 1) / 2, (pth - 1) / 2, ncs, nfs);

    gpuErrchk(cudaPeekAtLastError());

    return output;
}

at::Tensor line_cuda_forward(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor labels) {
    int N_ft = (int)xvar.size(1);
    int N_w  = (int)xvar.size(3);
    int N_h  = (int)xvar.size(2);

    int N_ps = N_w * N_h;
    int N_cl = (int)ylab.size(0);
    int N_cp = (int)ylab.size(2);

    auto yvar = at::zeros_like(xvar);

    int const threadsPerBlock = 1024;
    // int const threadsPerBlock = 256;
    int blocksPerGrid = (N_ps * N_ft + threadsPerBlock - 1) / threadsPerBlock;
    nnline_ker<<<blocksPerGrid, threadsPerBlock>>>(xlab.data<float>(), ylab.data<float>(), xvar.data<float>(), yvar.data<float>(), labels.data<float>(), N_ft, N_ps, N_cl, N_cp);

    gpuErrchk(cudaPeekAtLastError());

    return yvar;
    }

// CUDA backward declarations

at::Tensor conv_motion_cuda_backward(
    at::Tensor grad_out,
    at::Tensor mag,
    at::Tensor ori,
    at::Tensor mcos,
    at::Tensor msin) {
    int nch = (int)grad_out.size(1);
    int imw = (int)grad_out.size(3);
    int imh = (int)grad_out.size(2);
    int ptw = 12;
    int pth = 12;

    auto grad_in = at::zeros_like(grad_out);


    int const threadsPerBlock = (2 * pth + 1) * nch; //nfs;
    int const blocksPerGrid = imw * imh;

    int const threadsPerBlock1 = 1024; //nfs;
    int const blocksPerGrid1 = ( imw * imh + threadsPerBlock - 1) / threadsPerBlock;

    compCosSin<<<blocksPerGrid1, threadsPerBlock1>>>(mcos.data<float>(), msin.data<float>(), mag.data<float>(), ori.data<float>(), imw, imh);
    conv_cls_ker_mt_back<<<blocksPerGrid, threadsPerBlock>>>(grad_in.data<float>(), grad_out.data<float>(), mcos.data<float>(), msin.data<float>(), mag.data<float>(), imw, imh, nch, ptw, pth);

    gpuErrchk(cudaPeekAtLastError());
    return grad_in;
}


at::Tensor inv_conv_motion_cuda_backward(
    at::Tensor grad_out,
    at::Tensor labels,
    at::Tensor weight) {
    int nch = (int)weight.size(1);
    int imw = (int)grad_out.size(3);
    int imh = (int)grad_out.size(2);
    int pth = (int)weight.size(2);  // 31
    int ptw = (int)weight.size(3);  // 31
    int ncs = (int)weight.size(0);  // 361
    int nfs = 1;  // 1
    int dfs = nch * pth * ptw;
    int nps = imw * imh;

    auto grad_in = at::zeros_like(grad_out).contiguous();

    int threadsPerBlock = 1024;
    threadsPerBlock = pth * nch;
    int blocksPerGrid = imw * imh;
    inv_conv_motion_bg_ker<<<blocksPerGrid, threadsPerBlock>>>(grad_in.data<float>(), grad_out.data<float>(), labels.data<float>(), weight.data<float>(), imw, imh, (pth-1)/2, (ptw-1)/2, dfs, nfs, ncs, nch, 1, nps);

    gpuErrchk(cudaPeekAtLastError());
    return grad_in;
}


at::Tensor conv_cls_cuda_backward1(
    at::Tensor grad_out,
    at::Tensor labels,
    at::Tensor weight) {
    int nch = (int)weight.size(1);
    int imw = (int)grad_out.size(3);
    int imh = (int)grad_out.size(2);
    int pth = (int)weight.size(2);  // 31
    int ptw = (int)weight.size(3);  // 31
    int ncs = (int)weight.size(0);  // 361
    int nfs = (int)1;  // 1
    int dfs = nch * pth * ptw;
    int nps = imw * imh;

    auto grad_in = at::zeros_like(grad_out).expand({1,nch,imh,imw}).contiguous();

    int threadsPerBlock = 1024;
    threadsPerBlock = pth * nch;
    int blocksPerGrid = imw * imh;
    conv_cls_bg_ker<<<blocksPerGrid, threadsPerBlock>>>(grad_in.data<float>(), grad_out.data<float>(), labels.data<float>(), weight.data<float>(), imw, imh, (pth-1)/2, (ptw-1)/2, dfs, nfs, ncs, nch, 1, nps);

    gpuErrchk(cudaPeekAtLastError());
    return grad_in;
}

at::Tensor conv_cls_cuda_backward2(
    at::Tensor grad_out,
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor labels_unique) {
    int nch = (int)weight.size(1);
    int imw = (int)grad_out.size(3);
    int imh = (int)grad_out.size(2);
    int pth = (int)weight.size(2);  // 31
    int ptw = (int)weight.size(3);  // 31
    int ncs = (int)weight.size(0);  // 361
    int nfs = 1;  // 1
    int dfs = nch * pth * ptw;
    int nps = imw * imh;
    int nValidCls = (int)labels_unique.size(0);

    auto grad_w = at::zeros_like(weight);

    int threadsPerBlock = 1024;
    int blocksPerGrid_filt = nValidCls * dfs * nfs;
    conv_cls_bg_filt_ker2<<<blocksPerGrid_filt, threadsPerBlock>>>(grad_w.data<float>(), grad_out.data<float>(), input.data<float>(), labels.data<float>(), labels_unique.data<float>(), imw, imh, pth, ptw, dfs, nfs, ncs, nch, 1, nps);

    gpuErrchk(cudaPeekAtLastError());
    return grad_w;
}


at::Tensor line_cuda_backward1(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor yvar,
    at::Tensor labels) {
    int N_ft = (int)xvar.size(1);
    int N_w  = (int)xvar.size(3);
    int N_h  = (int)xvar.size(2);

    int N_ps = N_w * N_h;
    int N_cl = (int)ylab.size(0);
    int N_cp = (int)ylab.size(2);

    auto grad_x = at::zeros_like(xvar);

    int const threadsPerBlock = 1024;
    // int const threadsPerBlock = 256;
    int blocksPerGrid = (N_ps * N_ft + threadsPerBlock - 1) / threadsPerBlock;
    nnbackx_ker<<<blocksPerGrid, threadsPerBlock>>>(xlab.data<float>(), ylab.data<float>(), xvar.data<float>(), yvar.data<float>(), grad_x.data<float>(), labels.data<float>(),  N_ft, N_ps, N_cl, N_cp);
    
    gpuErrchk(cudaPeekAtLastError());
    return grad_x;
    }

at::Tensor line_cuda_backward2(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor yvar,
    at::Tensor labels) {
    int N_ft = (int)xvar.size(1);
    int N_w  = (int)xvar.size(3);
    int N_h  = (int)xvar.size(2);

    int N_ps = N_w * N_h;
    int N_cl = (int)ylab.size(0);
    int N_cp = (int)ylab.size(2);

    auto pind = at::zeros_like(xvar);
    auto grad_w = at::zeros_like(ylab);

    int const threadsPerBlock = 1024;
    int blocksPerGrid = (N_ps * N_ft + threadsPerBlock - 1) / threadsPerBlock;
    nngetp_ker<<<blocksPerGrid, threadsPerBlock>>>(xlab.data<float>(), xvar.data<float>(), pind.data<float>(), labels.data<float>(), N_ft, N_ps, N_cl, N_cp);
    
    int threadsPerBlock2 = threadsPerBlock;
    blocksPerGrid = (N_cp * N_ft * N_cl + threadsPerBlock2 - 1) / threadsPerBlock2;
    nnbackw_ker<<<blocksPerGrid, threadsPerBlock2>>>(xlab.data<float>(), ylab.data<float>(), xvar.data<float>(), yvar.data<float>(), pind.data<float>(), grad_w.data<float>(), labels.data<float>(), N_ft, N_ps, N_cl, N_cp);
    
    gpuErrchk(cudaPeekAtLastError());
    return grad_w;
    }
