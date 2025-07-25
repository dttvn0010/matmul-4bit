//nvcc -O3 -o matmul_q4.so -Xcompiler -fPIC --shared matmul_q4.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define bfloat162 __nv_bfloat162
#define bfloat16 __nv_bfloat16
#define float16 __half

#define int2bfloat16(x) __int2bfloat16_rn(x)
#define BLOCK_SIZE 256      // lcm(n_inp, n_out)

// inp 1xn_inp,  weight (n_inp/8) x n_out , zeros (n_inp/128) x n_out, scales (n_inp/128) x n_out
/*
for(int i = 0; i < n_out; i++)
{
  for(int j = 0; j < n_inp/8; j++)
  {
    uint w = weight[j * n_out + i];
    out[i] += ((w & 0xF) - zeros[j/16 * n_out + i]) * scales[j/16 * n_out + i] * inp[8*j];
    out[i] += ((w >> 4) -  zeros[j/16 * n_out + i]) * scales[j/16 * n_out + i] * inp[8*j+1];
    out[i] += ((w >> 8) -  zeros[j/16 * n_out + i]) * scales[j/16 * n_out + i] * inp[8*j+2];
    out[i] += ((w >> 12) -  zeros[j/16 * n_out + i]) * scales[j/16 * n_out + i] * inp[8*j+3];
    out[i] += ((w >> 16) -  zeros[j/16 * n_out + i]) * scales[j/16 * n_out + i] * inp[8*j+4];
    out[i] += ((w >> 20) -  zeros[j/16 * n_out + i]) * scales[j/16 * n_out + i] * inp[8*j+5];
    out[i] += ((w >> 24) -  zeros[j/16 * n_out + i]) * scales[j/16 * n_out + i] * inp[8*j+6];
    out[i] += ((w >> 28) -  zeros[j/16 * n_out + i]) * scales[j/16 * n_out + i] * inp[8*j+7];
  }
}
*/

template<typename half_type>
__global__ void matmul_q4_kernel(
    const  bfloat16* inp,
    const  uint*  q_weights,
    const  half_type*  scales,
    const  unsigned char*  qzeros,
    float* out,
    int n_inp,
    int n_out,
    int awq_group_size
) {
    int thrIdx = threadIdx.x;
    int i = blockIdx.y * 256 + thrIdx;    // max(i) = (n_out/256 - 1) * 256 + 255 = n_out - 1

    __shared__ bfloat16 blockinp[256];
    blockinp[thrIdx] = inp[blockIdx.x * 256 + thrIdx + blockIdx.z * n_inp];
    __syncthreads();

    int j1 = 32 * blockIdx.x;  //  max(j1) = 32 * (n_inp/256 - 1) = n_inp/8 - 32
    float sum = 0.0;

    #pragma unroll
    for(int n = 0; n < (256/awq_group_size); n++){
        int j2 = j1 + awq_group_size/8 * n;                                                 // max(j2) = n_inp/8 - awq_group_size/8
        float tmp_sum = 0.0;
        float  tmp_sum_inp = 0.0;
        float scale = static_cast<float>(scales[j2/16 * n_out + i]);
        float zero = scale * static_cast<float>(qzeros[j2/16 * n_out + i]);
        int offset = awq_group_size * n;

        #pragma unroll
        for(int k = 0; k < awq_group_size/8; k++) {
            int j = (j2 + k);                                                               // max(j) = n_inp/8 - 1
            uint w = q_weights[j * n_out + i];
            tmp_sum += float(
                int2bfloat16((w >> 0) & 0xF)  * blockinp[offset+8*k+0] + // inp[8*j];
                int2bfloat16((w >> 4) & 0xF)  * blockinp[offset+8*k+1] + // inp[8*j+1];
                int2bfloat16((w >> 8) & 0xF)  * blockinp[offset+8*k+2] + // inp[8*j+2];
                int2bfloat16((w >> 12) & 0xF) * blockinp[offset+8*k+3] + // inp[8*j+3];
                int2bfloat16((w >> 16) & 0xF) * blockinp[offset+8*k+4] + // inp[8*j+4];
                int2bfloat16((w >> 20) & 0xF) * blockinp[offset+8*k+5] + // inp[8*j+5];
                int2bfloat16((w >> 24) & 0xF) * blockinp[offset+8*k+6] + // inp[8*j+6];
                int2bfloat16((w >> 28) & 0xF) * blockinp[offset+8*k+7]   // inp[8*j+7];
            );
            /*
            bfloat162 acc = bfloat162{__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

            acc = __hfma2(
                bfloat162{int2bfloat16((w >> 0) & 0xF), int2bfloat16((w >> 4) & 0xF)}, 
                bfloat162{blockinp[offset+8*k+0], blockinp[offset+8*k+1]},
                acc
            );

            acc = __hfma2(
                bfloat162{int2bfloat16((w >> 8) & 0xF), int2bfloat16((w >> 12) & 0xF)}, 
                bfloat162{blockinp[offset+8*k+2], blockinp[offset+8*k+3]},
                acc
            );

            acc =__hfma2(
                bfloat162{int2bfloat16((w >> 16) & 0xF), int2bfloat16((w >> 20) & 0xF)}, 
                bfloat162{blockinp[offset+8*k+4], blockinp[offset+8*k+5]},
                acc
            );

            acc = __hfma2(
                bfloat162{int2bfloat16((w >> 24) & 0xF), int2bfloat16((w >> 28) & 0xF)}, 
                bfloat162{blockinp[offset+8*k+6], blockinp[offset+8*k+7]},
                acc
            );

            tmp_sum += float(acc.x + acc.y);
            */
        }
        for(int k = 0; k < awq_group_size; k++) tmp_sum_inp += static_cast<float>(blockinp[offset + k]);
        sum += (tmp_sum * scale - zero * tmp_sum_inp);
        
    }

    atomicAdd(&out[i+n_out * blockIdx.z], sum);
}


extern "C" void matmul_q4(
  const bfloat16* inp,
  const uint * q_weights,
  const void* scales,
  const unsigned char* qzeros,
  float* out,
  int batch,
  int n_inp,
  int n_out,
  int awq_group_size,
  int use_bf_scales
) {
    if(use_bf_scales){
        matmul_q4_kernel<bfloat16><<<dim3(n_inp/256, n_out/256, batch), 256>>>(
            inp, q_weights, (bfloat16*) scales, qzeros, out, n_inp, n_out, awq_group_size
        ); 
    }else{
        matmul_q4_kernel<float16><<<dim3(n_inp/256, n_out/256, batch), 256>>>(
            inp, q_weights, (float16*) scales, qzeros, out, n_inp, n_out, awq_group_size
        ); 
    }
}


template<typename half_type>
__global__ void populate_weight_kernel(
    const  uint*  q_weights,
    const  half_type*  scales,
    const  unsigned char*  qzeros,
    bfloat16* weight,
    int n_inp,
    int n_out,
    int awq_group_size
  ) {
    int i1 = blockIdx.y * 32 + threadIdx.y;
    int j = blockIdx.x * 32 + threadIdx.x;

    for (int n = 0; n < 8; n++)
    {
        int i = 8 * i1 + n;
        float scale = float(scales[8*j/awq_group_size * n_out + i]);
        float zero = scale * float(qzeros[8*j/awq_group_size * n_out + i]);
        uint w = q_weights[j * n_out + i];

        weight[(8*j + 0) * n_out + i] = __float2bfloat16(scale * float((w >> 0) & 0xF)  - zero);
        weight[(8*j + 1) * n_out + i] = __float2bfloat16(scale * float((w >> 4) & 0xF)  - zero);
        weight[(8*j + 2) * n_out + i] = __float2bfloat16(scale * float((w >> 8) & 0xF)  - zero);
        weight[(8*j + 3) * n_out + i] = __float2bfloat16(scale * float((w >> 12) & 0xF) - zero);
        weight[(8*j + 4) * n_out + i] = __float2bfloat16(scale * float((w >> 16) & 0xF) - zero);
        weight[(8*j + 5) * n_out + i] = __float2bfloat16(scale * float((w >> 20) & 0xF) - zero);
        weight[(8*j + 6) * n_out + i] = __float2bfloat16(scale * float((w >> 24) & 0xF) - zero);
        weight[(8*j + 7) * n_out + i] = __float2bfloat16(scale * float((w >> 28) & 0xF) - zero);
    }
}

template<typename half_type>
__global__ void populate_weight_kernel_transpose(
    const  uint*  q_weights,
    const  half_type*  scales,
    const  unsigned char*  qzeros,
    bfloat16* weight,
    int n_inp,
    int n_out,
    int awq_group_size
  ) {
    int i1 = blockIdx.y * 32 + threadIdx.y;
    int j = blockIdx.x * 32 + threadIdx.x;

    for (int n = 0; n < 8; n++)
    {
        int i = 8 * i1 + n;
        float scale = float(scales[j/(awq_group_size/8) * n_out + i]);
        float zero = scale * float(qzeros[j/(awq_group_size/8) * n_out + i]);
        uint w = q_weights[j * n_out + i];
        weight[i * n_inp + (8*j + 0)] = __float2bfloat16(scale * float((w >> 0) & 0xF)  - zero);
        weight[i * n_inp + (8*j + 1)] = __float2bfloat16(scale * float((w >> 4) & 0xF)  - zero);
        weight[i * n_inp + (8*j + 2)] = __float2bfloat16(scale * float((w >> 8) & 0xF)  - zero);
        weight[i * n_inp + (8*j + 3)] = __float2bfloat16(scale * float((w >> 12) & 0xF) - zero);
        weight[i * n_inp + (8*j + 4)] = __float2bfloat16(scale * float((w >> 16) & 0xF) - zero);
        weight[i * n_inp + (8*j + 5)] = __float2bfloat16(scale * float((w >> 20) & 0xF) - zero);
        weight[i * n_inp + (8*j + 6)] = __float2bfloat16(scale * float((w >> 24) & 0xF) - zero);
        weight[i * n_inp + (8*j + 7)] = __float2bfloat16(scale * float((w >> 28) & 0xF) - zero);
    }
}

extern "C" void populate_weight(
    const uint * q_weights,
    const void* scales,
    const unsigned char* qzeros,
    bfloat16* weight,
    int n_inp,
    int n_out,
    int awq_group_size,
    int transpose,
    int use_bf_scales
) {

    if(transpose){
        if(use_bf_scales) {
            populate_weight_kernel_transpose<bfloat16><<<dim3(n_inp/256, n_out/256), dim3(32, 32)>>>(
                q_weights, (bfloat16*) scales, qzeros, weight, n_inp, n_out, awq_group_size
            ); 
        }else{
            populate_weight_kernel_transpose<float16><<<dim3(n_inp/256, n_out/256), dim3(32, 32)>>>(
                q_weights, (float16*) scales, qzeros, weight, n_inp, n_out, awq_group_size
            ); 
        }
    }else{
        if(use_bf_scales) {
            populate_weight_kernel<bfloat16><<<dim3(n_inp/256, n_out/256), dim3(32, 32)>>>(
                q_weights, (bfloat16*) scales, qzeros, weight, n_inp, n_out, awq_group_size
            ); 
        }else{
            populate_weight_kernel<float16><<<dim3(n_inp/256, n_out/256), dim3(32, 32)>>>(
                q_weights, (float16*) scales, qzeros, weight, n_inp, n_out, awq_group_size
            ); 
        }
    }
}
