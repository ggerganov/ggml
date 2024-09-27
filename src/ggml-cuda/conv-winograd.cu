#include "conv-winograd.cuh"
#include "convert.cuh"

__device__ void __inline__ outer_product(float4* input_frag, float4* filter_frag, float4 accumulator[][16]){
    accumulator[0][0].x += input_frag[0].x*filter_frag[0].x;
    accumulator[0][0].y += input_frag[0].y*filter_frag[0].x;
    accumulator[0][0].z += input_frag[0].z*filter_frag[0].x;
    accumulator[0][0].w += input_frag[0].w*filter_frag[0].x;
    
    accumulator[0][1].x += input_frag[1].x*filter_frag[0].x;
    accumulator[0][1].y += input_frag[1].y*filter_frag[0].x;
    accumulator[0][1].z += input_frag[1].z*filter_frag[0].x;                                     
    accumulator[0][1].w += input_frag[1].w*filter_frag[0].x;
  
    accumulator[0][2].x += input_frag[0].x*filter_frag[0].y;
    accumulator[0][2].y += input_frag[0].y*filter_frag[0].y;                                
    accumulator[0][2].z += input_frag[0].z*filter_frag[0].y;                                
    accumulator[0][2].w += input_frag[0].w*filter_frag[0].y;                                  
  
    accumulator[0][3].x += input_frag[1].x*filter_frag[0].y;
    accumulator[0][3].y += input_frag[1].y*filter_frag[0].y;                                 
    accumulator[0][3].z += input_frag[1].z*filter_frag[0].y;                                 
    accumulator[0][3].w += input_frag[1].w*filter_frag[0].y;
                                      
    accumulator[0][4].x += input_frag[0].x*filter_frag[0].z;
    accumulator[0][4].y += input_frag[0].y*filter_frag[0].z;                                     
    accumulator[0][4].z += input_frag[0].z*filter_frag[0].z;                                     
    accumulator[0][4].w += input_frag[0].w*filter_frag[0].z;
                                        
    accumulator[0][5].x += input_frag[1].x*filter_frag[0].z;
    accumulator[0][5].y += input_frag[1].y*filter_frag[0].z;                                     
    accumulator[0][5].z += input_frag[1].z*filter_frag[0].z;                                     
    accumulator[0][5].w += input_frag[1].w*filter_frag[0].z;
  
    accumulator[0][6].x += input_frag[0].x*filter_frag[0].w;
    accumulator[0][6].y += input_frag[0].y*filter_frag[0].w;                                   
    accumulator[0][6].z += input_frag[0].z*filter_frag[0].w;                                   
    accumulator[0][6].w += input_frag[0].w*filter_frag[0].w;
                                        
    accumulator[0][7].x += input_frag[1].x*filter_frag[0].w;
    accumulator[0][7].y += input_frag[1].y*filter_frag[0].w;                                    
    accumulator[0][7].z += input_frag[1].z*filter_frag[0].w;                                    
    accumulator[0][7].w += input_frag[1].w*filter_frag[0].w;
  
    //
    accumulator[0][8].x += input_frag[0].x*filter_frag[1].x;
    accumulator[0][8].y += input_frag[0].y*filter_frag[1].x;
    accumulator[0][8].z += input_frag[0].z*filter_frag[1].x;
    accumulator[0][8].w += input_frag[0].w*filter_frag[1].x;
    
    accumulator[0][9].x += input_frag[1].x*filter_frag[1].x;
    accumulator[0][9].y += input_frag[1].y*filter_frag[1].x;
    accumulator[0][9].z += input_frag[1].z*filter_frag[1].x;                                     
    accumulator[0][9].w += input_frag[1].w*filter_frag[1].x;
  
    accumulator[0][10].x += input_frag[0].x*filter_frag[1].y;
    accumulator[0][10].y += input_frag[0].y*filter_frag[1].y;                                
    accumulator[0][10].z += input_frag[0].z*filter_frag[1].y;                                
    accumulator[0][10].w += input_frag[0].w*filter_frag[1].y;                                  
  
    accumulator[0][11].x += input_frag[1].x*filter_frag[1].y;
    accumulator[0][11].y += input_frag[1].y*filter_frag[1].y;                                 
    accumulator[0][11].z += input_frag[1].z*filter_frag[1].y;                                 
    accumulator[0][11].w += input_frag[1].w*filter_frag[1].y;
                                      
    accumulator[0][12].x += input_frag[0].x*filter_frag[1].z;
    accumulator[0][12].y += input_frag[0].y*filter_frag[1].z;                                     
    accumulator[0][12].z += input_frag[0].z*filter_frag[1].z;                                     
    accumulator[0][12].w += input_frag[0].w*filter_frag[1].z;
                                        
    accumulator[0][13].x += input_frag[1].x*filter_frag[1].z;
    accumulator[0][13].y += input_frag[1].y*filter_frag[1].z;                                     
    accumulator[0][13].z += input_frag[1].z*filter_frag[1].z;                                     
    accumulator[0][13].w += input_frag[1].w*filter_frag[1].z;
  
    accumulator[0][14].x += input_frag[0].x*filter_frag[1].w;
    accumulator[0][14].y += input_frag[0].y*filter_frag[1].w;                                   
    accumulator[0][14].z += input_frag[0].z*filter_frag[1].w;                                   
    accumulator[0][14].w += input_frag[0].w*filter_frag[1].w;
                                        
    accumulator[0][15].x += input_frag[1].x*filter_frag[1].w;
    accumulator[0][15].y += input_frag[1].y*filter_frag[1].w;                                    
    accumulator[0][15].z += input_frag[1].z*filter_frag[1].w;                                    
    accumulator[0][15].w += input_frag[1].w*filter_frag[1].w;
  
    //////
    accumulator[1][0].x += input_frag[2].x*filter_frag[2].x;
    accumulator[1][0].y += input_frag[2].y*filter_frag[2].x;
    accumulator[1][0].z += input_frag[2].z*filter_frag[2].x;
    accumulator[1][0].w += input_frag[2].w*filter_frag[2].x;
    
    accumulator[1][1].x += input_frag[3].x*filter_frag[2].x;
    accumulator[1][1].y += input_frag[3].y*filter_frag[2].x;
    accumulator[1][1].z += input_frag[3].z*filter_frag[2].x;                                     
    accumulator[1][1].w += input_frag[3].w*filter_frag[2].x;
  
    accumulator[1][2].x += input_frag[2].x*filter_frag[2].y;
    accumulator[1][2].y += input_frag[2].y*filter_frag[2].y;                                
    accumulator[1][2].z += input_frag[2].z*filter_frag[2].y;                                
    accumulator[1][2].w += input_frag[2].w*filter_frag[2].y;                                  
  
    accumulator[1][3].x += input_frag[3].x*filter_frag[2].y;
    accumulator[1][3].y += input_frag[3].y*filter_frag[2].y;                                 
    accumulator[1][3].z += input_frag[3].z*filter_frag[2].y;                                 
    accumulator[1][3].w += input_frag[3].w*filter_frag[2].y;
                                      
    accumulator[1][4].x += input_frag[2].x*filter_frag[2].z;
    accumulator[1][4].y += input_frag[2].y*filter_frag[2].z;                                     
    accumulator[1][4].z += input_frag[2].z*filter_frag[2].z;                                     
    accumulator[1][4].w += input_frag[2].w*filter_frag[2].z;
                                        
    accumulator[1][5].x += input_frag[3].x*filter_frag[2].z;
    accumulator[1][5].y += input_frag[3].y*filter_frag[2].z;                                     
    accumulator[1][5].z += input_frag[3].z*filter_frag[2].z;                                     
    accumulator[1][5].w += input_frag[3].w*filter_frag[2].z;
  
    accumulator[1][6].x += input_frag[2].x*filter_frag[2].w;
    accumulator[1][6].y += input_frag[2].y*filter_frag[2].w;                                   
    accumulator[1][6].z += input_frag[2].z*filter_frag[2].w;                                   
    accumulator[1][6].w += input_frag[2].w*filter_frag[2].w;
                                        
    accumulator[1][7].x += input_frag[3].x*filter_frag[2].w;
    accumulator[1][7].y += input_frag[3].y*filter_frag[2].w;                                    
    accumulator[1][7].z += input_frag[3].z*filter_frag[2].w;                                    
    accumulator[1][7].w += input_frag[3].w*filter_frag[2].w;
  
    //
    accumulator[1][8].x += input_frag[2].x*filter_frag[3].x;
    accumulator[1][8].y += input_frag[2].y*filter_frag[3].x;
    accumulator[1][8].z += input_frag[2].z*filter_frag[3].x;
    accumulator[1][8].w += input_frag[2].w*filter_frag[3].x;
    
    accumulator[1][9].x += input_frag[3].x*filter_frag[3].x;
    accumulator[1][9].y += input_frag[3].y*filter_frag[3].x;
    accumulator[1][9].z += input_frag[3].z*filter_frag[3].x;                                     
    accumulator[1][9].w += input_frag[3].w*filter_frag[3].x;
  
    accumulator[1][10].x += input_frag[2].x*filter_frag[3].y;
    accumulator[1][10].y += input_frag[2].y*filter_frag[3].y;                                
    accumulator[1][10].z += input_frag[2].z*filter_frag[3].y;                                
    accumulator[1][10].w += input_frag[2].w*filter_frag[3].y;                                  
  
    accumulator[1][11].x += input_frag[3].x*filter_frag[3].y;
    accumulator[1][11].y += input_frag[3].y*filter_frag[3].y;                                 
    accumulator[1][11].z += input_frag[3].z*filter_frag[3].y;                                 
    accumulator[1][11].w += input_frag[3].w*filter_frag[3].y;
                                      
    accumulator[1][12].x += input_frag[2].x*filter_frag[3].z;
    accumulator[1][12].y += input_frag[2].y*filter_frag[3].z;                                     
    accumulator[1][12].z += input_frag[2].z*filter_frag[3].z;                                     
    accumulator[1][12].w += input_frag[2].w*filter_frag[3].z;
                                        
    accumulator[1][13].x += input_frag[3].x*filter_frag[3].z;
    accumulator[1][13].y += input_frag[3].y*filter_frag[3].z;                                     
    accumulator[1][13].z += input_frag[3].z*filter_frag[3].z;                                     
    accumulator[1][13].w += input_frag[3].w*filter_frag[3].z;
  
    accumulator[1][14].x += input_frag[2].x*filter_frag[3].w;
    accumulator[1][14].y += input_frag[2].y*filter_frag[3].w;                                   
    accumulator[1][14].z += input_frag[2].z*filter_frag[3].w;                                   
    accumulator[1][14].w += input_frag[2].w*filter_frag[3].w;
                                        
    accumulator[1][15].x += input_frag[3].x*filter_frag[3].w;
    accumulator[1][15].y += input_frag[3].y*filter_frag[3].w;                                    
    accumulator[1][15].z += input_frag[3].z*filter_frag[3].w;                                    
    accumulator[1][15].w += input_frag[3].w*filter_frag[3].w;
  }

extern "C"
{

__device__ __forceinline__ void  transform_output_tile(float *pOutputs, float2 *C_tile, float2 *At, 
    int round, int c_tensor, int c_glb_offset, int i1, int i2,
    unsigned short mask1, unsigned short mask2, int out_w)
{                     

  c_tensor += (((round)/2)*32 + ((round)%2)*2)*c_glb_offset;
  int x, x1;

  #pragma unroll
  for(int j=0; j<4; j++){

    At[j].x = C_tile[j].x + C_tile[4+j].x + C_tile[8+j].x;
    At[j].y = C_tile[j].y + C_tile[4+j].y + C_tile[8+j].y;

    At[4+j].x = C_tile[4+j].x - C_tile[8+j].x - C_tile[12+j].x;
    At[4+j].y = C_tile[4+j].y - C_tile[8+j].y - C_tile[12+j].y;
    
  }

  #pragma unroll
  for(int i=0; i<2; i++){
    x = i*4;
    x1 = i*((out_w-(out_w%2)) + (out_w%2)/2);

    if(mask1&(1<<(i*2))){
      pOutputs[x1 + c_tensor + i1] = At[x].x + At[x+1].x + At[x+2].x;
    }
    if(mask2&(1<<(i*2))){
      pOutputs[x1 + c_tensor + i2] = At[x].y + At[x+1].y + At[x+2].y;
    }
    if(mask1&(1<<(i*2+1))){
      pOutputs[x1 + c_tensor + i1 + 1] = At[x+1].x - At[x+2].x - At[x+3].x;
    }
    if(mask2&(1<<(i*2+1))){
      pOutputs[x1 + c_tensor + i2 + 1] = At[x+1].y - At[x+2].y - At[x+3].y;
    }
  } 
}

__device__ __forceinline__ unsigned short get_mask(int idd, int tiles_dim_w, int tiles_dim_h, 
         int tw, int th, int out_w, int out_h){

  unsigned short mask = 0x000F;
  // if((blockIdx.y/tiles_dim)==(tiles_dim-1) && out_w%2) mask&=0x0003; // pad bottom row
  // if(!((blockIdx.y+1)%tiles_dim) && out_w%2)           mask&=0X0005; // pad right col
  // if(blockIdx.y==gridDim.y-1 && (idd / tw) == th-1 && out_h%2)  mask&=0x0003; // pad bottom row
  // if(blockIdx.x==gridDim.x-1 && (idd % tw) == tw-1 && out_w%2)  mask&=0X0005; // pad right col
  if(tiles_dim_w % tw == 0 && tiles_dim_h % th == 0){
    if(blockIdx.y==gridDim.y-1 && (idd / tw) == th-1 && out_h%2)  mask&=0x0003; // pad bottom row
    if(blockIdx.x==gridDim.x-1 && (idd % tw) == tw-1 && out_w%2)  mask&=0X0005; // pad right col
  }else if(tiles_dim_w % tw == 0){
    int k = out_h % TH;
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 4*k1 tiles
    if(blockIdx.y==gridDim.y-1 && (idd / tw) == k1-1 && k%2)  mask&=0x0003; // pad bottom row
    if(blockIdx.y==gridDim.y-1 && (idd / tw) > k1-1) mask &= 0x0; //pad all zeros since this tile does not exist
  }else if(tiles_dim_h % th == 0){
    int k = out_w % TW;
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 4*k1 tiles
    if(blockIdx.x==gridDim.x-1 && (idd % tw) == k1-1 && k%2)  mask&=0X0005; // pad right col
    if(blockIdx.x==gridDim.x-1 && (idd % tw) > k1-1)  mask&=0X0; // pad all zeroes
  }else{
    int kh = out_h % TH;
    int kw = out_w % TW;
    int kh1 =  kh % 2 ? (kh+1)/2 : kh/2; // there could be kh1*kw1 tiles
    int kw1 =  kw % 2 ? (kw+1)/2 : kw/2;
    if(blockIdx.y==gridDim.y-1 && (idd / tw) == kh1-1 && kh%2)  mask&=0x0003; // pad bottom row
    if(blockIdx.x==gridDim.x-1 && (idd % tw) == kw1-1 && kw%2)  mask&=0X0005; // pad right col
    if(blockIdx.y==gridDim.y-1 && (idd / tw) > kh1-1)  mask &= 0x0; //pad all zeros since this tile does not exist
    if(blockIdx.x==gridDim.x-1 && (idd % tw) > kw1-1)  mask &= 0X0; // pad all zeroes
  }
  return mask;
}

__device__ __forceinline__ void store_output_tile(float4 acumm_smem[][16], float *shared_mem, float *C, 
int out_h, int out_w, int tiles_dim_w, int tiles_dim_h,  int tw, int th, 
float4 *input_frag_mem, float4* filter_frag_mem){
  
  float2 *output_smem = (float2 *) shared_mem;
  float2 *accumulator = (float2 *) acumm_smem;
  float2 *C_out = (float2*)C;

  float2 *C_tile = (float2*) input_frag_mem;
  float2 *At = (float2*) filter_frag_mem;
  // for output transformation, the role of threadIdx.y changes again:
    // in the main loop, different threadIdx.y deal with different element of the 4x4 tile 
    // here, they are for 4 different groups of lane ids from optSTS64 layout
    // for init (and init+32), we need to identify its tile number (0-31) within the supertile     
    // first, from init, find out from which threadIdx.x it comes.

    // now we got l, which is the land id which computed accumulated sum for the tile element 
    // each lane id (or threadIdx.x) computed 8 tiles which are distributed into 4 locations spreading
    // over the smem. We need to find which of the 8 the current tile is.   
    // use tileid table to figure out    

   // for 2nd tile

  int idd1 = tileid[0][threadIdx.x];
  int id1 = (idd1 % tw) * 2 + (idd1 / tw) * out_w * 2;
  int idd2 = tileid[1][threadIdx.x];
  int id2 = (idd2 % tw) * 2 + (idd2 / tw) * out_w * 2;

  // unsigned short mask1 = 0x000F;
  unsigned short mask1 = get_mask(idd1, tiles_dim_w, tiles_dim_h, tw, th, out_w, out_h);
  unsigned short mask2 = get_mask(idd2, tiles_dim_w, tiles_dim_h, tw, th, out_w, out_h);
  
  // output transpose step
  int t=0;
  int acumm1, acumm2;
  // For transposing
  //acumm1 = access_s_out[Inx]; //* 4
  acumm1 = ((threadIdx.x%8)/2)*34 + threadIdx.x%2 + (threadIdx.x/16)*2 + ((threadIdx.x/8)%2)*8;
  acumm2 = acumm1+4;
                       
  int acumm4 = BN_p*16 ; //*4
  int idx  = threadIdx.y * BN_p;
  int idx2 = idx + BN_p*8; //(BN_p*2 *8)/2

  // For transformating
  int offset = BN_p *2; //*2/2
  int init = ( (threadIdx.y/4)*BN_p*16 + (threadIdx.y%4)*(32+2) ) *2 + threadIdx.x;

  int c_glb_offset = out_h*out_w;
  // int c_tensor = blockIdx.z*c_glb_offset*BK + (blockIdx.y%tiles_dim)*2 + (blockIdx.y/tiles_dim)*out_w*2 + 
  //               blockIdx.x*BN + (threadIdx.x%16)*2+
  //               ((threadIdx.x/16)*16 + (threadIdx.y%4)*4 + threadIdx.y/4)*c_glb_offset;

  int tx = TW, ty = TH;  
  // int c_tile = blockIdx.x * tx  + blockIdx.y * in_w * ty; 
  // int c_tensor = c_tile + (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * in_w * 2 + 
  //               threadIdx.y*(in_h*in_w) - (in_w+1);

  int c_tensor = blockIdx.z*c_glb_offset*BK + blockIdx.x * tx  + blockIdx.y * out_w * ty +
                //  (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * out_w * 2 + 
                 ((threadIdx.x/16)*16 + (threadIdx.y%4)*4 + threadIdx.y/4)*c_glb_offset;

  #pragma unroll                                  
  for(int round=0; round<4; round++){

    *( (float2*) (output_smem + idx + acumm1) )  = *(accumulator+t);
    *( (float2*) (output_smem + idx + acumm1 + 16) )  = *(accumulator+t+1); // float 4, t
    *( (float2*) (output_smem + idx + acumm2) )  = *(accumulator+t+2);
    *( (float2*) (output_smem + idx + acumm2 + 16) )  = *(accumulator+t+3); // float 4, t+1


    *( (float2*) (output_smem + idx2 + acumm1) ) = *(accumulator+t+32);
    *( (float2*) (output_smem + idx2 + acumm1 + 16) ) = *(accumulator+t+33); // float 4, t+16
    *( (float2*) (output_smem + idx2 + acumm2) ) = *(accumulator+t+34);
    *( (float2*) (output_smem + idx2 + acumm2 + 16) ) = *(accumulator+t+35); // float 4, t+17

    // the above 8 float2 will be consumed by theadIdx.y = [0,1,2,3]

    // the following 8 float2 will be consumed by theadIdx.y = [4,5,6,7]

    *( (float2*) (output_smem + idx + acumm4 + acumm1) )  = *(accumulator+t+4); 
    *( (float2*) (output_smem + idx + acumm4 + acumm1 + 16) )  = *(accumulator+t+5); // float 4, t+2
    *( (float2*) (output_smem + idx + acumm4 + acumm2) )  = *(accumulator+t+6);
    *( (float2*) (output_smem + idx + acumm4 + acumm2 + 16) )  = *(accumulator+t+7); // float 4, t+3

    *( (float2*) (output_smem + idx2 + acumm4 + acumm1) ) = *(accumulator+t+36);
    *( (float2*) (output_smem + idx2 + acumm4 + acumm1 + 16) ) = *(accumulator+t+37); // float 4, t+18
    *( (float2*) (output_smem + idx2 + acumm4 + acumm2) ) = *(accumulator+t+38);
    *( (float2*) (output_smem + idx2 + acumm4 + acumm2 + 16) ) = *(accumulator+t+39); // float 4, t+19
    
    

    t+=8;

    __syncthreads();

    
  
    
    for(int i=0; i<16; i++){
      C_tile[i].x = shared_mem[i*offset + init];
      C_tile[i].y = shared_mem[i*offset + init + 32];
    
    }
    

    // transform output tiles    
    transform_output_tile(C, C_tile, At, round, c_tensor, c_glb_offset, id1, id2, mask1, mask2, out_w);
    __syncthreads();
  }
}


// Set of functions per row in Gw product
__device__ float f_row1(float *Gw, int j){
    return Gw[j];
  }
  __device__ float f_row2(float *Gw, int j){
    return 0.5*(Gw[j] + Gw[6+j] + Gw[3+j]);
  }
  __device__ float f_row3(float *Gw, int j){
    return 0.5*(Gw[j] + Gw[6+j] - Gw[3+j]);
  }
  __device__ float f_row4(float *Gw, int j){
    return Gw[6+j];
  }
  // Set of functions per column in GwGt product
  __device__ float f_col1(float *Gw, int j){
    return Gw[j];
  }
  __device__ float f_col2(float *Gw, int j){
    return 0.5*(Gw[j] + Gw[j+2] + Gw[j+1]);
  }
  __device__ float f_col3(float *Gw, int j){
    return 0.5*(Gw[j] + Gw[j+2] - Gw[j+1]);
  }
  __device__ float f_col4(float *Gw, int j){
    return Gw[j+2];
  }
  
  typedef float(*pointFunction_t)(float *, int);
  
  __global__ void FX(const float *pInputs, float *pOutputs, int filt_k, 
                      int filt_c, int filt_h, int filt_w){

    // assumes CHWK layout                    
    int Inx = threadIdx.x, Iny = threadIdx.y;
    int TileX = blockIdx.x, TileY = blockIdx.y;
  
    int c_glb_offset = filt_k*filt_h*filt_w;
    int c_kernel = TileY*BC*c_glb_offset + TileX*BK + Iny*c_glb_offset + Inx;
    int c_glb_offset_s = filt_k*4*4;
    int c_kernel_s = TileY*BC*c_glb_offset_s + TileX*BK + Iny*c_glb_offset_s + Inx;
  
    float Gw[21]; //9+12. In registers
    float *Gw_buffer = Gw+9;
  
    pointFunction_t func1[4] = {f_row1, f_row2, f_row3, f_row4};
    pointFunction_t func2[4] = {f_col1, f_col2, f_col3, f_col4};
  
    for(int bk=0; bk<BK; bk+=blockDim.x){
      for(int i=0; i<9; i++){
        Gw[i] = pInputs[c_kernel + i*filt_k];
      }
  
      int aux;
      for(int i=0; i<4; i++){
        aux = i*3;
        for(int j=0; j<3; j++){
          Gw_buffer[j+aux] = (*func1[i])(Gw, j);
        }
      }
  
      int aux2;
      for(int i=0; i<4; i++){
        aux = i*3; aux2 = i<<2;
        for(int j=0; j<4; j++){
          pOutputs[c_kernel_s+aux2*filt_k+j*filt_k] = (*func2[j])(Gw_buffer, aux);
        }
      }
  
      c_kernel   += blockDim.x;
      c_kernel_s += blockDim.x;
    }
  }

#define d(input, i, j) ( input[(i<<2) + (j)] )

__device__ __forceinline__ void load_and_transform_input_tile(float *Btd, float *pOutputs){

  float workspace[3]; 

  #pragma unroll
  for(int j=0; j<4; j++){
    workspace[0] = Btd[j];
    workspace[1] = Btd[j+4];
    workspace[2] = Btd[j+8];

    Btd[j]    = workspace[0] - workspace[2];
    Btd[j+4]  = workspace[1] + workspace[2];
    Btd[j+8]  = workspace[2] - workspace[1];
    Btd[j+12] = workspace[1] - Btd[j+12];
  }
  
  int c_offset = BC*BN;
  int c_tensor = threadIdx.y*BN + threadIdx.x;
  
  #pragma unroll
  for(int i=0; i<4; i++){ // prefetch 1 input tile/thread
    pOutputs[c_tensor+i*c_offset*4] = d(Btd, i, 0) - d(Btd, i, 2);  
    pOutputs[c_tensor+i*c_offset*4+c_offset] = d(Btd, i, 1) + d(Btd, i, 2);
    pOutputs[c_tensor+i*c_offset*4+2*c_offset] = d(Btd, i, 2) - d(Btd, i, 1);
    pOutputs[c_tensor+i*c_offset*4+3*c_offset] = d(Btd, i, 1) - d(Btd, i, 3);    
  }     

}

__device__ __forceinline__ void load_filter_tile(float *tiles, float *pOutputs, 
                                int filt_c, int filt_k){
 
  int c_tensor_s = threadIdx.y*BK + threadIdx.x;
  int c_offset_s = BK*BC;
  // if(threadIdx.y >= BC) return;
  
  // each thread in row 0 puts its first element of 1st filter tile(loaded by the thread) in smem
  // taking 32 slots 
  // then puts its first element of 2nd filter tile immediately after, taking another 32 slots
  // then followed by threads in row 1, 2.. until 7

  // Note the next element is BK*BC (8*64) slots away, then another BK*BC ....
  // for every 64 values, the first 32 belongs to filter tile 1, the next 32 for filter tile 2 

  for(int k=0; k<2; k++){ // prefetch 2 filter tiles/thread
    for(int i=0; i<4; i++){
      #pragma unroll
      for(int j=0; j<4; j++){
        pOutputs[c_tensor_s + i*c_offset_s*4 + j*c_offset_s] = tiles[k*16 + i*4 + j];
      }
    }
    // 2nd tile right behind the 1st?
    c_tensor_s += BN; // BN has nothing to do with input tiles
  }
  
}

__device__ __forceinline__ void prefetch_filter_tile(const float *pInputs, float *tiles, int filt_k){

  int c_tensor = blockIdx.z*BK + (threadIdx.y*filt_k<<4) + threadIdx.x; // Iny*filt_k*4*4
  // each threadIdx.y corresponds to one channel; there are 8 different threadIdx.y so 8 channels 
  
  //each thread (32 threads in x direction) loads 2 kernel tiles (32 in K direction apart)
  // save the two tiles in a float[32] register, float[16] for each  
  
  int acumm;
  #pragma unroll  
  for(int i=0; i<4; i++){
      acumm = (i*filt_k<<2);
      #pragma unroll
      for(int j=0; j<4; j++){
          tiles[(i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor];
          tiles[16 + (i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor+BN];
      }
  }
}

__device__ __forceinline__ void prefetch_input_tile(const float *pInputs, float *tile, int in_h, 
                       int in_w, int tw, int th, unsigned short mask){
  
  // load one input tile  
  int tx = TW, ty = TH;
  int c_tile = blockIdx.x * tx  + blockIdx.y * in_w * ty; 
  int c_tensor = c_tile + (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * in_w * 2 + 
                threadIdx.y*(in_h*in_w) - (in_w+1);
  
  int acumm,x;
  
           
  if(mask==0xFFFF){
    #pragma unroll
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        tile[(i<<2) + j] = pInputs[acumm + j + c_tensor];        
      }
    }

  } else {
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        x = (i<<2) + j;
        tile[x] = 0.f;
        if(mask&(1<<x))
          tile[x]=pInputs[acumm + j + c_tensor];        
      }
    }
  }
}


// this remains the same as 32x64x8 case
__device__  __forceinline__ void prefetch_filter_frag(float4 *filter_frag, float4 *B_frag, int f_frag_offset, int offset1, int offset2){

  // from the land id table, 32 threads are actually divided into 2 big groups
  // first 16 and the last 16
  // each big group further divides into 8 pairs
  // threads within each pair load the same filter value   

  // the 2nd group just duplicates the 1st

  *((float4*) (filter_frag))     = *(B_frag + offset1); 
  *((float4*) (filter_frag + 1)) = *(B_frag + offset2); // + 32 floats (8 float4)

 // the next 8 floats are for the next next tile element 
  *((float4*) (filter_frag + 2)) = *(B_frag + f_frag_offset + offset1);
  *((float4*) (filter_frag + 3)) = *(B_frag + f_frag_offset + offset2);
}


__device__  __forceinline__ void prefetch_input_frag(float4* input_frag, float4 *A_frag, int frag_offset, int offset1, int offset2){  

  *((float4*) (input_frag))     = *(A_frag + offset1); //ld_shared(A_frag + offset1);
  *((float4*) (input_frag + 1)) = *(A_frag + offset2);

  *((float4*) (input_frag + 2)) = *(A_frag + frag_offset + offset1);
  *((float4*) (input_frag + 3)) = *(A_frag + frag_offset + offset2); //3=2+1
}

__global__ void Winograd_kernel(const float *A, const float *B, float *C,
                    int tiles_dim_w, int tiles_dim_h,
                    int in_c, int in_h, int in_w,
                    int tile_size, int X, int Y,
                    int filt_k, int filt_c,
                    int out_c,
                    int tile_2d_s, int out_h, int out_w){

  extern __shared__ float shared_mem[];
  float *input_smem  = (float*)shared_mem;
  float *filter_smem = (float*)&shared_mem[16*BC*BN];

  unsigned short m = 0xFFFF;  

  if(blockIdx.y==0 && (threadIdx.x / X) == 0)   m &= 0xFFF0;  // pad top row
  if(tiles_dim_w % X == 0 && tiles_dim_h % Y == 0){
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x % X) == X-1) m &= (!(in_w%2))?(0x7777):(0x3333); // pad right col or right 2 cols
  }else if(tiles_dim_w % X == 0){
    int k = in_h % TH; 
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 4*k1 tiles
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x % X) == X-1) m &= (!(in_w%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == k1-1) m &= (!(k%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X > k1-1) m &= 0x0; //pad all zeros since this tile does not exist
  }else if(tiles_dim_h % Y == 0){
    int k = in_w % TW;   
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 8*k1 tiles
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X == k1-1) m &= (!(k%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X > k1-1) m &= 0x0; //pad all zeros since this tile does not exist 
  }else{
    int kh = in_h % TH; 
    int kw = in_w % TW;   
    int kh1 =  kh % 2 ? (kh+1)/2 : kh/2; // there could be kh1*kw1 tiles
    int kw1 =  kw % 2 ? (kw+1)/2 : kw/2; 
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == kh1-1) m &= (!(kh%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X > kh1-1) m &= 0x0; //pad all zeros since this tile does not exist
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X == kw1-1) m &= (!(kw%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X > kw1-1) m &= 0x0; //pad all zeros since this tile does not exist
  }  
  if(blockIdx.x==0 && (threadIdx.x % X) == 0)   m &=0xeeee;  // pad left col
  
  float img_tile[16]; // Prefetch input from GMEM
  float filter_tile[32]; // Prefetch filter from GMEM

  float4 input_frag_mem[8];  //2*2(2*8/4) Data to do Outer Product + prefetch f. SMEM (double_buffer)
  float4 filter_frag_mem[8]; //2*2 Data to do Outer Product + prefetch f. SMEM (double_buffer)
  float4 accumulator[2][16] = {0.0f};  // Accumulators 

  float4 *A_frag; // Input data pointer
  int frag_offset = 2 * (BN*BC); // (2=8/4) SMEM input read offset

  float4 *B_frag; // Filter data pointer  
  int f_frag_offset = 2 * (BC*BK); // (2=8/4 with 4 being float4) SMEM filter read offset 
        

  float4 *input_frag  = (float4*) input_frag_mem;
  float4 *filter_frag = (float4*) filter_frag_mem;

  float4 *swap_filter;
  float4 *swap_input;

  prefetch_input_tile(A, img_tile, in_h, in_w, X, Y, m);
  prefetch_filter_tile(B, filter_tile, filt_k);

  float4 *input_frag_buffer  = (float4*) (input_frag+4);
  float4 *filter_frag_buffer = (float4*) (filter_frag+4);
  
  // Mainloop - iterates over the entire K dimension - not unrolled
  for(int iter=0; iter<in_c; iter+=BC){ // Current iteration

    A_frag = (float4*) (input_smem  + threadIdx.y*BN*BC);
    B_frag = (float4*) (filter_smem + threadIdx.y*BC*BK);

    load_and_transform_input_tile(img_tile, input_smem);
    load_filter_tile(filter_tile, filter_smem, filt_c, filt_k);

    __syncthreads();

    prefetch_input_frag(input_frag, A_frag, frag_offset, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
    prefetch_filter_frag(filter_frag, B_frag, f_frag_offset, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);

    
    #pragma unroll
    for(int i=0; i<BC; i++){

      if(i<(BC-1)){
        A_frag += BN/4;     // This actually moves 32 float (A_frag is float4*)
                          // 32 float is also of size of supertile of one input channel   
        B_frag += BK/4;   // This actually moves 16*4=64 floats (B_frag is float4*), 
                          // 64 floats is also of size of one filter channel 

        prefetch_input_frag(input_frag_buffer, A_frag, frag_offset, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
        prefetch_filter_frag(filter_frag_buffer, B_frag, f_frag_offset, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
      }
     
      outer_product(input_frag, filter_frag, accumulator);

      swap_input = input_frag;
      input_frag = input_frag_buffer;
      input_frag_buffer = swap_input;

      swap_filter = filter_frag;
      filter_frag = filter_frag_buffer;
      filter_frag_buffer = swap_filter;
      
    }
    
    A += BC*in_w*in_h;
    B += filt_k*BC*4*4;

    if(iter<(in_c-BC)){
      prefetch_input_tile(A, img_tile, in_h, in_w, X, Y, m);
      prefetch_filter_tile(B, filter_tile, filt_k);
    }

    __syncthreads();
  }

  // Transpose, transform and store accumulated result
  store_output_tile(accumulator, shared_mem, C, out_h, out_w, tiles_dim_w, tiles_dim_h, X, Y,
                  input_frag_mem, filter_frag_mem);
                     
}

cudaError_t convolutionForward_32Tx64x8(float *k, int in_h, int in_w, float *w, int out_h,
                  int out_w, int out_c, float *C, float *Ww,                 
                int tiles_dim_w, int tiles_dim_h, int tile_size,
                int in_c, int filt_k, int filt_c, int filt_h, int filt_w, int m){

  int tile_2d_s = tile_size*tile_size;
  int smem_size = (16*BN*BC + 16*BC*BK)*4;
  int X = 4, Y = 8;
  

  FX<<<dim3(filt_k/BK, filt_c/BC), dim3(32, BC)>>>(w, Ww, filt_k, filt_c, filt_h, filt_w);
        
  // each thread block will load 32 tiles (4x4) from the single image input
  // we let X*Y = 32 and arbitraraly pick X = 4 and Y = 8
  Winograd_kernel<<<dim3((tiles_dim_w+X-1)/X, (tiles_dim_h+Y-1)/Y, filt_k/BK), dim3(BN, 8), smem_size>>>(k, Ww, C,
  tiles_dim_w, tiles_dim_h, in_c, in_h, in_w, tile_size, X, Y, filt_k, filt_c, out_c, tile_2d_s, out_h, out_w);

  return cudaGetLastError();
}

}


static void conv_winograd_stage0_f32_f32_cuda(        
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,        
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, float * dst,
        cudaStream_t stream) {

    
    int64_t filt_k = src0_ne0;
    int64_t filt_c = src0_ne3;

    FX<<<dim3(filt_k/BK, filt_c/BC), dim3(32, BC)>>>(src0, dst, filt_k, filt_c, src0_ne2, src0_ne1);
    
}

static void conv_winograd_stage1_f32_f32_cuda(int tiles_dim_w, int tiles_dim_h, int X, int Y,   
        int tile_size, int tile_2d_s,    
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    int64_t filt_k = src0_ne0; 
    int64_t in_c   = src1_ne2;
    int64_t in_h   = src1_ne1;
    int64_t in_w   = src1_ne0;
    int64_t filt_c = src0_ne3;
    int64_t out_c  = filt_k;
    int64_t out_h  = in_h;
    int64_t out_w  = in_w;
    int smem_size = (16*BN*BC + 16*BC*BK)*4;

    // printf("A %d, %d\n", filt_k, filt_c);
    // printf("B %d, %d, %d \n", in_c, in_h, in_w);
    // printf("C %d, %d, %d \n", out_c, out_h, out_w);

    Winograd_kernel<<<dim3((tiles_dim_w+X-1)/X, (tiles_dim_h+Y-1)/Y, filt_k/BK), dim3(BN, 8), smem_size>>>(src1, src0, dst,
     tiles_dim_w, tiles_dim_h, in_c, in_h, in_w, tile_size, X, Y, filt_k, filt_c, out_c, tile_2d_s, out_h, out_w);    
}


void ggml_cuda_op_winograd_stage0(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    // const half * src0_d = (const float *)src0->data;
    
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();
    int id = ggml_cuda_get_device();

    // GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
    if (src0->type != GGML_TYPE_F32) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
        GGML_ASSERT(to_fp32_cuda != nullptr);
        int64_t nle = ggml_nelements(src0);
        src0_ddq_as_f32.alloc(nle);
        const half * src0_dd = (const half *)src0->data;
        to_fp32_cuda(src0_dd, src0_ddq_as_f32.get(), nle, stream);
    }

    // GGML_ASSERT(ggml_is_contiguous(src0));
    const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *)src0->data : src0_ddq_as_f32.get();
    
    conv_winograd_stage0_f32_f32_cuda(src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        src0_ddf_i, dst_d, stream);
}



void ggml_cuda_op_winograd_stage1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int m         = 2;
    const int r         = 3;
    const int tile_size = m+r-1; 
    int tiles_dim_w, tiles_dim_h;  
  
    tiles_dim_w = ceil(ceil((double)(src1->ne[0]+2)/2)-1);
    tiles_dim_h = ceil(ceil((double)(src1->ne[1]+2)/2)-1);

    int tile_2d_s = tile_size*tile_size;

    cudaMemcpyToSymbol(access_f_s, aux, 64*sizeof(int));
    cudaMemcpyToSymbol(access_s, aux2, 64*sizeof(int));  
    cudaMemcpyToSymbol(tileid, tid, 64*sizeof(int));
    // printf(" %d, %d, %d \n", tiles_dim_w, tiles_dim_h, tile_size);
    conv_winograd_stage1_f32_f32_cuda(tiles_dim_w, tiles_dim_h, 4, 8, 
        tile_size, tile_2d_s,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        src0_d, src1_d, dst_d, stream);
}


