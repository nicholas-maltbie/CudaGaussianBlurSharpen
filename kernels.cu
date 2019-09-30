#include "kernels.h"
#include "helpers.h"
#include <iostream>
#include <cmath>

__global__
void identity(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int offset = y * width + x;

    if(x < width && y < height) {
        output_image[offset * 3] = input_image[offset * 3]; 
        output_image[offset * 3 + 1] = input_image[offset * 3 + 1]; 
        output_image[offset * 3 + 2] = input_image[offset * 3 + 2]; 
    }
}

__global__
void gauss_blur(unsigned char* input_image, unsigned char* output_image, int width, int height,
        float* filter, int fsize) {
    
    const int size = 2*fsize+1;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int offset = y * width + x;

    if(x < width && y < height) {
        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;

        float total_filter = 0;
        for(int off_x = -fsize; off_x < fsize+1; ++off_x) {
            for(int off_y = -fsize; off_y < fsize+1; ++off_y) {
                if((x+off_x) < 0 || (x+off_x) > (width - 1) || (y+off_y) < 0 || (y+off_y) > (height - 1)) {
                    continue;
                }
                
                float filter_value = filter[off_x+fsize+(off_y+fsize)*size];
                total_filter += filter_value;

                const int currentoffset = (offset + off_x + off_y * width) * 3;
                output_red += filter_value * input_image[currentoffset]; 
                output_green += filter_value * input_image[currentoffset+1];
                output_blue += filter_value * input_image[currentoffset+2];
            }
        }

        output_image[offset*3] = output_red / total_filter;
        output_image[offset*3+1] = output_green / total_filter;
        output_image[offset*3+2] = output_blue / total_filter;
    }
}

__global__
void box_blur(unsigned char* input_image, unsigned char* output_image, int width, int height, int fsize) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int offset = y * width + x;

    int hits = 0;

    if(x < width && y < height) {
        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;
        for(int off_x = -fsize; off_x < fsize+1; ++off_x) {
            for(int off_y = -fsize; off_y < fsize+1; ++off_y) {
                if((x+off_x) < 0 || (x+off_x) > (width - 1) || (y+off_y) < 0 || (y+off_y) > (height - 1)) {
                    continue;
                }
                
                const int currentoffset = (offset + off_x + off_y * width) * 3;
                output_red += input_image[currentoffset]; 
                output_green += input_image[currentoffset+1];
                output_blue += input_image[currentoffset+2];
                hits++;
            }
        }

        output_image[offset*3] = output_red/hits;
        output_image[offset*3+1] = output_green/hits;
        output_image[offset*3+2] = output_blue/hits;
    }
}


void filter (unsigned char* input_image, unsigned char* output_image, int width, int height, int filter_type, int fsize) {

    unsigned char* dev_input;
    unsigned char* dev_output;
    getError(cudaMalloc( (void**) &dev_input, width*height*3*sizeof(unsigned char)));
    getError(cudaMemcpy( dev_input, input_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice ));
 
    getError(cudaMalloc( (void**) &dev_output, width*height*3*sizeof(unsigned char)));

    dim3 blockDims(128,128,1);
    dim3 gridDims(
        width/blockDims.x + 1, 
        height/blockDims.y + 1, 
        1
    );

    cudaDeviceSynchronize();

    if (filter_type == BOX_BLUR_FILTER) {
        box_blur<<<gridDims, blockDims>>>(dev_input, dev_output, width, height, fsize); 
    }
    else if (filter_type == IDENTITY_FILTER) {
        identity<<<gridDims, blockDims>>>(dev_input, dev_output, width, height); 
    }
    else if (filter_type == GAUSS_BLUR_FILTER) {
        float* filter = gaussianDistance(fsize / 3, fsize);

        /*for (int x = 0; x <= fsize * 2; x++) {
            for (int y = 0; y <= fsize * 2; y++) {
                printf("%f ", filter[x + y * (fsize * 2 + 1)]);
            }
            printf("\n");
        }*/

        float* dev_filter;

        const int size = 2*fsize+1;
        const int filter_elements = size*size;

        getError(cudaMalloc( (void**) &dev_filter, filter_elements*sizeof(float)));
        getError(cudaMemcpy( dev_filter, filter, filter_elements*sizeof(float), cudaMemcpyHostToDevice ));

        gauss_blur<<<gridDims, blockDims>>>(dev_input, dev_output, width, height, dev_filter, fsize); 
    }

    cudaDeviceSynchronize();


    getError(cudaMemcpy(output_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost ));

    getError(cudaFree(dev_input));
    getError(cudaFree(dev_output));

}

