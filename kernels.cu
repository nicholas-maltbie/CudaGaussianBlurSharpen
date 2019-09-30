#include "kernels.h"
#include "helpers.h"
#include <iostream>
#include <cmath>

__global__
void apply_filter(unsigned char* input_image, unsigned char* output_image, int width, int height,
        float* filter, int fsize) {
    
    const int size = 2*fsize+1;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int offset = y * width + x;

    if(x < width && y < height) {
        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;

        for(int off_x = -fsize; off_x < fsize+1; ++off_x) {
            for(int off_y = -fsize; off_y < fsize+1; ++off_y) {
                int clamp_x = off_x;
                int clamp_y = off_y;
                if((x+off_x) < 0) {
                    clamp_x = -x;
                }
                else if((x+off_x) > (width - 1)) {
                    clamp_x = width - 1 - x;
                }

                if((y+off_y) < 0) {
                    clamp_y = -y;
                }
                else if((y+off_y) > (height - 1)) {
                    clamp_y = height - 1 - y;
                }
                
                float filter_value = filter[off_x+fsize+(off_y+fsize)*size];

                const int currentoffset = (offset + clamp_x + clamp_y * width) * 3;
                output_red += filter_value * input_image[currentoffset]; 
                output_green += filter_value * input_image[currentoffset+1];
                output_blue += filter_value * input_image[currentoffset+2];
            }
        }

        output_image[offset*3] = min(max(output_red, 0.0), 255.0);
        output_image[offset*3+1] = min(max(output_green, 0.0), 255.0);
        output_image[offset*3+2] = min(max(output_blue, 0.0), 255.0);
    }
}


void filter (unsigned char* input_image, unsigned char* output_image, int width, int height, int filter_type, int fsize) {

    unsigned char* dev_input;
    unsigned char* dev_output;
    getError(cudaMalloc( (void**) &dev_input, width*height*3*sizeof(unsigned char)));
    getError(cudaMemcpy( dev_input, input_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice ));
 
    getError(cudaMalloc( (void**) &dev_output, width*height*3*sizeof(unsigned char)));

    dim3 blockDims(16,16,1);
    dim3 gridDims(
        (int)((double)width/blockDims.x) + 1, 
        (int)((double)height/blockDims.y) + 1, 
        1
    );

    cudaDeviceSynchronize();

    float* filter;

    if (filter_type == BOX_BLUR_FILTER) {
        int size = 2 * fsize + 1;
        filter = new float[size * size];
        for (int i = 0; i < size * size; i++) {
            filter[i] = 1.0 / (size * size);
        }
    }
    else if (filter_type == IDENTITY_FILTER) {
        fsize = 0;
        filter = new float[1];
        filter[0] = 1;
    }
    else if (filter_type == GAUSS_BLUR_FILTER) {
        filter = gaussianDistance(fsize / 3, fsize);
    }
    else if (filter_type == SHARPEN_FILTER) {
        fsize = 1;
        filter = new float[9];
        float sharpen[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
        memcpy(filter, sharpen, 9 * sizeof(float));
    }

    float* dev_filter;

    const int size = 2*fsize+1;
    const int filter_elements = size*size;

    getError(cudaMalloc( (void**) &dev_filter, filter_elements*sizeof(float)));
    getError(cudaMemcpy( dev_filter, filter, filter_elements*sizeof(float), cudaMemcpyHostToDevice ));

    delete[] filter;

    apply_filter<<<gridDims, blockDims>>>(dev_input, dev_output, width, height, dev_filter, fsize); 

    cudaDeviceSynchronize();

    getError(cudaMemcpy(output_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost ));

    getError(cudaFree(dev_input));
    getError(cudaFree(dev_output));

}

