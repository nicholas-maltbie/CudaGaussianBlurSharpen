#include <iostream>
#include <cstdlib>
#include "lodepng.h"
#include <cuda.h>
#include "kernels.h"
#include <functional>
#include <stdlib.h>


int main(int argc, char** argv) {
    if(argc != 5) {
        printf("Usage: %s img_in.png img_out.png filter_type filter_size\n", argv[0]);
        printf("    img_in.png - path to input image (png format)\n");
        printf("    img_out.png - path to output image (png format)\n");
        printf("    filter_type - type of filter to use, specified by int\n");
        printf("        0 : no filter\n");
        printf("        1 : box_blur\n");
        printf("        2 : gaussian_blur\n");
        printf("    filter_size - if a filter has a size parameter, this is the value\n");
        return 0;
    }

    // Read arguments for input and output files
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    const int filter_type = atoi(argv[3]);
    const int filter_size = atoi(argv[4]);

    // Prepare a vector to accept 
    std::vector<unsigned char> in_image;
    unsigned int width, height;

    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) {
        printf("decoder error %i: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    // Prepare the data
    unsigned char* input_image  = new unsigned char[(in_image.size()*3)/4];
    unsigned char* output_image = new unsigned char[(in_image.size()*3)/4];
    int where = 0;
    for(int i = 0; i < in_image.size(); ++i) {
       if((i+1) % 4 != 0) {
           input_image[where] = in_image.at(i);
           output_image[where] = 255;
           where++;
       }
    }

    int sel_filter = 0;

    if (filter_type == 0) {
        sel_filter = IDENTITY_FILTER;
    }
    else if (filter_type == 1) {
        sel_filter = BOX_BLUR_FILTER;
    }
    else if (filter_type == 2) {
        sel_filter = GAUSS_BLUR_FILTER;
    }
    else {
        printf("Invalid filter type %i selected", filter_type);
        return 1;
    }

    // Run the filter on it
    filter(input_image, output_image, width, height, sel_filter, filter_size); 

    // Prepare data for output
    std::vector<unsigned char> out_image;
    for(int i = 0; i < in_image.size(); ++i) {
        out_image.push_back(output_image[i]);
        if((i+1) % 3 == 0) {
            out_image.push_back(255);
        }
    }
    
    // Output the data
    error = lodepng::encode(output_file, out_image, width, height);

    //if there's an error, display it
    if(error) {
        printf("encoder error %i: %s\n", error, lodepng_error_text(error));
    }

    delete[] input_image;
    delete[] output_image;
    return 0;

}



