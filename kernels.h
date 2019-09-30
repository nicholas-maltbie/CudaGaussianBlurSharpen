#ifndef KERNELS_H_
#define KERNELS_H_

#define BLUR_FILTER     1
#define IDENTITY_FILTER 0

void filter (unsigned char* input_image, unsigned char* output_image, int width, int height, int filte_type);


#endif
