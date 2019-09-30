#ifndef KERNELS_H_
#define KERNELS_H_

#define IDENTITY_FILTER     10
#define BOX_BLUR_FILTER     20
#define GAUSS_BLUR_FILTER   21

void filter (unsigned char* input_image, unsigned char* output_image, int width, int height, int filte_type, int fsize);


#endif
