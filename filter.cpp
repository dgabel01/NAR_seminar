#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void meanFilter(const uchar* input, uchar* output, int width, int height, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int halfFilterWidth = filterWidth / 2;
        int filterArea = filterWidth * filterWidth;
        float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
        for (int dy = -halfFilterWidth; dy <= halfFilterWidth; ++dy) {
            for (int dx = -halfFilterWidth; dx <= halfFilterWidth; ++dx) {
                int x = min(max(col + dx, 0), width - 1);
                int y = min(max(row + dy, 0), height - 1);
                int idx = (y * width + x) * 3;
                sumB += input[idx];
                sumG += input[idx + 1];
                sumR += input[idx + 2];
            }
        }
        int outputIdx = (row * width + col) * 3;
        output[outputIdx] = static_cast<uchar>(sumB / filterArea);
        output[outputIdx + 1] = static_cast<uchar>(sumG / filterArea);
        output[outputIdx + 2] = static_cast<uchar>(sumR / filterArea);
    }
}

int main(int argc, char** argv) {
    const char* inputFileName = "Togir4.jpg";
    const char* outputFileName = "filtered_image.jpg";

    Mat image = imread(inputFileName, IMREAD_COLOR);
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }

    int height = image.rows;
    int width = image.cols;
    size_t sizeInBytes = image.total() * image.elemSize();
    int sizeMat = static_cast<int>(sizeInBytes);

    printf("slika je veličine %i bytes\n", sizeMat);
    printf("slika je veličine %i x %i\n", height, width);

    uchar *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeMat);
    cudaMalloc((void**)&d_output, sizeMat);

    cudaMemcpy(d_input, image.ptr<uchar>(0), sizeMat, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    meanFilter<<<dimGrid, dimBlock>>>(d_input, d_output, width, height, 7);

    uchar* h_output = (uchar*)malloc(sizeMat);
    cudaMemcpy(h_output, d_output, sizeMat, cudaMemcpyDeviceToHost);

    Mat image_out = Mat(height, width, CV_8UC3, (unsigned char*)h_output);
    imwrite(outputFileName, image_out);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    printf("Filtered image saved as %s\n", outputFileName);

    return 0;
}
