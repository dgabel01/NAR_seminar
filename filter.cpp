#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>   //biblioteka za obradu slika
 
using namespace cv;  //za korištenje funkcija i klasa bez cv::





__global__ void meanFilter(const uchar* input, uchar* output, int width, int height, int filterWidth) {

    //blockIxx.x indeks trenutnog bloka u gridu, blockDim.x i blockDim.y dimenzije bloka, 
    //threadId.x i threadId.y indeksi trenutnog threada u bloku
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    ////provjera unutar granica slike
    if (col < width && row < height) {

        //radijus djelovanja filtera
        int halfFilterWidth = filterWidth / 2;
        
        //broj piksela radijusu djelovanja filtera
        int filterArea = filterWidth * filterWidth;

        //ukupni za RGB
        float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;

        //prolazi kroz radijus djelovanja trenutnog piksela sa 2 for petlje
        for (int dy = -halfFilterWidth; dy <= halfFilterWidth; ++dy) {

            for (int dx = -halfFilterWidth; dx <= halfFilterWidth; ++dx) {

                //x i y za osiguravanje granica
                int x = min(max(col + dx, 0), width - 1);
                int y = min(max(row + dy, 0), height - 1);

                //indeks za (x,y) poziciju
                int idx = (y * width + x) * 3;

                //zbroj vrijednosti za svaki piksel(RGB)
                sumB += input[idx];
                sumG += input[idx + 1];
                sumR += input[idx + 2];
            }
        }

        //izračunaj mean za svaki RGB dijeleći sumu sa prostorom djelovanja
        int outputIdx = (row * width + col) * 3;
        output[outputIdx] = static_cast<uchar>(sumB / filterArea);
        output[outputIdx + 1] = static_cast<uchar>(sumG / filterArea);
        output[outputIdx + 2] = static_cast<uchar>(sumR / filterArea);
    }
}



int main(int argc, char** argv) {

    //ulazna i izlazna slika
    const char* inputFileName = "Togir4.jpg";
    const char* outputFileName = "filtered_image.jpg";


    //ucitaj ulaznu sliku
    Mat image = imread(inputFileName, IMREAD_COLOR);
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }


    //svojstva ulazne slike
    int height = image.rows;
    int width = image.cols;
    size_t sizeInBytes = image.total() * image.elemSize();
    int sizeMat = static_cast<int>(sizeInBytes);

    printf("slika je veličine %i bytes\n", sizeMat);
    printf("slika je veličine %i x %i\n", height, width);

    //pokazivaci za GPU memoriju, alociraj memoriju na GPU za ulaznu i izlaznu sliku
    uchar *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeMat);
    cudaMalloc((void**)&d_output, sizeMat);

    //kopiraj ulaznu sliku sa CPU na GPU
    cudaMemcpy(d_input, image.ptr<uchar>(0), sizeMat, cudaMemcpyHostToDevice);

    //definiraj dimenzije bloka threadova
    dim3 dimBlock(16, 16);

    //definiraj dimenzija grida blokova
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    //poziv funkcije na GPU
    meanFilter<<<dimGrid, dimBlock>>>(d_input, d_output, width, height, 7);

    //memorija na CPU za izlaznu sliku
    uchar* h_output = (uchar*)malloc(sizeMat);

    //kopiraj izlaznu sliku sa GPU na CPU
    cudaMemcpy(h_output, d_output, sizeMat, cudaMemcpyDeviceToHost);

    //stvori Mat objekt za izlaznu sliku
    Mat image_out = Mat(height, width, CV_8UC3, (unsigned char*)h_output);

    //pohrani kao sliku
    imwrite(outputFileName, image_out);

    //oslobodi memoriju
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    printf("Filtered image saved as %s\n", outputFileName);

    return 0;
}
