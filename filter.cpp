#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
    // Učitaj sliku
    cv::Mat image = cv::imread("Togir4.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // Upload slike na grafičku
    cv::cuda::GpuMat gpu_image;
    gpu_image.upload(image);

    // 7x7 filter kernel
    cv::Ptr<cv::cuda::Filter> mean_filter = cv::cuda::createBoxFilter(gpu_image.type(), -1, cv::Size(7, 7));

    // Primjena filtera
    cv::cuda::GpuMat filtered_gpu_image;
    mean_filter->apply(gpu_image, filtered_gpu_image);

    // Rezultat na CPU
    cv::Mat filtered_image;
    filtered_gpu_image.download(filtered_image);


    cv::imwrite("filtered_image.jpg", filtered_image);


    cv::imshow("Original Image", image);
    cv::imshow("Filtered Image", filtered_image);
    cv::waitKey(0);

    return 0;
}
