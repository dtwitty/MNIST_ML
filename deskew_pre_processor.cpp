#include "deskew_pre_processor.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#define DESKEW_THRESHOLD 1e-2

// Size of output images
#define OUTPUT_DIM 28

DeskewPreProcessor::DeskewPreProcessor() {}

void DeskewPreProcessor::PreProcess(const cv::Mat& input_image,
                               cv::Mat* output_image){
  cv::Moments moments = cv::moments(input_image);
  if (moments.mu02 < DESKEW_THRESHOLD) {
    *output_image = input_image;
    return;
  }

  double skew = moments.mu11 / moments.mu02;
  double transform_mat[2][3] = {{1., skew, -0.5 * skew * OUTPUT_DIM}, {0, 1, 0}};
  cv::Mat M(2, 3, CV_64FC1, &transform_mat);
  cv::warpAffine(input_image, *output_image, M,
                 cv::Size(OUTPUT_DIM, OUTPUT_DIM), 
                 cv::WARP_INVERSE_MAP | cv::INTER_LINEAR);
}
