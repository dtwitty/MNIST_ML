#include "euler_number_extractor.hpp"
#include <iostream>

#define DEFAULT_THRESHOLD 127

const unsigned char kernels_Q1[4][4] = {
    {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
const unsigned char kernels_Q2[4][4] = {
    {0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}};
const unsigned char kernels_Q3[2][4] = {{0, 1, 1, 0}, {1, 0, 0, 1}};

EulerNumberExtractor::EulerNumberExtractor(unsigned char threshold)
    : threshold_(threshold) {}

EulerNumberExtractor::EulerNumberExtractor()
    : EulerNumberExtractor(DEFAULT_THRESHOLD) {}

int EulerNumberExtractor::CountKernel(const cv::Mat& input_image,
                                      const unsigned char kernel[4]) {
  int rows = input_image.rows, cols = input_image.cols;
  int count = 0;
  for (int r = 0; r < rows - 1; r++) {
    for (int c = 0; c < cols - 1; c++) {
      bool im_kernel[4];
      im_kernel[0] = input_image.at<unsigned char>(r, c) >= threshold_;
      im_kernel[1] = input_image.at<unsigned char>(r, c + 1) >= threshold_;
      im_kernel[2] = input_image.at<unsigned char>(r + 1, c) >= threshold_;
      im_kernel[3] = input_image.at<unsigned char>(r + 1, c + 1) >= threshold_;
      bool match = true;
      for (int i = 0; i < 4; i++) {
        match &= (im_kernel[i] == kernel[i]);
      }
      if (match) {
        count += 1;
      }
    }
  }
  return count;
}

void EulerNumberExtractor::ExtractFeatures(const cv::Mat& input_image,
                                           cv::Mat* extracted_features) {
  // We use the Pratt algorithm as described by
  // http://www.isical.ac.in/~malay/Papers/Conf/VLSI_2000.pdf
  // We use 4-connectivity
  int c_1 = 0, c_2 = 0, c_3 = 0;
  for (const auto& kernel: kernels_Q1) {
    c_1 += CountKernel(input_image, kernel);
  }
  for (const auto& kernel : kernels_Q2) {
    c_2 += CountKernel(input_image, kernel);
  }
  for (const auto& kernel : kernels_Q3) {
    c_3 += CountKernel(input_image, kernel);
  }
  extracted_features->push_back((c_1 - c_2 + 2 * c_3) / 4);
}

int EulerNumberExtractor::GetEulerNumber(const cv::Mat& input_image) {
  cv::Mat res;
  ExtractFeatures(input_image, &res);
  return res.at<int>(0,0);
}
