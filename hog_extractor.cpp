#include "hog_extractor.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#define DEFAULT_SCALE 20
#define DEFAULT_NUM_BINS 16
#define PI 3.1415926535897

HOGExtractor::HOGExtractor() : HOGExtractor(DEFAULT_SCALE, DEFAULT_NUM_BINS) {}

HOGExtractor::HOGExtractor(unsigned int scale_dim, unsigned int num_bins)
    : scale_dim_(scale_dim), num_bins_(num_bins) {}

void HOGExtractor::ExtractFeatures(const cv::Mat& input_image, cv::Mat* extracted_features) {
  // Scale the image 
  cv::Mat resized;
  cv::resize(input_image, resized, cv::Size(scale_dim_, scale_dim_));

  // Take gradient in x and y direction
  cv::Mat gradient_x, gradient_y;
  cv::Sobel(resized, gradient_x, CV_32F, 1, 0);
  cv::Sobel(resized, gradient_y, CV_32F, 0, 1);
  
  // Take magnitude and angles for each pixel
  cv::Mat angs, mags;
  cv::cartToPolar(gradient_x, gradient_y, mags, angs);

  // Quantize angles into (0...num_bins_)
  cv::Mat_<int> bins(num_bins_ * angs / (2 * PI));

  // Take histogram for each sub-region
  cv::Mat_<float> vec(0, 1);
  vec.reserve(4 * num_bins_);
  for (int y_start = 0; y_start < scale_dim_; y_start += scale_dim_ / 2) {
    int y_end = y_start + scale_dim_ / 2;
    for (int x_start = 0; x_start < scale_dim_; x_start += scale_dim_ / 2) {
       int x_end = x_start + scale_dim_ / 2;
       CalculateHistSubRegion(y_start, y_end, x_start, x_end, bins, mags, &vec);
    }
  }

  // Transpose to row vector
  *extracted_features = vec.t();
}

void HOGExtractor::CalculateHistSubRegion(int y_start, int y_end, int x_start, int x_end, const cv::Mat& bins, const cv::Mat& mags, cv::Mat* vec) {

  // Allocate histogram
  cv::Mat hist = cv::Mat::zeros(num_bins_, 1, CV_32F);
  
  // Fill in histogram
  for (int r = y_start; r < y_end; r++) {
    for (int c = x_start; c < x_end; c++) {
      int bin = bins.at<int>(r, c);
      float mag = mags.at<float>(r, c);
      hist.at<float>(bin, 0) += mag;
    }
  }

  // Append to vector
  vec->push_back(hist);
}
  
