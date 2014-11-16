#include "hu_moments_extractor.hpp"

HuMomentsExtractor::HuMomentsExtractor() {}

void HuMomentsExtractor::ExtractFeatures(const cv::Mat& input_image, cv::Mat* extracted_features) {
  cv::Moments moments = cv::moments(input_image);
  cv::HuMoments(moments, *extracted_features);
}

