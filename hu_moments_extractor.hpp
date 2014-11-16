/*
 *  Extract the Hu moments of an image
 */

#ifndef HU_MOMENTS_EXTRACTOR
#define HU_MOMENTS_EXTRACTOR

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "feature_extractor.hpp"

class HuMomentsExtractor : public FeautureExtractor {
 public:
  HuMomentsExtractor();

  void ExtractFeatures(const cv::Mat& input_image, cv::Mat* extracted_features);
};
#endif  // HU_MOMENTS_EXTRACTOR
