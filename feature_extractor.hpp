/*
 *  Interface for feature extraction classes
 *  General idea for implementors:
 *    An implementor should take in an image
 *    and return a set features (doubles).
 */
#ifndef FEATURE_EXTRACTOR
#define FEATURE_EXTRACTOR

#include <vector>

#include <opencv2/core/core.hpp>

class FeautureExtractor {
 public:
  virtual void ExtractFeatures(const cv::Mat& input_image,
                              cv::Mat* extracted_features) = 0;
};

#endif  // FEATURE_EXTRACTOR
