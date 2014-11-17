/*
 *  Feature extractor that simply returns the pixel values of the
 *  input image as a feature vector.
 */

#ifndef PIXEL_DATA_FEATURE_EXTRACTOR
#define PIXEL_DATA_FEATURE_EXTRACTOR

#include <vector>

#include <opencv2/core/core.hpp>

#include "feature_extractor.hpp"

class PixelDataFeatureExtractor : public FeautureExtractor {
 public:
  void ExtractFeatures(const cv::Mat& input_image, cv::Mat* feature_vector) {
    // Convert to float
    cv::Mat_<float> converted_to_float(input_image);
    // Reshape to have the same number of channels but 1 row
    *feature_vector = converted_to_float.reshape(0, 1);
  }
};

#endif  // PIXEL_DATA_FEATURE_EXTRACTOR
