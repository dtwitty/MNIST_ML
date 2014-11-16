/*
 *  Extracts the euler number of an image.
 *  This is defined to be the number of objects in the image
 *    minus the number of holes.
 *  This process only works on binary images, so we threshold
 *    input images with a controllable parameter.
 */

#ifndef EULER_NUMBER_EXTRACTOR
#define EULER_NUMBER_EXTRACTOR

#include <opencv2/core/core.hpp>

#include "feature_extractor.hpp"

class EulerNumberExtractor : public FeautureExtractor {
 public:
  EulerNumberExtractor();

  // Set minimum pixel value to be considered "white"
  EulerNumberExtractor(unsigned char threshold);

  // Image should be of type unsigned char!!
  void ExtractFeatures(const cv::Mat& input_image, cv::Mat* extracted_features);

 private:
  // Count the number of occurences of a 2 * 2 binary kernel in an image
  int CountKernel(const cv::Mat& input_image, const unsigned char kernel[4]);

  // Binary thresholding parameter
  unsigned char threshold_;
};

#endif  // EULER_NUMBER_EXTRACTOR
