/* 
 * Extract the Hough Transform of an image
 */

#ifndef HOUGH_TRANSFORM_EXTRACTOR
#define HOUGH_TRANSFORM_EXTRACTOR

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "feature_extractor.hpp"

class HoughTransformExtractor : public FeautureExtractor {
  public:
   HoughTransformExtractor(unsigned int numvotes);
   void ExtractFeatures(const cv::Mat& input_image, cv::Mat* extracted_features);
   
  private:
    unsigned int numvotes;
   
};
#endif
