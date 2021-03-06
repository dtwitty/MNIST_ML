/*
 *  Interface for image pre-processing classes
 *  General idea for implementors:
 *    An implementor should take in an image to pre-process
 *    and return the pre-processed image.
 *    
 */
#ifndef PRE_PROCESSOR
#define PRE_PROCESSOR

#include <opencv2/core/core.hpp>

class PreProcessor {
 public:
  virtual void PreProcess(const cv::Mat& input_image,
                            cv::Mat* output_image) = 0;
  
  virtual cv::Mat operator() (const cv::Mat& input_image) {
    cv::Mat output_image;
    PreProcess(input_image, &output_image);
    return output_image;
  }
};

#endif  // PRE_PROCESSOR
