/*
 *  Deskews the input image
 */

#ifndef DESKEW_PRE_PROCESSOR
#define DESKEW_PRE_PROCESSOR

#include <opencv2/core/core.hpp>

#include "pre_processor.hpp"

class DeskewPreProcessor : public PreProcessor {
 public:
  DeskewPreProcessor();

  void PreProcess(const cv::Mat& input_image, cv::Mat* output_image);
};

#endif  // DESKEW_PRE_PROCESSOR
