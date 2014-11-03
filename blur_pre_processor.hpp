/*
 *  PreProcessor that blurs the input image
 */

#ifndef BLUR_PRE_PROCESSOR
#define BLUR_PRE_PROCESSOR

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pre_processor.hpp"

class BlurPreProcessor : public PreProcessor {
 public:
  /*
   *  Initialize with a kernel size to use
   */
  BlurPreProcessor(unsigned kernel_size_hor, unsigned kernel_size_vert)
      : kernel_size_hor_(kernel_size_hor),
        kernel_size_vert_(kernel_size_vert) {}

  /*
   *  Pre-processing simply applies a blur to the image
   */
  void PreProcess(const cv::Mat& input_image, cv::Mat* output_image) {
    cv::blur(input_image, *output_image,
             cv::Size(kernel_size_hor_, kernel_size_vert_));
  }

 private:
  unsigned kernel_size_hor_, kernel_size_vert_;
};

#endif  // BLUR_PRE_PRCOESSOR
