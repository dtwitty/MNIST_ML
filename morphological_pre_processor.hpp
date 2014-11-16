/*  Compute morphological transforms of an image
 *  Specifics can be found at:
 *    http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=morphologyex#getstructuringelement
 */

#ifndef MORPHOLOGICAL_PRE_PROCESSOR
#define MORPHOLOGICAL_PRE_PROCESSOR

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pre_processor.hpp"

enum StructuringElementShape {
  RECT = cv::MORPH_RECT,
  ELLIPSE = cv::MORPH_ELLIPSE,
  CROSS = cv::MORPH_CROSS
};

enum MorphologialOperation {
  OPEN = cv::MORPH_OPEN,
  CLOSE = cv::MORPH_CLOSE,
  GRADIENT = cv::MORPH_GRADIENT,
  TOPHAT = cv::MORPH_TOPHAT,
  BLACKHAT = cv::MORPH_BLACKHAT
};

class MorphologicalPreProcessor : public PreProcessor {
 public:
  // Specify kernel size and shape, and the type of operation
  MorphologicalPreProcessor(int size_hor, int size_vert,
                            StructuringElementShape shape,
                            MorphologialOperation operation);

  void PreProcess(const cv::Mat& input_image, cv::Mat* output_image);

 private:
  cv::Mat structuring_element_;
  MorphologialOperation operation_;
};

#endif  // MORPHOLOGICAL_PRE_PROCESSOR
