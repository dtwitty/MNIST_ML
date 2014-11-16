#include "morphological_pre_processor.hpp"

MorphologicalPreProcessor::MorphologicalPreProcessor(
    int size_hor, int size_vert, StructuringElementShape shape,
    MorphologialOperation operation) {
  structuring_element_ =
      cv::getStructuringElement(shape, cv::Size(size_hor, size_vert));
  operation_ = operation;
}

void MorphologicalPreProcessor::PreProcess(const cv::Mat& input_image,
                                           cv::Mat* output_image) {
  cv::morphologyEx(input_image, *output_image, operation_,
                   structuring_element_);
}
