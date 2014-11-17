/*
 *  Histogram-of-oriented-gradients
 */

#ifndef HOG_EXTRACTOR
#define HOG_EXTRACTOR

#include <opencv2/core/core.hpp>

#include "feature_extractor.hpp"

class HOGExtractor : public FeautureExtractor {
 public:
  HOGExtractor();

  //  Specify image rescale size (default 20)
  //    and number of bins (defualt 16)
  //   Image rescale should be a multiple of 2
  HOGExtractor(unsigned int scale_dim, unsigned int num_bins);

  void ExtractFeatures(const cv::Mat& input_image, cv::Mat* extracted_features);
  
 private:
  // Take histogram for a sub-region and append it to the end of 
  // the given column vector vec
  void CalculateHistSubRegion(int y_start, int y_end, int x_start, int x_end, const cv::Mat& bins, const cv::Mat& mags, cv::Mat* vec);
  
  unsigned int scale_dim_;
  unsigned int num_bins_;
};

#endif  // HOG_EXTRACTOR
