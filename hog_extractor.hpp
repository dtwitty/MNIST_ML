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

  //  Specify image rescale size (default 20),
  //    number of bins (defualt 16),
  //    sub-split degree (n^2 sub-regions) (default 2)
  //   Image rescale should be a multiple of sub_splt
  HOGExtractor(unsigned int scale_dim, unsigned int num_bins,
               unsigned int sub_split);

  void ExtractFeatures(const cv::Mat& input_image, cv::Mat* extracted_features);

 private:
  // Take histogram for a sub-region and append it to the end of
  // the given column vector vec
  void CalculateHistSubRegion(int y_start, int y_end, int x_start, int x_end,
                              const cv::Mat& bins, const cv::Mat& mags,
                              cv::Mat* vec);

  unsigned int scale_dim_;
  unsigned int num_bins_;
  unsigned int sub_split_;
};

#endif  // HOG_EXTRACTOR
