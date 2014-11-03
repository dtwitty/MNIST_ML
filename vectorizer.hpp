/*
 *  Interface for classes turning images into
 *  feature vectors ready for training/testing.
 *  General idea for implementors:
 *    An implementor should take in an image
 *    and return a vector of doubles, which
 *    will then be passed to machine learning
 *    algorithms.
 *    A vectorizer will be the main user of
 *    the PreProcessor and FeatureExtractor
 *    interfaces.
 */

#ifndef VECTORIZER
#define VECTORIZER

#include <opencv2/core/core.hpp>

class Vectorizer {
 public:
  virtual void Vectorize(const cv::Mat& input_image,
                         cv::Mat* feature_vector) const = 0;
};

#endif  // VECTORIZER
