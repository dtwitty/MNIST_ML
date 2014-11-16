/*
 *  A vectorizer that does not include any version of the
 *    original pixel data in its output.
 */

#ifndef NO_PIXEL_VECTORIZER
#define NO_PIXEL_VECTORIZER

#include <memory>

#include <opencv2/core/core.hpp>

#include "vectorizer.hpp"
#include "morphological_pre_processor.hpp"
#include "euler_number_extractor.hpp"
#include "hu_moments_extractor.hpp"

class NoPixelVectorizer : public Vectorizer {
 public:
  NoPixelVectorizer();

  void Vectorize(const cv::Mat& input_image, cv::Mat* feature_vector) const;

 private:
  std::unique_ptr<MorphologicalPreProcessor> morph_;
  std::unique_ptr<EulerNumberExtractor> euler_;
  std::unique_ptr<HuMomentsExtractor> hu_;
};

#endif  // NO_PIXEL_VECTORIZER
