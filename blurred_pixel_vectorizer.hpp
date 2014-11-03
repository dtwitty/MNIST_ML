/*
 *  Vectorizer that blurs the pixels of the input image and
 *  returns a vector of the resulting pixel data
 */

#ifndef BLURRED_PIXEL_VECTORIZER
#define BLURRED_PIXEL_VECTORIZER

#include <memory>

#include <opencv2/core/core.hpp>

#include "blur_pre_processor.hpp"
#include "pixel_data_feature_extractor.hpp"
#include "vectorizer.hpp"

class BlurredPixelVectorizer : public Vectorizer {
 public:
  BlurredPixelVectorizer(unsigned blur_width, unsigned blur_height) {
    blur_processor_.reset(new BlurPreProcessor(blur_width, blur_height));
    pixel_extractor_.reset(new PixelDataFeatureExtractor);
  }

  void Vectorize(const cv::Mat& input_image, cv::Mat* feature_vector) const {
    // Preprocess by blurring
    cv::Mat blurred_image;
    blur_processor_->PreProcess(input_image, &blurred_image);

    // Output vector is the pixel data of the blurred image
    pixel_extractor_->ExtractFeatures(blurred_image, feature_vector);
  }

 private:
  std::unique_ptr<BlurPreProcessor> blur_processor_;
  std::unique_ptr<PixelDataFeatureExtractor> pixel_extractor_;
};

#endif  // BLURRED_PIXEL_VECTORIZER
