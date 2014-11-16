#include "no_pixel_vectorizer.hpp"
#include <iostream>

NoPixelVectorizer::NoPixelVectorizer()
    : morph_(new MorphologicalPreProcessor(2, 2, ELLIPSE, CLOSE)),
      euler_(new EulerNumberExtractor(50)),
      hu_(new HuMomentsExtractor()) {}

void NoPixelVectorizer::Vectorize(const cv::Mat& input_image,
                             cv::Mat* feature_vector) const {
  // Close the image
  cv::Mat closed;
  morph_->PreProcess(input_image, &closed);

  // Extract euler number
  float euler_number = euler_->GetEulerNumber(closed);
  feature_vector->push_back(euler_number);

  // Get Hu Moments
  cv::Mat hu_moments;
  hu_->ExtractFeatures(input_image, &hu_moments);
  for (int i = 0; i < 7; i++) {
    feature_vector->push_back(float(hu_moments.at<double>(0, i)));
  }
  *feature_vector = feature_vector->t();
}
