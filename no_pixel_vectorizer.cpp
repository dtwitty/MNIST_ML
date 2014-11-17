#include "no_pixel_vectorizer.hpp"
#include "pixel_data_feature_extractor.hpp"
#include "deskew_pre_processor.hpp"
#include "blur_pre_processor.hpp"
#include "hog_extractor.hpp"
#include <iostream>

NoPixelVectorizer::NoPixelVectorizer()
    : morph_(new MorphologicalPreProcessor(2, 2, ELLIPSE, CLOSE)),
      euler_(new EulerNumberExtractor(50)),
      hu_(new HuMomentsExtractor()) {}

void NoPixelVectorizer::Vectorize(const cv::Mat& input_image,
                                  cv::Mat* feature_vector) const {
  DeskewPreProcessor deskew;
  cv::Mat deskewed = deskew(input_image);

  BlurPreProcessor blur(2,2);
  cv::Mat blurred = blur(deskewed);

  HOGExtractor hog(20, 16, 2);
  hog.ExtractFeatures(blurred, feature_vector);

  cv::normalize(*feature_vector, *feature_vector);
  /*
  // Close the image
  cv::Mat closed;
  morph_->PreProcess(deskewed, &closed);

  // Extract euler number
  float euler_number = euler_->GetEulerNumber(closed);
  feature_vector->push_back(euler_number);

  // Get Hu Moments
  cv::Mat hu_moments;
  hu_->ExtractFeatures(deskewed, &hu_moments);
  for (int i = 0; i < 7; i++) {
    feature_vector->push_back(float(hu_moments.at<double>(0, i)));
  }
  */
}
