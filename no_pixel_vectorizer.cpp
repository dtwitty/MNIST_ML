#include "no_pixel_vectorizer.hpp"
#include "pixel_data_feature_extractor.hpp"
#include "deskew_pre_processor.hpp"
#include "blur_pre_processor.hpp"
#include "hog_extractor.hpp"
#include "hough_transform_extractor.hpp"
#include <iostream>

NoPixelVectorizer::NoPixelVectorizer()
    : morph_(new MorphologicalPreProcessor(2, 2, ELLIPSE, CLOSE)),
      euler_(new EulerNumberExtractor(50)),
      hu_(new HuMomentsExtractor()),
      hough_(new HoughTransformExtractor(15)) {}

void NoPixelVectorizer::Vectorize(const cv::Mat& input_image,
                                  cv::Mat* feature_vector) const {

  // Close the image
  cv::Mat closed;
  morph_->PreProcess(input_image, &closed);

  // Deskew the image
  DeskewPreProcessor deskew;
  cv::Mat deskewed = deskew(closed);

  // Blur the image
  BlurPreProcessor blur(2,2);
  cv::Mat blurred = blur(deskewed);

  // Hough Transform
  //cv::Mat unused;
  //hough_->ExtractFeatures(blurred, &unused);
  //PixelDataFeatureExtractor houghtmp;
  //cv::Mat houghstuff;
  //houghtmp.ExtractFeatures(unused, &houghstuff);
  //houghtmp.ExtractFeatures(unused, feature_vector);

  // Extract euler number
  //float euler_number = euler_->GetEulerNumber(closed);

  // Get Hu Moments
  //cv::Mat hustuff;
  cv::Mat hu_moments;
  hu_->ExtractFeatures(blurred, &hu_moments);
  for (int i = 0; i < 7; i++) {
      //std::cout << "Did this ok!" << std::endl;
      feature_vector->push_back(float(hu_moments.at<double>(0, i)));
  }

  // Get Raw Data
  //cv::Mat rawstuff;
  //PixelDataFeatureExtractor rawdata;
  //rawdata.ExtractFeatures(blurred, feature_vector);
  //rawdata.ExtractFeatures(blurred, &rawstuff);
  //rawstuff = rawstuff.t();
  
  // Get HOG
  //cv::Mat hogstuff;
  //HOGExtractor hog(20, 16, 2);
  //hog.ExtractFeatures(blurred, &hogstuff);
  //hogstuff = hogstuff.t();

  //*feature_vector = feature_vector->t();
  //feature_vector->push_back(hogstuff);
  //feature_vector->push_back(rawstuff);
  //feature_vector->push_back(euler_number);
  //feature_vector->push_back(hustuff);
  //*feature_vector = feature_vector->t();

  cv::normalize(*feature_vector, *feature_vector);

}
