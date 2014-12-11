#include "hough_transform_extractor.hpp"
#include <iostream>

HoughTransformExtractor::HoughTransformExtractor(unsigned int numvotes) : numvotes(numvotes) {}

void HoughTransformExtractor::ExtractFeatures(const cv::Mat& input_image, cv::Mat* extracted_features) {
   
  cv::Mat lines;
  cv::HoughLinesP(input_image, lines, 1, CV_PI/180, numvotes, 0, 0);
  cv::Mat circles;
  cv::HoughCircles(input_image, circles, CV_HOUGH_GRADIENT, 1, 1, 100, numvotes);

  *extracted_features = cv::Mat::zeros(input_image.rows, input_image.cols, CV_8UC1);

  int x0, y0, x1, y1, xc, yc, rc;
  //float m;

  for (int r = 0; r < lines.rows; r++) {
        x0 = lines.at<int>(r,0);
        y0 = lines.at<int>(r,1);
        x1 = lines.at<int>(r,2);
        y1 = lines.at<int>(r,3);
        
	extracted_features->at<uchar>(x0,y0,0) += 1;
        extracted_features->at<uchar>(x1,y1,0) += 1;	
  }

  for (int r = 0; r < circles.rows; r++) {
	xc = (int)circles.at<float>(r,0);
        yc = (int)circles.at<float>(r,1);
        rc = (int)circles.at<float>(r,2);
	
        extracted_features->at<uchar>(xc,yc,0) += rc;
  }

}
