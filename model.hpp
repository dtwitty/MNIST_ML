/*
 *  Interface for a machine learning model.
 *  General idea for implementors:
 *    Implementors should perform the usual ML tasks of training and testing
 *    given feature vectors (single-row matrix of doubles)
 */

#ifndef MODEL
#define MODEL

#include <opencv2/core/core.hpp>

class Model {
 public:
  /*
   *  training_vectors is one feature vector per row
   *  training_labels is a column vector of labels (int)
   */
  virtual void Train(const cv::Mat& training_vectors,
                     const cv::Mat& training_labels) = 0;

  virtual void Predict(const cv::Mat& test_vectors,
                       cv::Mat* predicted_labels) = 0;

  virtual void Write(const std::string& filename) = 0;

  virtual void Load(const std::string& filename) = 0;
};

#endif  // MODEL
