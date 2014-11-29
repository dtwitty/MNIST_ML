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

  cv::Mat CrossValidate(const cv::Mat& all_vectors, const cv::Mat& all_labels, int leave_out){
    cv::Mat accuracies;
    for(int i = 0; i < all_vectors.rows / leave_out; i++){
      float correct = 0;
      cv::Mat training_set, training_labels, testing_set, testing_labels, predicted_labels;
      // Copy in testing
      for(int r = 0; r < all_vectors.rows; r++){
        if(r >= i * leave_out && r < (i+1) * (leave_out)){
          testing_set.push_back(all_vectors.row(r));
          testing_labels.push_back(all_labels.at<float>(0, r));
        }
        else{
          training_set.push_back(all_vectors.row(r));
          training_labels.push_back(all_labels.at<float>(0, r));
        }
      }

      Train(training_set, training_labels);
      Predict(testing_set, &predicted_labels);
      for(int k = 0; k < predicted_labels.rows; k++){
        int predicted = predicted_labels.at<float>(0, k);
        int actual = testing_labels.at<float>(0, k);
        if (predicted == actual)
          correct += 1.0;
      }
      accuracies.push_back(correct/(float)leave_out);
    }
    return accuracies;
  }
};

#endif  // MODEL
