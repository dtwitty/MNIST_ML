/*
 *  Model implementation based on neural networks
 */

#ifndef NN_MODEL
#define NN_MODEL

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "model.hpp"
#include "vectorizer.hpp"

class NNModel : public Model {
 public:
  NNModel() { nn_.reset(new CvANN_MLP); }

  virtual void Train(const cv::Mat& training_vectors,
                     const cv::Mat& training_labels) {
    cv::Mat layers(3, 1, CV_32S);
    layers.at<int>(0, 0) = training_vectors.cols;  // input layer
    layers.at<int>(1, 0) = 300;  // hidden layer
    layers.at<int>(2, 0) = 10;  // output layer
    nn_->create(layers);
    cv::Mat expanded_training;
    for (int i = 0; i < training_labels.rows; i++) {
      cv::Mat row = cv::Mat::zeros(1, 10, CV_32F);
      row.at<float>(0, int(training_labels.at<float>(0, i))) = 1;
      expanded_training.push_back(row);
    }
    std::cout << "Expanded training set" << std::endl;
    nn_->train(training_vectors, expanded_training, cv::Mat());
  }

  virtual void Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels) {
    cv::Mat responses;
    nn_->predict(test_vectors, responses);
    for (int i = 0; i < responses.rows; i++) {
      cv::Mat row = responses.row(i);
      float m = 0;
      int mi = 0;
      for (int j = 0; j < row.cols; j++) {
        float k = row.at<float>(0, j);
        if (k > m) {
          m = k;
          mi = j;
        }
      }
      predicted_labels->push_back(float(mi));
    }
  }

  virtual void Write(const std::string& filename) {
    nn_->save(filename.c_str());
  }

  virtual void Load(const std::string& filename) {
    nn_->load(filename.c_str());
  }

 private:
  std::unique_ptr<CvANN_MLP> nn_;
};

#endif  // SVM_MODEL
