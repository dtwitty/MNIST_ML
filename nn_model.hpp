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
  NNModel();

  void Train(const cv::Mat& training_vectors, const cv::Mat& training_labels);

  void Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels);

  void Write(const std::string& filename);

  void Load(const std::string& filename);

 private:
  std::unique_ptr<CvANN_MLP> nn_;
};

#endif  // NN_MODEL
