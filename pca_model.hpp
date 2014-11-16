/*
 * Wraps other models in PCA (principal component analysis)
 * This has the effect of shrinking vectors
 */

#ifndef PCA_MODEL
#define PCA_MODEL

#include <memory>
#include <iostream>

#include <opencv2/core/core.hpp>

#include "model.hpp"

template <class ModelType, class... Args>
class PCAModel : public Model {
 public:
  // Arguments are passed to constructor of ModelType
  PCAModel(Args&&... args) {
    pca_.reset(new cv::PCA);
    model_.reset(new ModelType(args...));
  }

  void Train(const cv::Mat& training_vectors, const cv::Mat& training_labels) {
    // Generate PCA
    *pca_ = pca_->computeVar(training_vectors, cv::Mat(), 0, 0.95);
    std::cout << "Computed PCA" << std::endl;
    cv::Mat projected_training = pca_->project(training_vectors);
    std::cout << "Computed projection with new vector size "
              << projected_training.cols << std::endl;
    model_->Train(projected_training, training_labels);
  }

  void Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels) {
    cv::Mat projected_testing = pca_->project(test_vectors);
    model_->Predict(projected_testing, predicted_labels);
  }

  void Write(const std::string& filename) {
    // NOT IMPLEMENTED
  }

  void Load(const std::string& filename) {
    // NOT IMPLEMENTED
  }

 private:
  std::unique_ptr<cv::PCA> pca_;
  std::unique_ptr<ModelType> model_;
};

#endif  // PCA_MODEL
