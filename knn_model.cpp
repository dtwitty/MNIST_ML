#include "knn_model.hpp"

KNNModel::KNNModel(int k) {
  knn_.reset(new CvKNearest);
  k_ = k;
}

void KNNModel::Train(const cv::Mat& training_vectors,
                     const cv::Mat& training_labels) {
  knn_->train(training_vectors, training_labels, cv::Mat(), false, k_, false);
}

void KNNModel::Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels) {
  knn_->find_nearest(test_vectors, k_, predicted_labels);
}

void KNNModel::Write(const std::string& filename) {
  knn_->save(filename.c_str());
}

void KNNModel::Load(const std::string& filename) {
  knn_->load(filename.c_str());
}
