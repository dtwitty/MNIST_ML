/*
 *  Model implementation based on k-nearest neighbors
 */

#ifndef KNN_MODEL
#define KNN_MODEL

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "model.hpp"
#include "vectorizer.hpp"

class KNNModel : public Model {
 public:
  KNNModel();
  KNNModel(int maximum_k);
  KNNModel(int maximum_k, bool is_regression);

  void SetRegression(bool is_regression);
  void SetMaxK(int maximum_k);

  void Train(const cv::Mat& training_vectors, const cv::Mat& training_labels);

  void Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels);
  void PredictK(const cv::Mat& test_vectors, cv::Mat* predicted_labels, int k);

  void Write(const std::string& filename);

  void Load(const std::string& filename);

 private:
  std::unique_ptr<CvKNearest> knn_;
  int max_k;
  bool regression;
};

#endif  // KNN_MODEL
