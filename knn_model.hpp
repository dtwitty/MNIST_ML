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
  KNNModel(int k);

  void Train(const cv::Mat& training_vectors, const cv::Mat& training_labels);

  void Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels);

  void Write(const std::string& filename);

  void Load(const std::string& filename);

 private:
  std::unique_ptr<CvKNearest> knn_;
  int k_;
};

#endif  // KNN_MODEL
