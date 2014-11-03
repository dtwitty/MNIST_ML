/*
 *  Model that takes in a vectorizer and uses an :Autoformat
 *  a
 */

#ifndef SVM_MODEL
#define SVM_MODEL

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "model.hpp"
#include "vectorizer.hpp"

class SVMModel : public Model {
 public:
  // This will take ownership of the vectorizer,
  // destroying it when this SVMModel is destroyed
  // The arguments passed to this constructor are passed to
  // the constructor of the vectorizer
  SVMModel() { svm_.reset(new CvSVM); }

  virtual void Train(const cv::Mat& training_vectors,
                     const cv::Mat& training_labels) {
    CvSVMParams params;
    // linear mulit-class SVM with parameter C
    params.svm_type = CvSVM::C_SVC;
    // RBF kernel
    params.kernel_type = CvSVM::RBF;
    // Train with 10-fold cross validation
    svm_->train_auto(training_vectors, training_labels, cv::Mat(), cv::Mat(),
                     params, 10);
    CvSVMParams trained_params = svm_->get_params();
    std::cout << "SVM C: " << trained_params.C << std::endl;
  }

  virtual void Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels) {
    for (int i = 0; i < test_vectors.rows; i++) {
      predicted_labels->push_back(svm_->predict(test_vectors.row(i)));
    }
  }

  virtual void Write(const std::string& filename) {
    svm_->save(filename.c_str());
  }

  virtual void Load(const std::string& filename) {
    svm_->load(filename.c_str());
  }

 private:
  std::unique_ptr<CvSVM> svm_;
};

#endif  // SVM_MODEL
