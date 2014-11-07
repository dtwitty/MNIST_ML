#include "svm_model.hpp"

SVMModel::SVMModel() { svm_.reset(new CvSVM); }

void SVMModel::Train(const cv::Mat& training_vectors,
                     const cv::Mat& training_labels) {
  CvSVMParams params;
  // linear mulit-class SVM with parameter C
  params.svm_type = CvSVM::C_SVC;
  // RBF kernel
  params.kernel_type = CvSVM::RBF;
  params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);
  // Train with 3-fold cross validation
  svm_->train_auto(training_vectors, training_labels, cv::Mat(), cv::Mat(),
                   params, 3);
  CvSVMParams trained_params = svm_->get_params();
}

void SVMModel::Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels) {
  for (int i = 0; i < test_vectors.rows; i++) {
    predicted_labels->push_back(svm_->predict(test_vectors.row(i)));
  }
}

void SVMModel::Write(const std::string& filename) {
  svm_->save(filename.c_str());
}

void SVMModel::Load(const std::string& filename) {
  svm_->load(filename.c_str());
}
