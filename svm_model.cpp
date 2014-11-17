#include "svm_model.hpp"

SVMModel::SVMModel() { 
  svm_.reset(new CvSVM);
  params_.svm_type = CvSVM::NU_SVC;
  params_.nu = 0.1;
  params_.kernel_type = CvSVM::POLY;
  params_.degree = 4;
}

SVMModel::SVMModel(CvSVMParams params) {
  svm_.reset(new CvSVM);
  params_ = params;
}


void SVMModel::Train(const cv::Mat& training_vectors,
                     const cv::Mat& training_labels) {

  // Stop after 100 iterations or small change threshold
  params_.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  // Train
  svm_->train(training_vectors, training_labels, cv::Mat(), cv::Mat(), params_);
}

void SVMModel::Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels) {
  svm_->predict(test_vectors, *predicted_labels);
}

void SVMModel::Write(const std::string& filename) {
  svm_->save(filename.c_str());
}

void SVMModel::Load(const std::string& filename) {
  svm_->load(filename.c_str());
}
