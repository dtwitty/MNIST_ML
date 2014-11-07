#include "svm_model.hpp"

SVMModel::SVMModel() { svm_.reset(new CvSVM); }

void SVMModel::Train(const cv::Mat& training_vectors,
                     const cv::Mat& training_labels) {
  // Set up SVM parameters
  CvSVMParams params;

  // Using nu svm
  params.svm_type = CvSVM::NU_SVC;
  params.nu = 0.1;
  // Kernel is degree-4 polynomial
  params.kernel_type = CvSVM::POLY;
  params.degree = 4;

  // Stop after 100 iterations or small change threshold
  params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  // Train
  svm_->train(training_vectors, training_labels, cv::Mat(), cv::Mat(), params);

  CvSVMParams trained_params = svm_->get_params();
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
