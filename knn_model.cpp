#include "knn_model.hpp"

KNNModel::KNNModel() {
	knn_.reset(new CvKNearest);
	max_k = 32;
	regression = false;
}

KNNModel::KNNModel(int maximum_k){
	knn_.reset(new CvKNearest);
	max_k = maximum_k;
	regression = false;
}

KNNModel::KNNModel(int maximum_k, bool is_regression){
	knn_.reset(new CvKNearest);
	max_k = maximum_k;
	regression = is_regression;
}

void KNNModel::SetRegression(bool is_regression){
	regression = is_regression;
}

void KNNModel::SetMaxK(int maximum_k){
	max_k = maximum_k;
}

void KNNModel::Train (const cv::Mat& training_vectors,
                      const cv::Mat& training_labels) {
	knn_->train(training_vectors, training_labels, cv::Mat(), regression, max_k, false);
}

void KNNModel::Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels) {
	for(int i = 1; i <= max_k; i++){
		cv::Mat predictions;
		cv::Mat responses;
		cv::Mat dists;
		knn_->find_nearest(test_vectors, i, predictions, responses, dists);
		predicted_labels->push_back(predictions);
	}
}

void KNNModel::PredictK(const cv::Mat& test_vectors, cv::Mat* predicted_labels, int k) {
	knn_->find_nearest(test_vectors, k, predicted_labels);
}

void KNNModel::Write(const std::string& filename) { knn_->save(filename.c_str()); }

void KNNModel::Load(const std::string& filename) { knn_->load(filename.c_str()); }