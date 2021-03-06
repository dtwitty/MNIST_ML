#include "nn_model.hpp"

NNModel::NNModel() : NNModel(cv::Mat(1, 1, CV_32S, 300)) {}

NNModel::NNModel(const cv::Mat& layer_sizes)
    : nn_(new CvANN_MLP), layer_sizes_(layer_sizes) {}

void NNModel::Train(const cv::Mat& training_vectors,
                    const cv::Mat& training_labels) {
  // Set up network topology
  int num_hidden_layers = layer_sizes_.rows;
  cv::Mat layers(num_hidden_layers + 2, 1, CV_32S);

  // Input layer is same size as training vector
  layers.at<int>(0, 0) = training_vectors.cols;
  // Output layer must be of size num_classes
  layers.at<int>(num_hidden_layers + 1, 0) = 10;
  for (int i = 0; i < num_hidden_layers; i++) {
    layers.at<int>(i + 1, 0) = layer_sizes_.at<int>(i, 0);
  }
  nn_->create(layers);

  // Expand training set
  // Eg. Label 2 -> 0 0 1 0 0 0 0 0 0 0
  cv::Mat expanded_training;
  for (int i = 0; i < training_labels.rows; i++) {
    cv::Mat row = cv::Mat::zeros(1, 10, CV_32F);
    row.at<float>(0, int(training_labels.at<float>(0, i))) = 1;
    expanded_training.push_back(row);
  }
  std::cout << "Expanded training set" << std::endl;

  // Train the neural network
  nn_->train(training_vectors, expanded_training, cv::Mat());
}

void NNModel::Predict(const cv::Mat& test_vectors, cv::Mat* predicted_labels) {
  // Run prediction
  cv::Mat responses;
  nn_->predict(test_vectors, responses);

  // Prediction result is index of maximum
  for (int i = 0; i < responses.rows; i++) {
    cv::Mat row = responses.row(i);
    float max_seen = -100000;
    int max_index = 0;
    for (int j = 0; j < row.cols; j++) {
      float k = row.at<float>(0, j);
      if (k > max_seen) {
        max_seen = k;
        max_index = j;
      }
    }
    predicted_labels->push_back(float(max_index));
  }
}

void NNModel::Write(const std::string& filename) {
  nn_->save(filename.c_str());
}

void NNModel::Load(const std::string& filename) { nn_->load(filename.c_str()); }
