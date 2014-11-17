#include <iostream>
#include <vector>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "blurred_pixel_vectorizer.hpp"
#include "svm_model.hpp"
#include "nn_model.hpp"
#include "mnist.hpp"
#include "morphological_pre_processor.hpp"
#include "euler_number_extractor.hpp"
#include "pca_model.hpp"
#include "no_pixel_vectorizer.hpp"
#include "deskew_pre_processor.hpp"
#include "knn_model.hpp"

#define MNIST_TRAINING_IMAGE_FILE "../MNIST/train-images-idx3-ubyte"
#define MNIST_TESTING_IMAGE_FILE "../MNIST/t10k-images-idx3-ubyte"
#define MNIST_TRAINING_LABELS_FILE "../MNIST/train-labels-idx1-ubyte"
#define MNIST_TESTING_LABELS_FILE "../MNIST/t10k-labels-idx1-ubyte"

void ImageSetToFeatureSet(const Vectorizer& vectorizer,
                          const std::vector<cv::Mat>& image_set,
                          cv::Mat* feature_set) {
  for (const cv::Mat& image : image_set) {
    cv::Mat row;
    vectorizer.Vectorize(image, &row);
    feature_set->push_back(row);
  }
}

void TrainAndTest(Model& model, const Vectorizer& vectorizer,
                  const std::vector<cv::Mat>& training_images,
                  const cv::Mat& training_labels,
                  const std::vector<cv::Mat>& testing_images,
                  const cv::Mat& testing_labels) {
  cv::Mat training_vectors;
  ImageSetToFeatureSet(vectorizer, training_images, &training_vectors);

  std::cout << "Converted training images to features" << std::endl;

  model.Train(training_vectors, training_labels);

  std::cout << "Trained model" << std::endl;

  cv::Mat testing_vectors;
  ImageSetToFeatureSet(vectorizer, testing_images, &testing_vectors);

  std::cout << "Converted testing images to features" << std::endl;

  cv::Mat predicted_labels;
  model.Predict(testing_vectors, &predicted_labels);
  std::cout << "Prediction finished" << std::endl;

  std::vector<int> failed_indices;

  cv::Mat confusion_matrix = cv::Mat::zeros(10, 10, CV_32S);
  unsigned count = 0, correct = 0;
  for (int i = 0; i < predicted_labels.rows; i++) {
    count += 1;
    int predicted = predicted_labels.at<float>(0, i);
    int actual = testing_labels.at<float>(0, i);
    confusion_matrix.at<unsigned>(actual, predicted) += 1;
    if (predicted == actual) {
      correct += 1;
    } else {
      failed_indices.push_back(i);
    }
  }

  // Create image of failed examples
  int num_failed = failed_indices.size();
  int num_cols = 20;
  int num_rows = (num_failed + num_cols - 1) / num_cols;
  cv::Mat failed_image = cv::Mat::zeros(num_rows * 28, num_cols * 28, CV_8UC1);
  std::cout << failed_image.size() << std::endl;
  int ind = 0;
  for (int row = 0; row < num_rows; row++) {
    for (int col = 0; col < num_cols; col++) {
      if (ind < num_failed) {
        cv::Mat failed = testing_images[failed_indices[ind]];
        cv::Mat roi = failed_image(cv::Rect(col * 28, row * 28, 28, 28));
        failed.copyTo(roi);
      }
      ind++;
    }
  }
  imwrite("Failed.jpg", failed_image);

  std::cout << "Got " << correct << " out of " << count << std::endl;
  std::cout << "Confusion matrix: Row is label, column is prediction"
            << std::endl;

  printf("%7s", "|");
  for (int i = 0; i < 10; i++) {
    printf("%7d", i);
  }
  printf("\n______|");
  for (int i = 0; i < 10; i++) {
    printf("_______");
  }
  printf("\n");
  for (int i = 0; i < 10; i++) {
    printf("%-6d|", i);
    for (int j = 0; j < 10; j++) {
      printf("%7d", confusion_matrix.at<unsigned>(i, j));
    }
    std::cout << std::endl;
  }
}

int main(int argc, char* argv[]) {
  std::vector<cv::Mat> training_images, testing_images;
  cv::Mat training_labels, testing_labels;

  ReadMNISTImages(MNIST_TRAINING_IMAGE_FILE, &training_images);
  ReadMNISTImages(MNIST_TESTING_IMAGE_FILE, &testing_images);

  ReadMNISTLabels(MNIST_TRAINING_LABELS_FILE, &training_labels);
  ReadMNISTLabels(MNIST_TESTING_LABELS_FILE, &testing_labels);

  std::cout << "Finished loading MNIST" << std::endl;

  // Use only the first n test examples (for debugging)
  training_images.resize(60000);
  training_labels.resize(60000);

  testing_images.resize(10000);
  testing_labels.resize(10000);

  {
    std::cout << std::endl << "KNN model" << std::endl;

    PCAModel<KNNModel> model(5);
    NoPixelVectorizer vectorizer;

    TrainAndTest(model, vectorizer, training_images, training_labels,
                 testing_images, testing_labels);
  }
}
