#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

#include "blurred_pixel_vectorizer.hpp"
#include "svm_model.hpp"
#include "mnist.hpp"

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
  std::cout << predicted_labels.size() << " " << testing_labels.size()
            << std::endl;
  unsigned count = 0, correct = 0;
  for (int i = 0; i < predicted_labels.rows; i++) {
    count += 1;
    if (predicted_labels.at<uint8_t>(0, i) ==
        testing_labels.at<uint8_t>(0, i)) {
      correct += 1;
    }
  }
  std::cout << "Got " << correct << " out of " << count << std::endl;
}

int main(int argc, char* argv[]) {
  std::vector<cv::Mat> training_images, testing_images;
  cv::Mat training_labels, testing_labels;

  ReadMNISTImages(MNIST_TRAINING_IMAGE_FILE, &training_images);
  ReadMNISTImages(MNIST_TESTING_IMAGE_FILE, &testing_images);

  ReadMNISTLabels(MNIST_TRAINING_LABELS_FILE, &training_labels);
  ReadMNISTLabels(MNIST_TESTING_LABELS_FILE, &testing_labels);

  std::cout << "Finished loading MNIST" << std::endl;

  SVMModel model;
  BlurredPixelVectorizer vectorizer(1, 1);

  // Use only the first 100 test examples (for debugging)
  training_images.resize(1000);
  training_labels.resize(1000);

  TrainAndTest(model, vectorizer, training_images, training_labels,
               testing_images, testing_labels);
}
