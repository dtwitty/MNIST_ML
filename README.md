MNIST_ML
========

CS 4780 final project code

Requirements:
```
  opencv
  cmake
```

To build:
```
  mkdir build
  cd build
  cmake ..
  make
```

Run cmake again whenever adding a new file.


Code Structure
==============

Most of the logic is in main.cpp, which ties all the parts of our code together.
We've created five general .hpp classes, that represent the flow of how we run
everything.

mnist.hpp: Loads the images from the MNIST dataset file.
pre_processor.hpp: Preprocesses the loaded images.
feature_extractor.hpp: Extracts a desired feature from a preprocessed image.
vectorizer.hpp: Combined multiple features into a single vector.
model.hpp: A classifier to actually train and test vectors outputted by vectorizers.

Right now, we use NoPixelVectorizer for preprocessing, feature extraction, and vectorizing,
and pass it to the general TrainAndTest function in main.cpp to get everything ready for a
model. Pretty much every other file extends one of the four above, and the name of the file
is pretty clear about what it's doing (i.e. hu_moments_extractor extracts Hu moments).
