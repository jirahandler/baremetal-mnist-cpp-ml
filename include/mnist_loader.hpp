#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <map>

// "using Image =" creates a type alias: Image becomes a readable
// name for std::vector<uint8_t>. Each Image is a flattened MNIST
// image containing 784 pixel values for a standard 28 x 28 image.
using Image = std::vector<uint8_t>;

/**
 * @brief This structure holds the MNIST dataset in memory.
 * You could do this with three separate vectors, but this structure groups the
 * data in a more convenient way for working with the dataset.
 * Convenience is all that matters here; the MNIST dataset is small enough to fit in memory.
 * Also readability is important, and this structure makes the code easier to read.
 * The `by_label` map is not strictly necessary, but it is convenient for evaluating
 * the network's performance on each digit class.
 */
struct MNISTData {
    // Stores one byte-vector for each image. Image is a 28 x 28 MNIST image,
    // so each inner vector normally contains 784 pixel values.
    std::vector<Image> images;

    // Stores one digit label for each image. labels[i] is the true digit for
    // images[i], and uint8_t is sufficient for values from 0 through 255.
    std::vector<uint8_t> labels;

    // Groups image positions by their true digit. For example, by_label[3]
    // contains the indexes of all images whose label is 3.
    std::map<uint8_t, std::vector<size_t>> by_label;

    // Rebuilds by_label from labels after the image and label data are loaded.
    void build_index();
};

MNISTData load_mnist(const std::string& data_dir,
                     const std::string& images_file,
                     const std::string& labels_file);