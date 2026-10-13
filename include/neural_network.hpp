#pragma once  // Instructs the compiler to include this header only once per translation unit.

#include <cstdint>  // Provides fixed-width types such as uint8_t; "_t" means type.
#include <map>      // Provides the ordered map used for per-digit statistics.
#include <string>
#include <vector>
/**
 * @brief Running evaluation counters for one digit class.
 *
 * This structure stores the two values needed to calculate
 * recall for a class:
 *
 *     recall = correct / total
 *
 * The class is identified by the key in NeuralNetwork::per_class_stats_; the
 * structure itself stores only the counters.
 */
struct ClassStats {
  int correct = 0;
  int total = 0;
  /**
   * @brief Ternary is frowned upon and regally hated because of readability,
   * but this is a simple case.
   */
  double accuracy() const {
    /*
     * static_cast<double>(correct) explicitly converts the int counter to
     * double before division. static_cast is C++'s explicit conversion
     * operator for compatible types; it documents the intended conversion
     * and lets the compiler reject incompatible conversions. Here it is used
     * to select floating-point division: 3 / 4 as ints is 0, while
     * static_cast<double>(3) / 4 is 0.75.
     */
    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
  }
};

/**
 * @brief Represents a fully connected network with one hidden layer.
 *
 * For MNIST, the application creates a 784 -> 128 -> 10 network:
 *
 * - 784 input values: one normalized value for each 28 x 28 pixel.
 * - 128 hidden values: intermediate features learned from the pixels.
 * - 10 output values: one score for each digit from 0 through 9.
 */
class NeuralNetwork {
public:
    /**
     * @brief Allocates and initializes the network parameters.
     * @param input_size Number of values accepted by the input layer.
     * @param hidden_size Number of neurons in the hidden layer.
     * @param output_size Number of scores produced by the output layer.
     *
     * Construction allocates W1_, b1_, W2_, and b2_ and
     * fills them with initial values before training begins.
     */
    NeuralNetwork(int input_size, int hidden_size, int output_size);

    /**
     * @brief Runs one forward pass through the network.
     * @param input Image pixels represented as unsigned bytes.
     * @return One computed score for each output class.
     *
     * This method normalizes the pixels, multiplies them by
     * W1_, adds b1_, applies the hidden activation, then repeats the affine
     * operation with W2_ and b2_ for the output layer.
     */
    std::vector<double> forward(const std::vector<uint8_t>& input) const;

    /**
     * @brief Updates parameters using mini-batch stochastic gradient descent.
     * @param images Training images, already normalized as double values.
     * @param labels True digit for each image in images.
     * @param epochs Number of complete passes over the training data.
     * @param learning_rate Size of each gradient-based parameter update.
     *
     * Samples are processed in mini-batches. Each sample's forward and
     * backward pass contributes to thread-local gradient accumulators; those
     * accumulators are reduced into a batch gradient, and the weights and
     * biases are updated once per batch. The caller is responsible for
     * converting byte pixels to normalized values in the range 0.0 to 1.0.
     */
    // header
    void train(const std::vector<std::vector<double>>& images,
           const std::vector<uint8_t>& labels,
           int epochs, double learning_rate, int batch_size);

    /**
     * @brief Converts output scores into a predicted class.
     * @param image Image to classify.
     * @return Index of the largest output score.
     *
     * This calls forward() and performs an argmax over the
     * returned scores. It does not change the network parameters.
     *
     */
    int predict(const std::vector<uint8_t>& image) const;

    /**
     * @brief Resets all accumulated evaluation counters.
     *
     * Removes every entry from per_class_stats_, so the
     * next record() call starts a fresh evaluation.
     */
    void reset_stats();

    /**
     * @brief Adds one classification result to the statistics.
     * @param actual True digit associated with the input image.
     * @param predicted Digit returned by predict().
     *
     */
    void record(uint8_t actual, uint8_t predicted);

    /**
     * @brief Provides read-only access to evaluation statistics.
     * @return Ordered map keyed by the true digit class.
     *
     * The reference avoids copying the map. The const qualifier prevents the
     * caller from modifying the network's internal counters through this API.
     * That's the use of the later const.
     */
    const std::map<uint8_t, ClassStats>& stats() const;

    /**
     * @brief Saves the network dimensions, weights, and biases to disk.
     * @param path Destination file path.
     *
     * This serializes the layer sizes followed by the
     * parameter arrays into a binary stream.
     */
    void save(const std::string& path) const;

    /**
     * @brief Loads network dimensions, weights, and biases from disk.
     * @param path Source file path created by save().
     *
     * Well, this reads the dimensions, resizes the parameter
     * arrays, and fills them with the serialized values.
     */
    void load(const std::string& path);

private:
    int input_size_;   // Number of values consumed by the input layer.
    int hidden_size_;  // Number of neurons in the hidden layer.
    int output_size_;  // Number of scores produced by the output layer.

    // The trailing underscore follows a common C++ member-variable convention.
    // It distinguishes data members from constructor parameters and locals:
    // input_size_ is the member; input_size is typically a parameter.

    // W means "weight matrix"; the digit identifies the layer connection.
    // W1_ maps input values to hidden neurons; W2_ maps hidden neurons to
    // output scores. Each inner vector stores one dense row of parameters.
    std::vector<std::vector<double>> W1_, W2_;  // Weight matrices.

    // b means "bias vector"; the digit identifies the layer receiving it.
    std::vector<double> b1_, b2_;  // Bias vectors.

    // The trailing underscore on W1_, W2_, b1_, and b2_ has the same meaning:
    // each name is a private data member, not a local variable.

    // Statistics are keyed by the true class. Therefore stats()[3] answers:
    // "Among samples that were actually 3, how many were classified correctly?"
    // std::map keeps classes ordered for deterministic reporting.
    std::map<uint8_t, ClassStats> per_class_stats_;  // Per-digit counters.
};
