#include "neural_network.hpp"
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <stdexcept>

/*
 * FUNCTION: sigmoid
 *
 * PURPOSE:
 * Convert a real-valued input to a bounded activation value.
 *
 * The sigmoid function returns a value greater than 0.0 and less than 1.0.
 * The neural network uses this function after each weighted sum. The bounded
 * result is then passed to the next layer.
 *
 * INPUT:
 * x - Weighted sum of inputs and bias.
 *
 * RETURN:
 * Sigmoid activation value.
 */
static double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/*
 * FUNCTION: sigmoid_deriv
 *
 * PURPOSE: Derivative value used during backpropagation.
 */
static double sigmoid_deriv(double x) {
    return x * (1.0 - x);
}

/*
 * Convert one byte-valued image to floating-point values in the range 0..1.
 *
 * std::vector does not provide whole-vector arithmetic such as
 * `normalized = input / 255.0`. This separate preprocessing pass performs
 * the division once per pixel, outside the matrix-computation loops, and can
 * later be replaced by a vectorized implementation.
 *
 * std::transform(first, last, destination, operation) visits the input
 * range [first, last), applies operation to each input element, and writes
 * each result beginning at destination:
 *
 * - input.begin() is an iterator pointing to the first pixel.
 * - input.end() is the one-past-the-last iterator; it is not dereferenced.
 * - normalized.begin() is the first output position.
 * - The lambda is the unary operation: "unary" means it accepts one input
 *   value. Here it receives one uint8_t pixel and returns pixel / 255.0 as a
 *   double.
 *
 * For example, an input pixel of 128 is written as approximately 0.502 to
 * the corresponding position in normalized. The two ranges must contain
 * enough positions for every result written by the operation.
 *
 * The function is static because it is an implementation helper used only
 * in this source file. File-local linkage prevents other translation units
 * from calling or conflicting with a function having the same name.
 */
static std::vector<double> normalize_input(const std::vector<uint8_t>& input) {
    std::vector<double> normalized(input.size());
    // Copy and convert each pixel: input[i] becomes normalized[i].
    std::transform(input.begin(), input.end(), normalized.begin(),
                   [](uint8_t pixel) {
                       return pixel / 255.0;
                   });
    return normalized;
}
/*
 * CONSTRUCTOR: NeuralNetwork
 *
 * PURPOSE:
 * Allocate the network storage and assign initial parameter values.
 *
 * NETWORK CONFIGURATION:
 * input_size_  - Number of values accepted by the input layer.
 * hidden_size_ - Number of neurons in the hidden layer.
 * output_size_ - Number of values produced by the output layer.
 *
 * Weights and biases are stored in vectors of vectors. Each inner vector is a
 * contiguous array of doubles. The outer vector contains one inner vector for
 * each row of the weight matrix or each element of the bias vector.
 */
NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
    : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size)
{
    /*
     * Use a fixed seed so that repeated program runs start with the same
     * parameters. This supports repeatable tests and comparisons.
     * std::mt19937 is a Mersenne Twister pseudo-random number generator.
     * The 32-bit variant has a period of 2^19937-1, which is more than enough for this application.
     */
    std::mt19937 gen(42);

    /*
     * Initial weights and biases are selected uniformly from -0.1 through
     * 0.1. Small initial values reduce the possibility of saturated sigmoid
     * outputs before training has adjusted the parameters.
     */
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    // W1_ maps input values to hidden neurons.
    /*
     * resize() changes W1_'s size to hidden_size_ rows. Here, 128 means the
     * 128 hidden neurons in the network. For each row, the nested vector
     * constructor creates input_size_ zero-initialized double values; here,
     * 784 means the 784 pixels in one 28 x 28 MNIST image.
     *
     * Example:
     *   hidden_size_ = 128, input_size_ = 784
     *   W1_ = 128 rows x 784 values per row
     *   W1_[0][0] and W1_[127][783] are valid matrix elements.
     */
    W1_.resize(hidden_size_, std::vector<double>(input_size_));
    // b1_ contains one bias for each hidden neuron.
    b1_.resize(hidden_size_);

    // W2_ maps hidden-neuron outputs to output scores.
    W2_.resize(output_size_, std::vector<double>(hidden_size_));
    // b2_ contains one bias for each output score.
    b2_.resize(output_size_);

    /*
     * Assign one independent random value to every trainable parameter.
     *
     * W1_ and W2_ are two-dimensional vectors. The outer loop visits each
     * matrix row, and the inner loop visits each value in that row:
     *
     *   W1_: hidden_size_ x input_size_
     *   W2_: output_size_ x hidden_size_
     *
     * b1_ and b2_ are one-dimensional bias vectors, so each single loop
     * visits and initializes one bias value. dist(gen) produces a double
     * uniformly distributed between -0.1 and 0.1. These initial values are
     * the starting weights and biases; training changes them later.
     */
    for (auto& row : W1_) for (auto& v : row) v = dist(gen);
    for (auto& v : b1_) v = dist(gen);

    for (auto& row : W2_) for (auto& v : row) v = dist(gen);
    for (auto& v : b2_) v = dist(gen);
}

/*
 * FUNCTION: forward
 *
 * PURPOSE:
 * Calculate the output values for one image without changing the network.
 *
 * This forward pass moves data from the input layer,
 * through the hidden layer, to the output layer. No training update occurs in
 * this function.
 *
 * INPUT:
 * input - Image pixels stored as unsigned 8-bit values. For MNIST, the normal
 *         input length is 784 values, representing a 28 by 28 image.
 *
 * RETURN:
 * Vector containing one output value for each configured output class.
 */
std::vector<double> NeuralNetwork::forward(const std::vector<uint8_t>& input) const {
    /*
     * The hidden vector holds the result produced by each hidden neuron.
     * Its length is hidden_size_.
     */
    const std::vector<double> normalized_input = normalize_input(input);
    std::vector<double> hidden(hidden_size_);

    for (int i = 0; i < hidden_size_; ++i) {
        // Start the weighted sum with the bias for hidden neuron i.
        double sum = b1_[i];

        for (int j = 0; j < input_size_; ++j)
            sum += W1_[i][j] * normalized_input[j];

        // Apply the activation function and store the hidden-neuron output.
        hidden[i] = sigmoid(sum);
    }

    /*
     * The output vector holds the score calculated for each output class.
     * With MNIST, output index 0 represents digit 0 and output index 9
     * represents digit 9.
     */
    std::vector<double> output(output_size_);
    for (int i = 0; i < output_size_; ++i) {
        // Start the output weighted sum with the output-layer bias.
        double sum = b2_[i];

        for (int j = 0; j < hidden_size_; ++j)
            sum += W2_[i][j] * hidden[j];

        // Convert the weighted sum to the final output score.
        output[i] = sigmoid(sum);
    }

    return output;
}


/*
 * FUNCTION: train
 *
 * PURPOSE:
 * Adjust the network parameters so that the output values approach the
 * supplied labels.
 *
 * The method implements mini-batch stochastic gradient descent. It processes
 * the training data in batches of 64 samples. Each batch accumulates the
 * gradient contributions from its samples, then updates the weights and
 * biases once per batch. This reduces memory traffic. The samples inside a
 * batch are processed in parallel with thread-local workspaces and gradient
 * accumulators. The model parameters are updated only after all threads have
 * finished the batch.
 */
void NeuralNetwork::train(const std::vector<std::vector<double>>& images,
                          const std::vector<uint8_t>& labels,
                          int epochs, double learning_rate) {
    /*
     * MINI-BATCH STOCHASTIC GRADIENT DESCENT WITH HOISTED PARALLEL REGION
     *
     * The previous version of this function processed one sample, computed
     * its gradient, and immediately updated every weight. That was pure
     * stochastic gradient descent (SGD). It had two problems at this scale:
     *
     *   1. Memory traffic. Each update touches all 100,352 entries of W1_
     *      (128-hidden x 784-input) plus all 128 of b1_, plus all 1,280
     *      of W2_ (10-output x 128-hidden) plus all 10 of b2_. At 8 bytes
     *      per double, that is roughly 795 KB of writes per sample. Across
     *      1,200,000 sample-updates per training run, that is about 954 GB
     *      of DRAM traffic, most of it redundant because the weight matrix
     *      is being overwritten before its cache lines can be fully
     *      exploited.
     *
     *      The two matrices plus their biases account for every trainable
     *      parameter in the network:
     *
     *        W1_      100,352
     *        b1_          128
     *        W2_        1,280
     *        b2_           10
     *        ----------------
     *        total    101,770
     *
     *      At 8 bytes per double, that is 814,160 bytes, or roughly 795 KB.
     *
     *   2. No parallelizable unit. Because sample s+1's forward pass reads
     *      weights that sample s just modified, the outer sample loop is
     *      strictly serial. There is no safe place to update parameters
     *      concurrently.
     *
     * Mini-batch SGD fixes both. The weights are read once per batch (64
     * samples) rather than once per sample, reducing DRAM traffic by 64x.
     * The inner sample loop accumulates the gradient contributions from
     * each sample into a separate buffer. After all samples in the batch
     * are processed, the accumulated gradient is applied to the weights
     * and biases in one update. Samples within a batch do not modify the
     * weights; they only accumulate gradient contributions. That makes
     * the sample loop suitable for OpenMP, provided each thread owns its
     * temporary vectors and its gradient accumulator.
     *
     * The math is unchanged. The gradient for a batch is the arithmetic
     * mean of the per-sample gradients:
     *
     *     grad_W_batch = (1 / B) * sum over b in [1, B] of grad_W_b
     *
     * We implement this by summing unnormalized gradients into accumulator
     * buffers and dividing by the batch size only at the moment of the
     * update. This defers the division to one place instead of doing it
     * inside the accumulation loop.
     *
     * ------------------------------------------------------------------
     * WHY THE PARALLEL REGION IS HOISTED OUTSIDE THE EPOCH LOOP
     * ------------------------------------------------------------------
     *
     * An earlier version placed `#pragma omp parallel` inside the batch
     * loop, which meant that a fresh parallel region and fresh thread-local
     * allocations for the four gradient accumulators were created for
     * every batch. With thousands of batches per epoch and 8 threads,
     * that added up to tens of millions of std::vector allocations
     * across a training run. Each allocation also incurs memory and thread
     * startup overhead. The extra work would reduce the benefit of
     * parallelization.
     *
     * This version hoists `#pragma omp parallel` OUTSIDE the epoch loop.
     * The thread team is created once and persists for the entire run.
     * Thread-local accumulators and working buffers are declared inside
     * the parallel region but before the epoch loop, so they are
     * allocated once per thread and reused across every batch of every
     * epoch. The per-batch cost drops to zeroing (unavoidable) and the
     * merge (small relative to the compute).
     *
     * ------------------------------------------------------------------
     * THE FOUR SYNCHRONIZATION POINTS INSIDE THE BATCH LOOP
     * ------------------------------------------------------------------
     *
     * Correctness depends on getting the order of these four steps right.
     * Each is guarded by a specific OpenMP construct; the barriers that
     * separate them are not optional.
     *
     *   1. Parallel sample loop (`#pragma omp for`)
     *      Each thread processes its assigned samples, accumulating
     *      gradients into its own private buffers. The `for` construct
     *      has an implicit barrier at the end: no thread may proceed
     *      until every thread has finished its samples.
     *
     *   2. Zero the shared accumulators (`#pragma omp single`)
     *      One thread resets grad_W1, grad_W2, grad_b1, grad_b2 to zero.
     *      The `single` construct has an implicit barrier at the end: no
     *      thread may proceed until the zeroing thread has finished.
     *
     *   3. Merge thread-local into shared (`#pragma omp critical`)
     *      Threads take turns adding their local buffers into the shared
     *      accumulators. The `critical` construct does NOT have an
     *      implicit barrier at the end. The critical section only protects
     *      the enclosed code while one thread is executing it.
     *      Without an explicit barrier after the critical section, a
     *      thread that finishes its merge first will proceed directly to
     *      the weight update and consume a partially-merged gradient,
     *      losing the contributions of threads still in the merge. An
     *      explicit `#pragma omp barrier` is required.
     *
     *   4. Apply the averaged gradient (`#pragma omp single`)
     *      One thread performs the parameter update. The `single`
     *      construct has an implicit barrier at the end: no thread may
     *      proceed to the next batch until the update has finished.
     *
     * Step 3's explicit barrier is the one that was missing in an earlier
     * revision of this code, and its absence was the cause of a systematic
     * accuracy drop.
     */
    const int batch_size = 64;
    const size_t num_samples = images.size();

    /*
     * Number of batches. The formula rounds up so that a final partial
     * batch is handled. For 60,000 samples and a batch of 64:
     *
     *     (60000 + 63) / 64 = 60063 / 64 = 938
     *
     * The final batch contains fewer than 64 samples only when the dataset
     * size is not an exact multiple of the batch size. The training loop
     * accounts for this by using `end - start` as the divisor for that
     * batch, not the constant `batch_size`.
     */
    const size_t num_batches = (num_samples + batch_size - 1) / batch_size;

    /*
     * One-hot targets, built once and reused for all epochs. Previously
     * this was rebuilt inside train() because there was only one pass over
     * the data. With multiple epochs, building it once outside the epoch
     * loop saves (N_epochs - 1) rebuilds of a 60,000-element structure.
     */
    std::vector<std::vector<double>> targets(
        labels.size(), std::vector<double>(output_size_, 0.0));
    for (size_t i = 0; i < labels.size(); ++i)
        targets[i][labels[i]] = 1.0;

    /*
     * Shared gradient accumulators. These are written only during the
     * `#pragma omp critical` merge, once per batch, and read during the
     * update step at the bottom of the batch. During the parallel sample
     * loop they are untouched. Each thread accumulates into its own
     * private copy instead.
     *
     * Dimensions match the parameter matrices they accumulate into:
     *
     *     grad_W1 : hidden_size_ x input_size_  (128 x 784)
     *     grad_W2 : output_size_ x hidden_size_ (10 x 128)
     *     grad_b1 : hidden_size_                (128)
     *     grad_b2 : output_size_                (10)
     */
    std::vector<std::vector<double>> grad_W1(
        hidden_size_, std::vector<double>(input_size_));
    std::vector<std::vector<double>> grad_W2(
        output_size_, std::vector<double>(hidden_size_));
    std::vector<double> grad_b1(hidden_size_);
    std::vector<double> grad_b2(output_size_);

    /*
     * ==================================================================
     * HOISTED PARALLEL REGION
     * ==================================================================
     *
     * `#pragma omp parallel` with no `for` clause opens a parallel region
     * whose extent is the block below. Every thread in the team
     * executes the block. Declarations inside the block that are not
     * marked `shared` are private to the executing thread. The four
     * gradient accumulators and the four working buffers declared below
     * are per-thread variables, not shared.
     *
     * Because this region encloses the epoch loop, the thread team is
     * created once and remains alive throughout training. The
     * only per-batch synchronization costs are the four points described
     * above.
     */
#pragma omp parallel
    {
        /*
         * Thread-local gradient accumulators. Each thread sums its own
         * share of the batch's samples into these buffers, then merges
         * them into the shared accumulators at the end of the batch.
         *
         * The shared `grad_*` buffers cannot be used directly because two threads
         * could execute `grad_W1[i][j] += ...` on the same (i, j) at the
         * same time. Floating-point addition is not atomic at the memory
         * level; one thread's read would occur before the other's write
         * and one contribution would be lost. Thread-local buffers make
         * The accumulation loop can therefore proceed without locking.
         *
         * Allocated once per thread, outside the epoch and batch loops,
         * and reused across every iteration. This is the change that
         * distinguishes this version from the previous one.
         */
        std::vector<std::vector<double>> local_W1(
            hidden_size_, std::vector<double>(input_size_, 0.0));
        std::vector<std::vector<double>> local_W2(
            output_size_, std::vector<double>(hidden_size_, 0.0));
        std::vector<double> local_b1(hidden_size_, 0.0);
        std::vector<double> local_b2(output_size_, 0.0);

        /*
         * Thread-local working buffers. These hold intermediate values
         * for one forward/backward pass:
         *
         *     hidden    - activations of the hidden layer (size 128)
         *     output    - activations of the output layer (size 10)
         *     d_output  - error signal at the output layer
         *     d_hidden  - error signal at the hidden layer
         *
         * They must be private per thread because two threads computing
         * different samples would otherwise overwrite each other's
         * intermediates. Allocated once per thread and reused.
         */
        std::vector<double> hidden(hidden_size_);
        std::vector<double> output(output_size_);
        std::vector<double> d_output(output_size_);
        std::vector<double> d_hidden(hidden_size_);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            /*
             * Outer loop: one iteration per batch. For 60,000 samples and
             * a batch size of 64, this is 938 iterations per epoch, 18,760
             * iterations across 20 epochs.
             */
            for (size_t b = 0; b < num_batches; ++b) {
                /*
                 * Compute the sample range for this batch. start is
                 * inclusive; end is exclusive. For all but the last
                 * batch, end - start equals batch_size. For the last
                 * batch, end is clamped to num_samples.
                 */
                const size_t start = b * batch_size;
                const size_t end = std::min(start + (size_t)batch_size, num_samples);
                const size_t current_batch_size = end - start;

                /*
                 * Zero this thread's local accumulators. This is not a
                 * synchronization point. Each thread zeroes its own
                 * memory with no interaction with other threads. Only
                 * the shared `grad_*` buffers require a barrier-protected
                 * zeroing, and that happens below.
                 */
                for (auto& row : local_W1) std::fill(row.begin(), row.end(), 0.0);
                for (auto& row : local_W2) std::fill(row.begin(), row.end(), 0.0);
                std::fill(local_b1.begin(), local_b1.end(), 0.0);
                std::fill(local_b2.begin(), local_b2.end(), 0.0);

                /*
                 * STEP 1: distribute the batch's samples across the
                 * thread team.
                 *
                 * The `#pragma omp for` worksharing construct splits the
                 * iteration range [start, end) into chunks assigned to
                 * each thread. With `schedule(static)` each thread
                 * receives a fixed, roughly equal number of iterations.
                 * Static scheduling is chosen because every sample takes
                 * the same amount of work. There is no need for the
                 * dynamic work-stealing that `schedule(dynamic)`
                 * provides.
                 *
                 * An implicit barrier sits at the end of the `for`. No
                 * thread can reach the zero-accumulators step until every
                 * thread has finished its assigned samples.
                 *
                 * The network weights (W1_, W2_, b1_, b2_) are read-only
                 * inside this loop. The samples do not modify them. This
                 * is what makes the loop parallelizable: no thread writes
                 * to any memory another thread reads or writes, except
                 * the thread-local accumulators, which are private.
                 */
#pragma omp for schedule(static)
                for (size_t s = start; s < end; ++s) {
                    /*
                     * The sample's input is already normalized. A
                     * reference avoids a copy of the 784-element vector.
                     */
                    const std::vector<double>& x = images[s];

                    /*
                     * ----- Forward pass, hidden layer -----
                     *
                     * For each of the 128 hidden neurons:
                     *   z1[i] = sum over j of W1[i][j] * x[j] + b1[i]
                     *   hidden[i] = sigmoid(z1[i])
                     *
                     * The inner loop accumulates the dot product of row i
                     * of W1_ with the input vector x. This is the
                     * dominant cost of the training loop:
                     * 128 * 784 = 100,352 multiply-adds per sample.
                     */
                    for (int i = 0; i < hidden_size_; ++i) {
                        double sum = b1_[i];
                        for (int j = 0; j < input_size_; ++j)
                            sum += W1_[i][j] * x[j];
                        hidden[i] = sigmoid(sum);
                    }

                    /*
                     * ----- Forward pass, output layer -----
                     *
                     * For each of the 10 output neurons:
                     *   z2[i] = sum over j of W2[i][j] * hidden[j] + b2[i]
                     *   output[i] = sigmoid(z2[i])
                     *
                     * This produces the network's current prediction for
                     * this sample, as 10 scores in [0, 1].
                     */
                    for (int i = 0; i < output_size_; ++i) {
                        double sum = b2_[i];
                        for (int j = 0; j < hidden_size_; ++j)
                            sum += W2_[i][j] * hidden[j];
                        output[i] = sigmoid(sum);
                    }

                    /*
                     * ----- Output-layer error signal -----
                     *
                     * delta2[i] = dL/dz2[i]
                     *           = (output[i] - target[i])
                     *             * sigmoid_deriv(output[i])
                     *
                     * The first factor is the derivative of the loss
                     * with respect to the output activation. The second
                     * factor is the derivative of the output activation
                     * with respect to its own pre-activation. Together
                     * they give the gradient at the output layer's
                     * weighted sum.
                     */
                    for (int i = 0; i < output_size_; ++i)
                        d_output[i] = (output[i] - targets[s][i]) * sigmoid_deriv(output[i]);

                    /*
                     * ----- Hidden-layer error signal -----
                     *
                     * delta1[i] = dL/dz1[i]
                     *           = (sum over j of delta2[j] * W2[j][i])
                     *             * sigmoid_deriv(hidden[i])
                     *
                     * The sum propagates the output error back through
                     * the second weight matrix. The multiplication by
                     * the sigmoid derivative of the hidden activation
                     * completes the chain rule at this layer.
                     */
                    for (int i = 0; i < hidden_size_; ++i) {
                        double sum = 0.0;
                        for (int j = 0; j < output_size_; ++j)
                            sum += d_output[j] * W2_[j][i];
                        d_hidden[i] = sum * sigmoid_deriv(hidden[i]);
                    }

                    /*
                     * ----- Accumulate W2 and b2 gradients -----
                     *
                     * For each parameter, the gradient is:
                     *   dL/dW2[i][j] = delta2[i] * hidden[j]
                     *   dL/db2[i]    = delta2[i]
                     *
                     * These are added into the THREAD-LOCAL accumulators.
                     * The shared grad_W2 and grad_b2 are not touched here.
                     */
                    for (int i = 0; i < output_size_; ++i) {
                        for (int j = 0; j < hidden_size_; ++j)
                            local_W2[i][j] += d_output[i] * hidden[j];
                        local_b2[i] += d_output[i];
                    }

                    /*
                     * ----- Accumulate W1 and b1 gradients -----
                     *
                     * For each parameter, the gradient is:
                     *   dL/dW1[i][j] = delta1[i] * x[j]
                     *   dL/db1[i]    = delta1[i]
                     *
                     * The source activation here is the input pixel x[j],
                     * not a hidden activation, because layer 1 connects
                     * the input to the hidden layer. Accumulated into
                     * thread-local buffers, same as above.
                     */
                    for (int i = 0; i < hidden_size_; ++i) {
                        for (int j = 0; j < input_size_; ++j)
                            local_W1[i][j] += d_hidden[i] * x[j];
                        local_b1[i] += d_hidden[i];
                    }
                }
                // Implicit barrier at the end of `omp for`.

                /*
                 * STEP 2: zero the shared accumulators.
                 *
                 * Only one thread needs to perform the fill, hence
                 * `#pragma omp single`. The `single` construct has an
                 * implicit barrier at the end, so no thread can begin
                 * its merge before the zeroing thread has finished and
                 * the zeroed memory is visible to all threads.
                 *
                 * Without this barrier the fast thread could reach the
                 * critical merge and add into a buffer that has not yet
                 * been cleared, corrupting the accumulated gradient.
                 */
#pragma omp single
                {
                    for (auto& row : grad_W1) std::fill(row.begin(), row.end(), 0.0);
                    for (auto& row : grad_W2) std::fill(row.begin(), row.end(), 0.0);
                    std::fill(grad_b1.begin(), grad_b1.end(), 0.0);
                    std::fill(grad_b2.begin(), grad_b2.end(), 0.0);
                }
                // Implicit barrier at the end of `omp single`.

                /*
                 * STEP 3: merge each thread's local accumulators into
                 * the shared ones.
                 *
                 * The `#pragma omp critical` region guarantees that only
                 * one thread executes the enclosed block at a time, so
                 * two threads cannot add into the same shared slot
                 * concurrently. This is the only place in the batch loop
                 * where threads write to shared parameter-gradient
                 * memory. It happens once per batch, not once per sample,
                 * which is an important advantage of batching
                 * design.
                 *
                 * `#pragma omp critical` DOES NOT have an implicit barrier
                 * at the end. Which is bad for business. After the critical
                 * section, threads proceed independently. Without the
                 * explicit `#pragma omp barrier` immediately below, the
                 * first thread out of the critical section would begin
                 * the weight update while other threads are still
                 * merging, and the update would consume a partially-
                 * merged gradient. The result would be incomplete and could
                 * reduce model accuracy without producing an error.
                 */
#pragma omp critical
                {
                    for (int i = 0; i < hidden_size_; ++i) {
                        for (int j = 0; j < input_size_; ++j)
                            grad_W1[i][j] += local_W1[i][j];
                        grad_b1[i] += local_b1[i];
                    }
                    for (int i = 0; i < output_size_; ++i) {
                        for (int j = 0; j < hidden_size_; ++j)
                            grad_W2[i][j] += local_W2[i][j];
                        grad_b2[i] += local_b2[i];
                    }
                }
                /*
                 * Explicit barrier. Every thread must reach this point
                 * before any thread proceeds to the update. This is the
                 * barrier that `#pragma omp critical` does not provide.
                 */
#pragma omp barrier

                /*
                 * STEP 4: apply the averaged gradient.
                 *
                 * Only one thread performs the weight update, hence
                 * `#pragma omp single`. Other threads wait at the
                 * implicit barrier at the end of the single block, then
                 * proceed to the next batch.
                 *
                 * The update could in principle be parallelized across
                 * the parameter rows, but each thread would then write
                 * to a disjoint region of W1_ and W2_. This would be a valid split,
                 * but the update is a small fraction of the batch's cost
                 * and does not justify the extra barrier. A single
                 * thread suffices.
                 *
                 * Accumulated gradients are sums, not averages. To
                 * convert the sum to a mean, divide by the number of
                 * samples that contributed. The division is folded into
                 * the learning rate:
                 *
                 *     scale = learning_rate / current_batch_size
                 *
                 * Each parameter update is then:
                 *
                 *     W[i][j] -= scale * grad_W[i][j]
                 *
                 * mathematically identical to averaging each sample's
                 * gradient first and then multiplying by the learning
                 * rate. Doing it this way avoids a division per sample
                 * inside the hot loop, and performs one division per
                 * batch instead (938 per epoch instead of 60,000).
                 *
                 * current_batch_size is used rather than the constant
                 * batch_size, because the final partial batch may have
                 * fewer than batch_size samples.
                 */
#pragma omp single
                {
                    const double scale =
                        learning_rate / static_cast<double>(current_batch_size);

                    for (int i = 0; i < output_size_; ++i) {
                        for (int j = 0; j < hidden_size_; ++j)
                            W2_[i][j] -= scale * grad_W2[i][j];
                        b2_[i] -= scale * grad_b2[i];
                    }
                    for (int i = 0; i < hidden_size_; ++i) {
                        for (int j = 0; j < input_size_; ++j)
                            W1_[i][j] -= scale * grad_W1[i][j];
                        b1_[i] -= scale * grad_b1[i];
                    }
                }
                // Implicit barrier at the end of `omp single`.
            }
        }
    }
}

/*
 * FUNCTION: predict
 *
 * PURPOSE:
 * Return the class index with the highest output score.
 *
 * The operation used to select the largest value is called argmax. It returns
 * the position of the largest element, not the score itself.
 */
int NeuralNetwork::predict(const std::vector<uint8_t>& image) const {
    // Calculate all class scores using the current network parameters.
    auto out = forward(image);

    /*
     * Find the largest score. std::distance returns the position as a signed
     * difference type. static_cast<int>(...) explicitly converts that
     * compatible integer result to the int type required by predict().
     * static_cast is used here instead of an implicit conversion to make the
     * change of type visible; the output vector is small enough that the
     * position fits in an int.
     */
    return static_cast<int>(std::distance(out.begin(),
                                          std::max_element(out.begin(), out.end())));
}

/*
 * FUNCTION: reset_stats
 *
 * PURPOSE:
 * Remove all accumulated evaluation results.
 *
 * This operation does not change weights, biases, or layer dimensions.
 */
void NeuralNetwork::reset_stats() {
    per_class_stats_.clear();
}

/*
 * FUNCTION: record
 *
 * PURPOSE:
 * Add one prediction result to the statistics for its true class.
 *
 * The map key is the actual class. This produces per-class recall:
 * correct predictions for a class divided by all evaluated samples of that
 * class.
 */
void NeuralNetwork::record(uint8_t actual, uint8_t predicted) {
    // operator[] creates a zero-initialized entry when actual is not present.
    ClassStats& s = per_class_stats_[actual];

    // Every evaluated sample increases the denominator.
    s.total += 1;

    // A matching prediction increases the numerator.
    if (actual == predicted)
        s.correct += 1;
}

/*
 * FUNCTION: stats
 *
 * PURPOSE:
 * Provide read-only access to the accumulated evaluation counters.
 *
 * The reference avoids copying the map. The const return type prevents the
 * caller from changing the map through the returned reference.
 */
const std::map<uint8_t, ClassStats>& NeuralNetwork::stats() const {
    return per_class_stats_;
}

/*
 * FUNCTION: save
 *
 * PURPOSE:
 * Write the network configuration and parameters to a binary file.
 *
 * FILE RECORD ORDER:
 * 1. input_size_
 * 2. hidden_size_
 * 3. output_size_
 * 4. all W1_ rows
 * 5. b1_
 * 6. all W2_ rows
 * 7. b2_
 *
 * The load() function must use the same order and data representation.
 */
void NeuralNetwork::save(const std::string& path) const {
    // Open the destination in binary mode so bytes are written without text conversion.
    std::ofstream out(path, std::ios::binary);

    // Report an unavailable or inaccessible destination to the caller.
    if (!out) throw std::runtime_error("Cannot open for writing: " + path);

    // Write the three dimensions required to reconstruct the array shapes.
    out.write(reinterpret_cast<const char*>(&input_size_),  sizeof(input_size_));
    out.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(hidden_size_));
    out.write(reinterpret_cast<const char*>(&output_size_), sizeof(output_size_));

    // Write each contiguous row of the first weight matrix.
    for (const auto& row : W1_)
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));

    // Write the first bias vector.
    out.write(reinterpret_cast<const char*>(b1_.data()), b1_.size() * sizeof(double));

    // Write each contiguous row of the second weight matrix.
    for (const auto& row : W2_)
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));

    // Write the second bias vector.
    out.write(reinterpret_cast<const char*>(b2_.data()), b2_.size() * sizeof(double));
}

/*
 * FUNCTION: load
 *
 * PURPOSE:
 * Read a file created by save() and replace the current network parameters.
 *
 * CAUTION:
 * The file is a native binary format. It is intended for compatible builds
 * and is not a portable interchange format. The file must contain records in
 * the order documented in save().
 */
void NeuralNetwork::load(const std::string& path) {
    // Open the source in binary mode so the stored bytes are read unchanged.
    std::ifstream in(path, std::ios::binary);

    // Report an unavailable or inaccessible source to the caller.
    if (!in) throw std::runtime_error("Cannot open for reading: " + path);

    // Read the dimensions before allocating the parameter arrays.
    in.read(reinterpret_cast<char*>(&input_size_),  sizeof(input_size_));
    in.read(reinterpret_cast<char*>(&hidden_size_), sizeof(hidden_size_));
    in.read(reinterpret_cast<char*>(&output_size_), sizeof(output_size_));

    /*
     * Allocate arrays with the dimensions read from the file. assign() also
     * removes any parameters belonging to the previous network configuration.
     */
    W1_.assign(hidden_size_, std::vector<double>(input_size_));
    b1_.assign(hidden_size_, 0.0);
    W2_.assign(output_size_, std::vector<double>(hidden_size_));
    b2_.assign(output_size_, 0.0);

    // Read the first weight matrix row by row.
    for (auto& row : W1_)
        in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));

    // Read the first bias vector.
    in.read(reinterpret_cast<char*>(b1_.data()), b1_.size() * sizeof(double));

    // Read the second weight matrix row by row.
    for (auto& row : W2_)
        in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));

    // Read the second bias vector.
    in.read(reinterpret_cast<char*>(b2_.data()), b2_.size() * sizeof(double));
}