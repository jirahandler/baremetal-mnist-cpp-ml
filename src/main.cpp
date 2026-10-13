#include "mnist_loader.hpp"
#include "neural_network.hpp"
#include "canvas.hpp"
#include <iostream>
#include <cstdio>
#include <chrono>
#include "config.hpp"

// AI is responsible for the function below, LOL.
// Resize and center the drawing to match MNIST's preprocessing convention:
// crop to the bounding box, scale so the longest side fits in 20 pixels,
// then place in the 28x28 grid centered with 4 pixels of margin.
static std::vector<uint8_t> preprocess_canvas(const std::vector<uint8_t>& pixels) {
    // Find bounding box of non-zero pixels.
    int min_x = 28, min_y = 28, max_x = -1, max_y = -1;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            if (pixels[y * 28 + x] > 0) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    // Empty canvas: return as-is; predict() will do something, but caller
    // will see a blank image.
    if (max_x < 0) return pixels;

    int w = max_x - min_x + 1;
    int h = max_y - min_y + 1;

    // Scale factor so the longest side becomes 20 pixels.
    double scale = 20.0 / std::max(w, h);
    int new_w = std::max(1, (int)(w * scale));
    int new_h = std::max(1, (int)(h * scale));

    // Output buffer, all zeros.
    std::vector<uint8_t> out(28 * 28, 0);

    // Compute where to place the scaled drawing so it's centered.
    int offset_x = (28 - new_w) / 2;
    int offset_y = (28 - new_h) / 2;

    // Nearest-neighbor sampling from the source bounding box.
    for (int oy = 0; oy < new_h; ++oy) {
        for (int ox = 0; ox < new_w; ++ox) {
            int sx = min_x + ox * w / new_w;
            int sy = min_y + oy * h / new_h;
            out[(offset_y + oy) * 28 + (offset_x + ox)] = pixels[sy * 28 + sx];
        }
    }

    return out;
}
// The control comes back to me.
int main() {
    Config cfg = load_config("config.yaml");

    std::cout << "Loading MNIST data...\n";
    auto train = load_mnist(cfg.data.dir, cfg.data.train_images, cfg.data.train_labels);
    auto test  = load_mnist(cfg.data.dir, cfg.data.test_images,  cfg.data.test_labels);

    train.build_index();

    std::cout << "Training set size: " << train.images.size() << "\n";
    for (const auto& [digit, indices] : train.by_label) {
        std::cout << "  digit " << int(digit) << ": "
                  << indices.size() << " samples\n";
    }

    std::cout << "\nTraining network (this may take a minute or more, panick only if it takes too long or your CPU is on fire)...\n";
    NeuralNetwork net(cfg.network.input_size,
                  cfg.network.hidden_size,
                  cfg.network.output_size);

    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::vector<double>> train_norm(train.images.size());
    for (size_t i = 0; i < train.images.size(); ++i) {
        train_norm[i].resize(cfg.network.input_size);
        for (int j = 0; j < cfg.network.input_size; ++j)
            train_norm[i][j] = train.images[i][j] / 255.0;
    }

    net.train(train_norm, train.labels,
          cfg.training.epochs,
          cfg.training.learning_rate,
          cfg.training.batch_size);

    auto t1 = std::chrono::steady_clock::now();

    std::cout << "Training done in "
              << std::chrono::duration<double>(t1 - t0).count()
              << " seconds.\n\n";

    net.reset_stats();
    for (size_t i = 0; i < test.images.size(); ++i) {
        /*
         * The predict() method returns an int representing the predicted class.
         * We need to convert it to uint8_t for the record() method.
         */
        int pred = net.predict(test.images[i]);
        /*
         * static_cast<uint8_t>(pred) explicitly converts the int returned by
         * predict() to the uint8_t type required by record(). static_cast is
         * used to make this numeric conversion explicit at the interface
         * boundary. predict() returns only 0..9, so the value fits in uint8_t.
         */
        net.record(test.labels[i], static_cast<uint8_t>(pred));
    }

    std::cout << "Per-class accuracy on the test set:\n";
    int overall_correct = 0;
    int overall_total   = 0;

    for (const auto& [digit, s] : net.stats()) {
        std::cout << "  digit " << int(digit) << ": "
                  << s.correct << "/" << s.total
                  << "  (" << (100.0 * s.accuracy()) << "%)\n";
        overall_correct += s.correct;
        overall_total   += s.total;
    }
    std::cout << "  overall: " << overall_correct << "/" << overall_total
              << "  (" << (100.0 * overall_correct / overall_total) << "%)\n\n";

    while (true) {
        Canvas canvas(28, 28);
        std::cout << "Draw a digit (press Enter to classify, q to quit).\n";
        canvas.run();

        int nonzero = 0;
    int first_x = -1, first_y = -1, last_x = -1, last_y = -1;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            if (canvas.pixels()[y * 28 + x] > 0) {
                ++nonzero;
                if (first_x < 0) { first_x = x; first_y = y; }
                last_x = x; last_y = y;
            }
        }
    }
    std::cout << "[debug] set pixels: " << nonzero
          << "  bbox: (" << first_x << "," << first_y << ") to ("
          << last_x << "," << last_y << ")\n";

        std::vector<uint8_t> preprocessed = preprocess_canvas(canvas.pixels());
        int digit = net.predict(preprocessed);
        std::cout << "\n>>> The network predicts: " << digit << " <<<\n";
        std::cout << "Press Enter to draw again, or q to quit.\n";

        char c = getchar();
        if (c == 'q') break;
    }

    return 0;
}