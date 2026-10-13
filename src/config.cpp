#include "config.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>


Config load_config(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load config: " + std::string(e.what()));
    }

    Config cfg;
    auto d = root["data"];
    cfg.data.dir          = d["dir"].as<std::string>();
    cfg.data.train_images = d["train_images"].as<std::string>();
    cfg.data.train_labels = d["train_labels"].as<std::string>();
    cfg.data.test_images  = d["test_images"].as<std::string>();
    cfg.data.test_labels  = d["test_labels"].as<std::string>();

    auto n = root["network"];
    cfg.network.input_size        = n["input_size"].as<int>();
    cfg.network.hidden_size       = n["hidden_size"].as<int>();
    cfg.network.output_size       = n["output_size"].as<int>();
    cfg.network.seed              = n["seed"].as<int>();
    cfg.network.weight_init_range = n["weight_init_range"].as<double>();

    auto t = root["training"];
    cfg.training.epochs        = t["epochs"].as<int>();
    cfg.training.learning_rate = t["learning_rate"].as<double>();
    cfg.training.batch_size    = t["batch_size"].as<int>();

    return cfg;
}