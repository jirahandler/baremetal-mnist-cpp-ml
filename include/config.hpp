#pragma once
#include <string>

struct Config{
    struct data{
        std::string dir;
        std::string train_images;
        std::string train_labels;
        std::string test_images;
        std::string test_labels;
    }data;

    struct Network{
        int input_size;
        int hidden_size;
        int output_size;
        int seed;
        double weight_init_range;
    }network;

    struct Training{
        int epochs;
        double learning_rate;
        int batch_size;
    }training;

};

Config load_config(const std::string& config_file);