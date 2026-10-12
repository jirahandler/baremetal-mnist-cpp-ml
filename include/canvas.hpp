#pragma once
#include <vector>
#include <cstdint>

/**
 * @brief So DeepSeekv4 wrote this method of drawing figures on the terminal, I didn't.
 * The idea is mine though.
 * I don't know stuff well enough to use c++ on a tty/pty,
 * that one is just niche knowledge of (game) devs.
 * so I just used this code to get a canvas to draw on.
 */

class Canvas {
public:
    Canvas(int width = 28, int height = 28);
    void run();
    const std::vector<uint8_t>& pixels() const { return pixels_; }

private:
    int width_, height_;
    std::vector<uint8_t> pixels_;
    int cursor_x_, cursor_y_;
    void draw();
    void clear();
};