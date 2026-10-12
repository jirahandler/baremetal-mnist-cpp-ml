#include "canvas.hpp"
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>

/**
 * @brief So DeepSeekv4 wrote this method of drawing figures on the terminal, I didn't.
 * The idea is mine though.
 * I don't know stuff well enough to use c++ on a tty/pty,
 * that one is just niche knowledge of (game) devs.
 * so I just used this code to get a canvas to draw on.
 */

static struct termios old_tio;

static void enable_raw_mode() {
    tcgetattr(STDIN_FILENO, &old_tio);
    struct termios new_tio = old_tio;
    new_tio.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
}

static void disable_raw_mode() {
    tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
}

Canvas::Canvas(int width, int height)
    : width_(width), height_(height), pixels_(width * height, 0),
      cursor_x_(0), cursor_y_(0) {}

void Canvas::draw() {
    std::cout << "\033[2J\033[H";
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            if (x == cursor_x_ && y == cursor_y_)
                std::cout << "\033[7m";
            if (pixels_[y * width_ + x] > 0)
                std::cout << "##";
            else
                std::cout << "  ";
            if (x == cursor_x_ && y == cursor_y_)
                std::cout << "\033[0m";
        }
        std::cout << "\n";
    }
    std::cout << "\nArrow keys: move  |  Space: toggle  |  c: clear  |  Enter: classify  |  q: quit\n";
    std::cout.flush();
}

void Canvas::clear() {
    std::fill(pixels_.begin(), pixels_.end(), 0);
}

void Canvas::run() {
    enable_raw_mode();
    draw();

    while (true) {
        char c = getchar();
        if (c == '\033') {
            getchar();
            switch (getchar()) {
                case 'A': if (cursor_y_ > 0) --cursor_y_; break;
                case 'B': if (cursor_y_ < height_ - 1) ++cursor_y_; break;
                case 'C': if (cursor_x_ < width_ - 1) ++cursor_x_; break;
                case 'D': if (cursor_x_ > 0) --cursor_x_; break;
            }
        } else if (c == ' ') {
            auto& p = pixels_[cursor_y_ * width_ + cursor_x_];
            p = (p == 0) ? 255 : 0;
        } else if (c == 'c') {
            clear();
        } else if (c == '\n' || c == '\r') {
            break;
        } else if (c == 'q') {
            disable_raw_mode();
            std::cout << "\nGoodbye.\n";
            std::exit(0);
        }
        draw();
    }
    disable_raw_mode();
}