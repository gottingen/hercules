#include <atomic>
#include <chrono>
#include <random>
#include "turbo/format/table.h"
#include <thread>
#include "turbo/random/random.h"

using namespace turbo;
using Row_t = Table::Row_t;
std::atomic_bool keep_running(true);

void waitingForWorkEnterKey() {
    while (keep_running) {
        if (std::cin.get() == 10) {
            keep_running = false;
        }
    }
    return;
}

int main() {
    std::thread tUserInput(waitingForWorkEnterKey);
    std::vector<turbo::color> colors = {turbo::color::red, turbo::color::green,
                                        turbo::color::yellow, turbo::color::blue, turbo::color::magenta,
                                        turbo::color::cyan, turbo::color::white, turbo::color::black,
                                        turbo::color::light_blue, turbo::color::light_cyan, turbo::color::light_green,
                                        turbo::color::light_yellow,
                                        turbo::color::dark_blue, turbo::color::dark_cyan, turbo::color::dark_green,
                                        turbo::color::dark_magenta, turbo::color::dark_red, turbo::color::dark_gray,
                                        turbo::color::light_gray
    };

    std::vector<turbo::terminal_color> terminal_colors = {turbo::terminal_color::red, turbo::terminal_color::green,
                                                          turbo::terminal_color::yellow, turbo::terminal_color::blue,
                                                          turbo::terminal_color::magenta, turbo::terminal_color::cyan,
                                                          turbo::terminal_color::white, turbo::terminal_color::black,
                                                          turbo::terminal_color::bright_blue,
                                                          turbo::terminal_color::bright_cyan,
                                                          turbo::terminal_color::bright_green,
                                                          turbo::terminal_color::bright_yellow};
    size_t lines = 0;
    while (keep_running) {
        Table process_table;
        process_table.add_row(Row_t{" ", "|", " ", " ", " ", " ", " ", "|", " ", "|", " ", " ", " ", " ", " ", " "});
        process_table.add_row(Row_t{" ", "|", " ", "|", " ", "|", " ", "|", " ", "|", " ", " ", " ", " ", " ", " "});
        process_table.add_row(Row_t{"-", "|", "-", "|", " ", "|", " ", "|", " ", "|", " ", " ", " ", " ", " ", " "});
        process_table.add_row(Row_t{" ", "|", " ", "|", " ", "|", " ", "|", "/", "|", " ", " ", " ", " ", " ", " "});
        process_table.add_row(Row_t{" ", "|", " ", "|", " ", "|", " ", "|", " ", "|", " ", " ", " ", "-", "-", " "});
        process_table.add_row(Row_t{" ", "|", " ", "|", " ", "|", " ", "|", " ", "|", "/", "\\", "/", "", " ", "\\"});
        process_table.add_row(Row_t{" ", "|", " ", "|", " ", "|", " ", "|", " ", "|", " ", "|", "|", " ", " ", "|"});
        process_table.add_row(Row_t{" ", "|", "/", "\\", " ", "/", " ", "|", " ", "|", " ", "/", "\\", " ", " ", "/"});
        process_table.add_row(Row_t{" ", "|", " ", " ", "-", " ", " ", "|", " ", "|", "-", " ", " ", "-", "-", " "});
        bool need_to_change = false;
        size_t bc;
        size_t fc;
        for (int j = 0; j < 16; ++j) {
            if (j == 0 || j == 3 || j == 6 || j == 9 || j == 12) {
                need_to_change = true;
            }
            if (need_to_change) {
                bc = turbo::uniform(0ul, terminal_colors.size() - 1);
                fc = turbo::uniform(0ul, terminal_colors.size() - 1);
                need_to_change = false;
            }
            for (size_t i = 0; i < 9; ++i) {
                if (process_table[i][j].get_text() != " ") {

                    //process_table[i][j].set_text(std::string(1, cc));
                    process_table[i][j].format().font_color(bg(terminal_colors[bc]));
                    process_table[i][j].format().font_color(fg(terminal_colors[fc]));
                    process_table[i][j].format().border_left_color(bg(colors[bc]));
                    process_table[i][j].format().border_left_color(fg(colors[bc]));
                    process_table[i][j].format().border_top_color(fg(colors[bc]));
                    process_table[i][j].format().border_top_color(bg(colors[bc]));
                    process_table[i][j].format().border_bottom_color(fg(colors[bc]));
                    process_table[i][j].format().border_bottom_color(bg(colors[bc]));
                    process_table[i][j].format().border_right_color(fg(colors[bc]));
                    process_table[i][j].format().border_right_color(bg(colors[bc]));
                    process_table[i][j].format().corner_top_left_color(fg(colors[bc]));
                    process_table[i][j].format().corner_top_left_color(bg(colors[bc]));
                    process_table[i][j].format().corner_bottom_left_color(fg(colors[bc]));
                    process_table[i][j].format().corner_bottom_left_color(bg(colors[bc]));
                    process_table[i][j].format().corner_bottom_right_color(fg(colors[bc]));
                    process_table[i][j].format().corner_bottom_right_color(bg(colors[bc]));
                    process_table[i][j].format().corner_top_right_color(fg(colors[bc]));
                    process_table[i][j].format().corner_top_right_color(bg(colors[bc]));
                }


            }
        }

        std::cout << process_table << std::endl;
        std::cout << "Press ENTER to exit... table lines: " << process_table.size()<<std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        lines = table_print_line(process_table) + 1;
        turbo::backwards_lines(std::cout, lines);
    }

    turbo::forward_lines(std::cout, lines);
    tUserInput.join();

    return 0;
}
