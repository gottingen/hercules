#include "turbo/format/table.h"

using namespace turbo;
using Row_t = Table::Row_t;

int main() {
    Table table;
    table.format().multi_byte_characters(true);

    /*
      This is a story of a bear and
      a wolf, who wandered the
      realms nine to fulfill a promise
      to one before; they walk the
      twilight path, destined to
      discower the truth
      that is to come.
    */

    table.add_row(Row_t{"ᛏᚺᛁᛊ ᛁᛊ ᚨ ᛊᛏᛟᚱy ᛟᚠᚨ ᛒᛖᚨᚱ ᚨᚾᛞ\n"
                        "ᚨ ᚹᛟᛚᚠ, ᚹᚺᛟ ᚹᚨᚾᛞᛖᚱᛖᛞ ᛏᚺᛖ\n"
                        "ᚱᛖᚨᛚᛗᛊ ᚾᛁᚾᛖ ᛏᛟ ᚠᚢᛚᚠᛁᛚᛚ ᚨ ᛈᚱᛟᛗᛁᛊᛖ\n"
                        "ᛏᛟ ᛟᚾᛖ ᛒᛖᚠᛟᚱᛖ; ᛏᚺᛖy ᚹᚨᛚᚲ ᛏᚺᛖ\n"
                        "ᛏᚹᛁᛚᛁᚷᚺᛏ ᛈᚨᛏᚺ, ᛞᛖᛊᛏᛁᚾᛖᛞ ᛏᛟ\n"
                        "ᛞᛁᛊcᛟᚹᛖᚱ ᛏᚺᛖ ᛏᚱᚢᛏᚺ\nᛏᚺᚨᛏ ᛁᛊ ᛏᛟ cᛟᛗᛖ."});

    table.format()
            .font_style(emphasis::bold | emphasis::faint)
            .font_align(FontAlign::center)
            .font_color(fg(color::red))
            .font_color(bg(color::yellow))
                    // Corners
            .corner_top_left("ᛰ")
            .corner_top_right("ᛯ")
            .corner_bottom_left("ᛮ")
            .corner_bottom_right("ᛸ")
            .corner_top_left_color(fg(color::cyan))
            .corner_top_right_color(fg(color::yellow))
            .corner_bottom_left_color(fg(color::green))
            .corner_bottom_right_color(fg(color::red))
                    // Borders
            .border_top("ᛜ")
            .border_bottom("ᛜ")
            .border_left("ᚿ")
            .border_right("ᛆ")
            .border_left_color(fg(color::yellow))
            .border_right_color(fg(color::green))
            .border_top_color(fg(color::cyan))
            .border_bottom_color(fg(color::red));

    std::cout << table << std::endl;
}