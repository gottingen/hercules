// Formatting library for C++ - color support
//
// Copyright (c) 2018 - present, Victor Zverovich and fmt contributors
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_COLOR_H_
#define FMT_COLOR_H_

#include <cstdint>
#include "turbo/format/fmt/format.h"

namespace turbo {

    enum class color : uint32_t {
        alice_blue = 0xF0F8FF,               // rgb(240,248,255)
        antique_white = 0xFAEBD7,            // rgb(250,235,215)
        aqua = 0x00FFFF,                     // rgb(0,255,255)
        aquamarine = 0x7FFFD4,               // rgb(127,255,212)
        azure = 0xF0FFFF,                    // rgb(240,255,255)
        beige = 0xF5F5DC,                    // rgb(245,245,220)
        bisque = 0xFFE4C4,                   // rgb(255,228,196)
        black = 0x000000,                    // rgb(0,0,0)
        blanched_almond = 0xFFEBCD,          // rgb(255,235,205)
        blue = 0x0000FF,                     // rgb(0,0,255)
        blue_violet = 0x8A2BE2,              // rgb(138,43,226)
        brown = 0xA52A2A,                    // rgb(165,42,42)
        burly_wood = 0xDEB887,               // rgb(222,184,135)
        cadet_blue = 0x5F9EA0,               // rgb(95,158,160)
        chartreuse = 0x7FFF00,               // rgb(127,255,0)
        chocolate = 0xD2691E,                // rgb(210,105,30)
        coral = 0xFF7F50,                    // rgb(255,127,80)
        cornflower_blue = 0x6495ED,          // rgb(100,149,237)
        cornsilk = 0xFFF8DC,                 // rgb(255,248,220)
        crimson = 0xDC143C,                  // rgb(220,20,60)
        cyan = 0x00FFFF,                     // rgb(0,255,255)
        dark_blue = 0x00008B,                // rgb(0,0,139)
        dark_cyan = 0x008B8B,                // rgb(0,139,139)
        dark_golden_rod = 0xB8860B,          // rgb(184,134,11)
        dark_gray = 0xA9A9A9,                // rgb(169,169,169)
        dark_green = 0x006400,               // rgb(0,100,0)
        dark_khaki = 0xBDB76B,               // rgb(189,183,107)
        dark_magenta = 0x8B008B,             // rgb(139,0,139)
        dark_olive_green = 0x556B2F,         // rgb(85,107,47)
        dark_orange = 0xFF8C00,              // rgb(255,140,0)
        dark_orchid = 0x9932CC,              // rgb(153,50,204)
        dark_red = 0x8B0000,                 // rgb(139,0,0)
        dark_salmon = 0xE9967A,              // rgb(233,150,122)
        dark_sea_green = 0x8FBC8F,           // rgb(143,188,143)
        dark_slate_blue = 0x483D8B,          // rgb(72,61,139)
        dark_slate_gray = 0x2F4F4F,          // rgb(47,79,79)
        dark_turquoise = 0x00CED1,           // rgb(0,206,209)
        dark_violet = 0x9400D3,              // rgb(148,0,211)
        deep_pink = 0xFF1493,                // rgb(255,20,147)
        deep_sky_blue = 0x00BFFF,            // rgb(0,191,255)
        dim_gray = 0x696969,                 // rgb(105,105,105)
        dodger_blue = 0x1E90FF,              // rgb(30,144,255)
        fire_brick = 0xB22222,               // rgb(178,34,34)
        floral_white = 0xFFFAF0,             // rgb(255,250,240)
        forest_green = 0x228B22,             // rgb(34,139,34)
        fuchsia = 0xFF00FF,                  // rgb(255,0,255)
        gainsboro = 0xDCDCDC,                // rgb(220,220,220)
        ghost_white = 0xF8F8FF,              // rgb(248,248,255)
        gold = 0xFFD700,                     // rgb(255,215,0)
        golden_rod = 0xDAA520,               // rgb(218,165,32)
        gray = 0x808080,                     // rgb(128,128,128)
        green = 0x008000,                    // rgb(0,128,0)
        green_yellow = 0xADFF2F,             // rgb(173,255,47)
        honey_dew = 0xF0FFF0,                // rgb(240,255,240)
        hot_pink = 0xFF69B4,                 // rgb(255,105,180)
        indian_red = 0xCD5C5C,               // rgb(205,92,92)
        indigo = 0x4B0082,                   // rgb(75,0,130)
        ivory = 0xFFFFF0,                    // rgb(255,255,240)
        khaki = 0xF0E68C,                    // rgb(240,230,140)
        lavender = 0xE6E6FA,                 // rgb(230,230,250)
        lavender_blush = 0xFFF0F5,           // rgb(255,240,245)
        lawn_green = 0x7CFC00,               // rgb(124,252,0)
        lemon_chiffon = 0xFFFACD,            // rgb(255,250,205)
        light_blue = 0xADD8E6,               // rgb(173,216,230)
        light_coral = 0xF08080,              // rgb(240,128,128)
        light_cyan = 0xE0FFFF,               // rgb(224,255,255)
        light_golden_rod_yellow = 0xFAFAD2,  // rgb(250,250,210)
        light_gray = 0xD3D3D3,               // rgb(211,211,211)
        light_green = 0x90EE90,              // rgb(144,238,144)
        light_pink = 0xFFB6C1,               // rgb(255,182,193)
        light_salmon = 0xFFA07A,             // rgb(255,160,122)
        light_sea_green = 0x20B2AA,          // rgb(32,178,170)
        light_sky_blue = 0x87CEFA,           // rgb(135,206,250)
        light_slate_gray = 0x778899,         // rgb(119,136,153)
        light_steel_blue = 0xB0C4DE,         // rgb(176,196,222)
        light_yellow = 0xFFFFE0,             // rgb(255,255,224)
        lime = 0x00FF00,                     // rgb(0,255,0)
        lime_green = 0x32CD32,               // rgb(50,205,50)
        linen = 0xFAF0E6,                    // rgb(250,240,230)
        magenta = 0xFF00FF,                  // rgb(255,0,255)
        maroon = 0x800000,                   // rgb(128,0,0)
        medium_aquamarine = 0x66CDAA,        // rgb(102,205,170)
        medium_blue = 0x0000CD,              // rgb(0,0,205)
        medium_orchid = 0xBA55D3,            // rgb(186,85,211)
        medium_purple = 0x9370DB,            // rgb(147,112,219)
        medium_sea_green = 0x3CB371,         // rgb(60,179,113)
        medium_slate_blue = 0x7B68EE,        // rgb(123,104,238)
        medium_spring_green = 0x00FA9A,      // rgb(0,250,154)
        medium_turquoise = 0x48D1CC,         // rgb(72,209,204)
        medium_violet_red = 0xC71585,        // rgb(199,21,133)
        midnight_blue = 0x191970,            // rgb(25,25,112)
        mint_cream = 0xF5FFFA,               // rgb(245,255,250)
        misty_rose = 0xFFE4E1,               // rgb(255,228,225)
        moccasin = 0xFFE4B5,                 // rgb(255,228,181)
        navajo_white = 0xFFDEAD,             // rgb(255,222,173)
        navy = 0x000080,                     // rgb(0,0,128)
        old_lace = 0xFDF5E6,                 // rgb(253,245,230)
        olive = 0x808000,                    // rgb(128,128,0)
        olive_drab = 0x6B8E23,               // rgb(107,142,35)
        orange = 0xFFA500,                   // rgb(255,165,0)
        orange_red = 0xFF4500,               // rgb(255,69,0)
        orchid = 0xDA70D6,                   // rgb(218,112,214)
        pale_golden_rod = 0xEEE8AA,          // rgb(238,232,170)
        pale_green = 0x98FB98,               // rgb(152,251,152)
        pale_turquoise = 0xAFEEEE,           // rgb(175,238,238)
        pale_violet_red = 0xDB7093,          // rgb(219,112,147)
        papaya_whip = 0xFFEFD5,              // rgb(255,239,213)
        peach_puff = 0xFFDAB9,               // rgb(255,218,185)
        peru = 0xCD853F,                     // rgb(205,133,63)
        pink = 0xFFC0CB,                     // rgb(255,192,203)
        plum = 0xDDA0DD,                     // rgb(221,160,221)
        powder_blue = 0xB0E0E6,              // rgb(176,224,230)
        purple = 0x800080,                   // rgb(128,0,128)
        rebecca_purple = 0x663399,           // rgb(102,51,153)
        red = 0xFF0000,                      // rgb(255,0,0)
        rosy_brown = 0xBC8F8F,               // rgb(188,143,143)
        royal_blue = 0x4169E1,               // rgb(65,105,225)
        saddle_brown = 0x8B4513,             // rgb(139,69,19)
        salmon = 0xFA8072,                   // rgb(250,128,114)
        sandy_brown = 0xF4A460,              // rgb(244,164,96)
        sea_green = 0x2E8B57,                // rgb(46,139,87)
        sea_shell = 0xFFF5EE,                // rgb(255,245,238)
        sienna = 0xA0522D,                   // rgb(160,82,45)
        silver = 0xC0C0C0,                   // rgb(192,192,192)
        sky_blue = 0x87CEEB,                 // rgb(135,206,235)
        slate_blue = 0x6A5ACD,               // rgb(106,90,205)
        slate_gray = 0x708090,               // rgb(112,128,144)
        snow = 0xFFFAFA,                     // rgb(255,250,250)
        spring_green = 0x00FF7F,             // rgb(0,255,127)
        steel_blue = 0x4682B4,               // rgb(70,130,180)
        tan = 0xD2B48C,                      // rgb(210,180,140)
        teal = 0x008080,                     // rgb(0,128,128)
        thistle = 0xD8BFD8,                  // rgb(216,191,216)
        tomato = 0xFF6347,                   // rgb(255,99,71)
        turquoise = 0x40E0D0,                // rgb(64,224,208)
        violet = 0xEE82EE,                   // rgb(238,130,238)
        wheat = 0xF5DEB3,                    // rgb(245,222,179)
        white = 0xFFFFFF,                    // rgb(255,255,255)
        white_smoke = 0xF5F5F5,              // rgb(245,245,245)
        yellow = 0xFFFF00,                   // rgb(255,255,0)
        yellow_green = 0x9ACD32              // rgb(154,205,50)
    };                                     // enum class color

    enum class terminal_color : uint8_t {
        black = 30,
        red,
        green,
        yellow,
        blue,
        magenta,
        cyan,
        white,
        bright_black = 90,
        bright_red,
        bright_green,
        bright_yellow,
        bright_blue,
        bright_magenta,
        bright_cyan,
        bright_white
    };

    enum class emphasis : uint8_t {
        bold = 1,
        faint = 1 << 1,
        italic = 1 << 2,
        underline = 1 << 3,
        blink = 1 << 4,
        reverse = 1 << 5,
        conceal = 1 << 6,
        strikethrough = 1 << 7,
    };

    constexpr bool operator&(emphasis lhs, emphasis rhs) noexcept {
        return static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs);
    }

    // rgb is a struct for red, green and blue colors.
    // Using the name "rgb" makes some editors show the color in a tooltip.
    struct rgb {
        constexpr rgb() : r(0), g(0), b(0) {}

        constexpr rgb(uint8_t r_, uint8_t g_, uint8_t b_) : r(r_), g(g_), b(b_) {}

        constexpr rgb(uint32_t hex)
                : r((hex >> 16) & 0xFF), g((hex >> 8) & 0xFF), b(hex & 0xFF) {}

        constexpr rgb(color hex)
                : r((uint32_t(hex) >> 16) & 0xFF),
                  g((uint32_t(hex) >> 8) & 0xFF),
                  b(uint32_t(hex) & 0xFF) {}

        uint8_t r;
        uint8_t g;
        uint8_t b;
    };

    namespace fmt_detail {

        // color is a struct of either a rgb color or a terminal color.
        struct color_type {
            constexpr color_type() noexcept: is_rgb(), value{} {}

            constexpr color_type(color rgb_color) noexcept: is_rgb(true), value{} {
                value.rgb_color = static_cast<uint32_t>(rgb_color);
            }

            constexpr color_type(rgb rgb_color) noexcept: is_rgb(true), value{} {
                value.rgb_color = (static_cast<uint32_t>(rgb_color.r) << 16) |
                                  (static_cast<uint32_t>(rgb_color.g) << 8) | rgb_color.b;
            }

            constexpr color_type(terminal_color term_color) noexcept
                    : is_rgb(), value{} {
                value.term_color = static_cast<uint8_t>(term_color);
            }

            bool is_rgb;
            union color_union {
                uint8_t term_color;
                uint32_t rgb_color;
            } value;
        };

    }  // namespace fmt_detail

    /** A text style consisting of foreground and background colors and emphasis. */
    class text_style {
    public:
        constexpr text_style(emphasis em = emphasis()) noexcept
                : set_foreground_color(), set_background_color(), ems(em) {}

        constexpr text_style &operator|=(const text_style &rhs) {
            if (!set_foreground_color) {
                set_foreground_color = rhs.set_foreground_color;
                foreground_color = rhs.foreground_color;
            } else if (rhs.set_foreground_color) {
                if (!foreground_color.is_rgb || !rhs.foreground_color.is_rgb)
                    FMT_THROW(format_error("can't OR a terminal color"));
                foreground_color.value.rgb_color |= rhs.foreground_color.value.rgb_color;
            }

            if (!set_background_color) {
                set_background_color = rhs.set_background_color;
                background_color = rhs.background_color;
            } else if (rhs.set_background_color) {
                if (!background_color.is_rgb || !rhs.background_color.is_rgb)
                    FMT_THROW(format_error("can't OR a terminal color"));
                background_color.value.rgb_color |= rhs.background_color.value.rgb_color;
            }

            ems = static_cast<emphasis>(static_cast<uint8_t>(ems) |
                                        static_cast<uint8_t>(rhs.ems));
            return *this;
        }

        friend constexpr text_style operator|(text_style lhs,
                                              const text_style &rhs) {
            return lhs |= rhs;
        }

        constexpr bool has_foreground() const noexcept {
            return set_foreground_color;
        }

        constexpr bool has_background() const noexcept {
            return set_background_color;
        }

        constexpr bool has_emphasis() const noexcept {
            return static_cast<uint8_t>(ems) != 0;
        }

        constexpr fmt_detail::color_type get_foreground() const noexcept {
            TURBO_ASSERT(has_foreground()&&"no foreground specified for this style");
            return foreground_color;
        }

        constexpr fmt_detail::color_type get_background() const noexcept {
            TURBO_ASSERT(has_background()&&"no background specified for this style");
            return background_color;
        }

        constexpr emphasis get_emphasis() const noexcept {
            TURBO_ASSERT(has_emphasis()&&"no emphasis specified for this style");
            return ems;
        }

    private:
        constexpr text_style(bool is_foreground,
                             fmt_detail::color_type text_color) noexcept
                : set_foreground_color(), set_background_color(), ems() {
            if (is_foreground) {
                foreground_color = text_color;
                set_foreground_color = true;
            } else {
                background_color = text_color;
                set_background_color = true;
            }
        }

        friend constexpr text_style fg(fmt_detail::color_type foreground) noexcept;

        friend constexpr text_style bg(fmt_detail::color_type background) noexcept;

        fmt_detail::color_type foreground_color;
        fmt_detail::color_type background_color;
        bool set_foreground_color;
        bool set_background_color;
        emphasis ems;
    };

    /** Creates a text style from the foreground (text) color. */
    constexpr inline text_style fg(fmt_detail::color_type foreground) noexcept {
        return text_style(true, foreground);
    }

    /** Creates a text style from the background color. */
    constexpr inline text_style bg(fmt_detail::color_type background) noexcept {
        return text_style(false, background);
    }

    constexpr inline text_style operator|(emphasis lhs, emphasis rhs) noexcept {
        return text_style(lhs) | rhs;
    }

    namespace fmt_detail {

        template<typename Char>
        struct ansi_color_escape {
            constexpr ansi_color_escape(fmt_detail::color_type text_color,
                                        const char *esc) noexcept {
                // If we have a terminal color, we need to output another escape code
                // sequence.
                if (!text_color.is_rgb) {
                    bool is_background = esc == std::string_view("\x1b[48;2;");
                    uint32_t value = text_color.value.term_color;
                    // Background ASCII codes are the same as the foreground ones but with
                    // 10 more.
                    if (is_background) value += 10u;

                    size_t index = 0;
                    buffer[index++] = static_cast<Char>('\x1b');
                    buffer[index++] = static_cast<Char>('[');

                    if (value >= 100u) {
                        buffer[index++] = static_cast<Char>('1');
                        value %= 100u;
                    }
                    buffer[index++] = static_cast<Char>('0' + value / 10u);
                    buffer[index++] = static_cast<Char>('0' + value % 10u);

                    buffer[index++] = static_cast<Char>('m');
                    buffer[index++] = static_cast<Char>('\0');
                    return;
                }

                for (int i = 0; i < 7; i++) {
                    buffer[i] = static_cast<Char>(esc[i]);
                }
                rgb color(text_color.value.rgb_color);
                to_esc(color.r, buffer + 7, ';');
                to_esc(color.g, buffer + 11, ';');
                to_esc(color.b, buffer + 15, 'm');
                buffer[19] = static_cast<Char>(0);
            }

            constexpr ansi_color_escape(emphasis em) noexcept {
                uint8_t em_codes[num_emphases] = {};
                if (has_emphasis(em, emphasis::bold)) em_codes[0] = 1;
                if (has_emphasis(em, emphasis::faint)) em_codes[1] = 2;
                if (has_emphasis(em, emphasis::italic)) em_codes[2] = 3;
                if (has_emphasis(em, emphasis::underline)) em_codes[3] = 4;
                if (has_emphasis(em, emphasis::blink)) em_codes[4] = 5;
                if (has_emphasis(em, emphasis::reverse)) em_codes[5] = 7;
                if (has_emphasis(em, emphasis::conceal)) em_codes[6] = 8;
                if (has_emphasis(em, emphasis::strikethrough)) em_codes[7] = 9;

                size_t index = 0;
                for (size_t i = 0; i < num_emphases; ++i) {
                    if (!em_codes[i]) continue;
                    buffer[index++] = static_cast<Char>('\x1b');
                    buffer[index++] = static_cast<Char>('[');
                    buffer[index++] = static_cast<Char>('0' + em_codes[i]);
                    buffer[index++] = static_cast<Char>('m');
                }
                buffer[index++] = static_cast<Char>(0);
            }

            constexpr operator const Char *() const noexcept { return buffer; }

            constexpr const Char *begin() const noexcept { return buffer; }

            constexpr const Char *end() const noexcept {
                return buffer + std::char_traits<Char>::length(buffer);
            }

        private:
            static constexpr size_t num_emphases = 8;
            Char buffer[7u + 3u * num_emphases + 1u];

            static constexpr void to_esc(uint8_t c, Char *out,
                                         char delimiter) noexcept {
                out[0] = static_cast<Char>('0' + c / 100);
                out[1] = static_cast<Char>('0' + c / 10 % 10);
                out[2] = static_cast<Char>('0' + c % 10);
                out[3] = static_cast<Char>(delimiter);
            }

            static constexpr bool has_emphasis(emphasis em, emphasis mask) noexcept {
                return static_cast<uint8_t>(em) & static_cast<uint8_t>(mask);
            }
        };

        template<typename Char>
        constexpr ansi_color_escape<Char> make_foreground_color(
                fmt_detail::color_type foreground) noexcept {
            return ansi_color_escape<Char>(foreground, "\x1b[38;2;");
        }

        template<typename Char>
        constexpr ansi_color_escape<Char> make_background_color(
                fmt_detail::color_type background) noexcept {
            return ansi_color_escape<Char>(background, "\x1b[48;2;");
        }

        template<typename Char>
        constexpr ansi_color_escape<Char> make_emphasis(emphasis em) noexcept {
            return ansi_color_escape<Char>(em);
        }

        template<typename Char>
        inline void reset_color(buffer<Char> &buffer) {
            auto reset_color = std::string_view("\x1b[0m");
            buffer.append(reset_color.begin(), reset_color.end());
        }

        template<typename T>
        struct styled_arg {
            const T &value;
            text_style style;
        };

        template<typename Char>
        void vformat_to(buffer<Char> &buf, const text_style &ts,
                        std::basic_string_view<Char> format_str,
                        basic_format_args<buffer_context<type_identity_t<Char>>> args) {
            bool has_style = false;
            if (ts.has_emphasis()) {
                has_style = true;
                auto emphasis = fmt_detail::make_emphasis<Char>(ts.get_emphasis());
                buf.append(emphasis.begin(), emphasis.end());
            }
            if (ts.has_foreground()) {
                has_style = true;
                auto foreground = fmt_detail::make_foreground_color<Char>(ts.get_foreground());
                buf.append(foreground.begin(), foreground.end());
            }
            if (ts.has_background()) {
                has_style = true;
                auto background = fmt_detail::make_background_color<Char>(ts.get_background());
                buf.append(background.begin(), background.end());
            }
            fmt_detail::vformat_to(buf, format_str, args, {});
            if (has_style)
                fmt_detail::reset_color<Char>(buf);
        }

    }  // namespace fmt_detail

    inline void vprint(std::FILE *f, const text_style &ts, std::string_view fmt, format_args args) {
        // Legacy wide streams are not supported.
        auto buf = memory_buffer();
        fmt_detail::vformat_to(buf, ts, fmt, args);
        if (fmt_detail::is_utf8()) {
            fmt_detail::print(f, std::string_view(buf.begin(), buf.size()));
            return;
        }
        buf.push_back('\0');
        int result = std::fputs(buf.data(), f);
        if (result < 0)
            FMT_THROW(system_error(errno, FMT_STRING("cannot write to file")));
    }


    template<typename S, typename Char = char_t<S>>
    inline std::basic_string<Char> vformat(
            const text_style &ts, const S &format_str,
            basic_format_args<buffer_context<type_identity_t<Char>>> args) {
        basic_memory_buffer<Char> buf;
        fmt_detail::vformat_to(buf, ts, to_string_view(format_str), args
        );
        return
                turbo::to_string(buf);
    }


    /**
      Formats a string with the given text_style and writes the output to ``out``.
     */
    template<typename OutputIt, typename Char,
            TURBO_ENABLE_IF(fmt_detail::is_output_iterator<OutputIt, Char>::value)>

    OutputIt vformat_to(
            OutputIt out, const text_style &ts, std::basic_string_view<Char> format_str,
            basic_format_args<buffer_context<type_identity_t<Char>>> args) {
        auto &&buf = fmt_detail::get_buffer<Char>(out);
        fmt_detail::vformat_to(buf, ts, format_str, args
        );
        return
                fmt_detail::get_iterator(buf, out
                );
    }


    template<typename T, typename Char>
    struct formatter<fmt_detail::styled_arg<T>, Char> : formatter<T, Char> {
        template<typename FormatContext>
        auto format(const fmt_detail::styled_arg<T> &arg, FormatContext &ctx) const
        -> decltype(ctx.out()) {
            const auto &ts = arg.style;
            const auto &value = arg.value;
            auto out = ctx.out();

            bool has_style = false;
            if (ts.has_emphasis()) {
                has_style = true;
                auto emphasis = fmt_detail::make_emphasis<Char>(ts.get_emphasis());
                out = std::copy(emphasis.begin(), emphasis.end(), out);
            }
            if (ts.has_foreground()) {
                has_style = true;
                auto foreground =
                        fmt_detail::make_foreground_color<Char>(ts.get_foreground());
                out = std::copy(foreground.begin(), foreground.end(), out);
            }
            if (ts.has_background()) {
                has_style = true;
                auto background =
                        fmt_detail::make_background_color<Char>(ts.get_background());
                out = std::copy(background.begin(), background.end(), out);
            }
            out = formatter<T, Char>::format(value, ctx);
            if (has_style) {
                auto reset_color = std::string_view("\x1b[0m");
                out = std::copy(reset_color.begin(), reset_color.end(), out);
            }
            return out;
        }

    };

    /**
     * @brief apply the style to a buff
     * @tparam Char charactor type
     * @tparam Out output iterator
     * @param ts
     * @return the iterator has write
     */
    template<typename Char, typename Out>
    Out apply_text_style(text_style ts, Out output);

    class text_style_builder {
    public:
        constexpr text_style_builder() noexcept: ts_() {}

        explicit constexpr text_style_builder(const text_style &ts) noexcept: ts_(ts) {}

        constexpr text_style_builder &operator|=(const text_style &rhs) noexcept {
            ts_ |= rhs;
            return *this;
        }

        constexpr text_style_builder &operator|=(const text_style_builder &rhs) noexcept {
            ts_ |= rhs.ts_;
            return *this;
        }

        constexpr text_style_builder &operator|=(emphasis em) noexcept {
            ts_ |= em;
            return *this;
        }

        explicit constexpr operator text_style() const noexcept { return ts_; }

        [[nodiscard]] constexpr text_style get() const noexcept { return ts_; }

        void set(text_style ts) noexcept { ts_ = ts; }

        void reset() noexcept { ts_ = text_style(); }

        text_style_builder &merge(text_style_builder rhs) {
            ts_ |= rhs.ts_;
            return *this;
        }

        text_style_builder &merge(text_style rhs) {
            ts_ |= rhs;
            return *this;
        }

        text_style_builder &fg(color foreground) noexcept {
            ts_ | turbo::fg(foreground);
            return *this;
        }

        text_style_builder &fg(terminal_color tc) noexcept {
            ts_ |= turbo::fg(fmt_detail::color_type(tc));
            return *this;
        }

        text_style_builder &bg(color background) noexcept {
            ts_ | turbo::bg(background);
            return *this;
        }

        text_style_builder &bg(terminal_color tc) noexcept {
            ts_ |= turbo::bg(fmt_detail::color_type(tc));
            return *this;
        }

        text_style_builder &emphasis(emphasis em) noexcept {
            ts_ | em;
            return *this;
        }

        text_style_builder &bold() noexcept {
            ts_ | emphasis::bold;
            return *this;
        }

        text_style_builder &faint() noexcept {
            ts_ | emphasis::faint;
            return *this;
        }

        text_style_builder &italic() noexcept {
            ts_ | emphasis::italic;
            return *this;
        }

        text_style_builder &underline() noexcept {
            ts_ | emphasis::underline;
            return *this;
        }

        text_style_builder &blink() noexcept {
            ts_ | emphasis::blink;
            return *this;
        }

        text_style_builder &reverse() noexcept {
            ts_ | emphasis::reverse;
            return *this;
        }

        text_style_builder &conceal() noexcept {
            ts_ | emphasis::conceal;
            return *this;
        }

        text_style_builder strikethrough() noexcept {
            ts_ | emphasis::strikethrough;
            return *this;
        }

        [[nodiscard]] std::string build() const {
            std::string buff;
            buff.resize(500);
            auto out = apply_text_style<char>(ts_, buff.begin());
            buff.resize(out - buff.begin());
            return buff;
        }

    private:
        text_style ts_;
    };

    template<typename Char, typename Out>
    Out apply_text_style(text_style ts, Out output) {
        if (ts.has_emphasis()) {
            auto emphasis = fmt_detail::make_emphasis<char>(ts.get_emphasis());
            output = std::copy(emphasis.begin(), emphasis.end(), output);
        }
        if (ts.has_foreground()) {
            auto foreground = fmt_detail::make_foreground_color<char>(ts.get_foreground());
            output = std::copy(foreground.begin(), foreground.end(), output);
        }
        if (ts.has_background()) {
            auto background = fmt_detail::make_background_color<char>(ts.get_background());
            output = std::copy(background.begin(), background.end(), output);
        }
        return output;
    }


    inline text_style_builder operator|(const text_style_builder &lhs,
                                        const text_style &rhs) noexcept {
        return text_style_builder(lhs) |= rhs;
    }

    inline text_style_builder operator|(const text_style &lhs,
                                        const text_style_builder &rhs) noexcept {
        return text_style_builder(lhs) |= rhs;
    }

    inline text_style_builder operator|(const text_style_builder &lhs,
                                        const text_style_builder &rhs) noexcept {
        return text_style_builder(lhs) |= rhs;
    }

    inline std::string apply_text_style(text_style_builder ts) {
        return std::move(ts.build());
    }

    inline std::string apply_text_style(text_style ts) {
        return apply_text_style(text_style_builder(ts));
    }

    static constexpr std::string_view KResetStyle = "\x1b[0m";

    inline std::string_view reset_text_style() {
        return KResetStyle;
    }

}  // namespace turbo

#endif  // FMT_COLOR_H_
