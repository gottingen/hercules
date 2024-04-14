#
# Copyright 2023 The Turbo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

carbin_cc_binary(
        NAME
        string_print
        SOURCES
        "string_print.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        asciidoc_export
        SOURCES
        "asciidoc_export.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        class_diagram
        SOURCES
        "class_diagram.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        colors
        SOURCES
        "colors.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        refresh_color
        SOURCES
        "refresh_color.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        employees
        SOURCES
        "employees.cc"
        DEPS
        turbo::turbo
)
carbin_cc_binary(
        NAME
        font_styles
        SOURCES
        "font_styles.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        iterators
        SOURCES
        "iterators.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        latex_export
        SOURCES
        "latex_export.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        mario
        SOURCES
        "mario.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        markdown_export
        SOURCES
        "markdown_export.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        movies
        SOURCES
        "movies.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        padding
        SOURCES
        "padding.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        refresh_table
        SOURCES
        "refresh_table.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        runic
        SOURCES
        "runic.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        shape
        SOURCES
        "shape.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        string_view_in_row
        SOURCES
        "string_view_in_row.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        summary
        SOURCES
        "summary.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        unicode
        SOURCES
        "unicode.cc"
        DEPS
        turbo::turbo
)


carbin_cc_binary(
        NAME
        universal_constants
        SOURCES
        "universal_constants.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        word_wrapping
        SOURCES
        "word_wrapping.cc"
        DEPS
        turbo::turbo
)