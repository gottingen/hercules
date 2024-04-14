// Copyright 2024 The Elastic AI Search Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <atomic>
#include <chrono>
#include <random>
#include <collie/table/table.h>
#include <thread>

using namespace collie::table;
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
  while (keep_running) {
    Table process_table;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    process_table.add_row(Row_t{"PID", "%CPU", "%MEM", "User", "NI"});
    process_table.add_row(Row_t{"4297", std::to_string((int)round(dis(gen) * 100)),
                                std::to_string((int)round(dis(gen) * 100)), "ubuntu", "20"});
    process_table.add_row(Row_t{"12671", std::to_string((int)round(dis(gen) * 100)),
                                std::to_string((int)round(dis(gen) * 100)), "root", "0"});
    process_table.add_row({"810", std::to_string((int)round(dis(gen) * 100)),
                           std::to_string((int)round(dis(gen) * 100)), "root", "-20"});

    process_table.column(2).format().font_align(FontAlign::center);
    process_table.column(3).format().font_align(FontAlign::right);
    process_table.column(4).format().font_align(FontAlign::right);

    for (size_t i = 0; i < 5; ++i) {
      process_table[0][i]
          .format()
          .font_color(Color::yellow)
          .font_align(FontAlign::center)
          .font_style({FontStyle::bold});
    }

    std::cout << process_table << std::endl;
    std::cout << "Press ENTER to exit..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F";
  }
  std::cout << "\033[B\033[B\033[B\033[B\033[B\033[B\033[B\033[B\033[B\033[B";
  tUserInput.join();

  return 0;
}
