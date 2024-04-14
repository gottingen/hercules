// Copyright 2023 The titan-search Authors.
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

namespace flags_internal {
std::mutex* GetGlobalConstructionGuard();
}  // namespace flags_internal

template <typename T>
class Flag {
 public:
  // No constructor and destructor to ensure this is an aggregate type.
  // Visual Studio 2015 still requires the constructor for class to be
  // constexpr initializable.
#if _MSC_VER <= 1900
  constexpr Flag(const char* name, const char* filename,
                 const flags_internal::HelpGenFunc help_gen,
                 const flags_internal::FlagDfltGenFunc default_value_gen)
      : name_(name),
        filename_(filename),
        help_gen_(help_gen),
        default_value_gen_(default_value_gen),
        inited_(false),
        impl_(nullptr) {}
#endif

  flags_internal::Flag<T>& GetImpl() const {
    if (!inited_.load(std::memory_order_acquire)) {
      turbo::MutexLock l(flags_internal::GetGlobalConstructionGuard());

      if (inited_.load(std::memory_order_acquire)) {
        return *impl_;
      }

      impl_ = new flags_internal::Flag<T>(
          name_, filename_,
          {flags_internal::FlagHelpMsg(help_gen_),
           flags_internal::FlagHelpKind::kGenFunc},
          {flags_internal::FlagDefaultSrc(default_value_gen_),
           flags_internal::FlagDefaultKind::kGenFunc});
      inited_.store(true, std::memory_order_release);
    }

    return *impl_;
  }

  bool is_retired() const { return GetImpl().is_retired(); }
  std::string_view Name() const { return GetImpl().Name(); }
  std::string Help() const { return GetImpl().Help(); }
  bool IsModified() const { return GetImpl().IsModified(); }
  bool is_specified_on_command_line() const {
    return GetImpl().is_specified_on_command_line();
  }
  std::string Filename() const { return GetImpl().Filename(); }
  std::string default_value() const { return GetImpl().default_value(); }
  std::string current_value() const { return GetImpl().current_value(); }
  template <typename U>
  inline bool is_of_type() const {
    return GetImpl().template is_of_type<U>();
  }
  T Get() const {
    return flags_internal::FlagImplPeer::InvokeGet<T>(GetImpl());
  }
  void Set(const T& v) {
    flags_internal::FlagImplPeer::InvokeSet(GetImpl(), v);
  }
  void InvokeCallback() { GetImpl().InvokeCallback(); }

  const CommandLineFlag& Reflect() const {
    return flags_internal::FlagImplPeer::InvokeReflect(GetImpl());
  }

  // The data members are logically private, but they need to be public for
  // this to be an aggregate type.
  const char* name_;
  const char* filename_;
  const flags_internal::HelpGenFunc help_gen_;
  const flags_internal::FlagDfltGenFunc default_value_gen_;

  mutable std::atomic<bool> inited_;
  mutable flags_internal::Flag<T>* impl_;
};
