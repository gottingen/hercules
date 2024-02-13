// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file hvm/runtime/module.h
 * \brief Runtime container of the functions generated by Hercules,
 *  This is used to support dynamically link, load and save
 *  functions from different convention under unified API.
 */
// !!!NOTICE!!!
// Keep the header guard unchanged.
#ifndef HERCULES_RUNTIME_MODULE_H
#define HERCULES_RUNTIME_MODULE_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hercules/runtime/c_runtime_api.h>
#include <hercules/runtime/container/string.h>
#include <hercules/runtime/function.h>
#include <hercules/runtime/memory.h>
#include <hercules/runtime/object.h>

namespace hercules {
namespace runtime {

class ModuleNode;

class Module : public ObjectRef {
 public:
  Module() {
  }
  // constructor from container.
  explicit Module(ObjectPtr<Object> n) : ObjectRef(std::move(n)) {
  }
  /*!
   * \brief Get packed function from current module by name.
   *
   * \param name The name of the function.
   * \param query_imports Whether also query dependency modules.
   * \return The result function.
   *  This function will return PackedFunc(nullptr) if function do not exist.
   * \note Implemented in packed_func.cc
   */
  inline NativeFunction GetFunction(const String& name, bool query_imports = false);
  // The following functions requires link with runtime.
  /*!
   * \brief Import another module into this module.
   * \param other The module to be imported.
   *
   * \note Cyclic dependency is not allowed among modules,
   *  An error will be thrown when cyclic dependency is detected.
   */
  inline void Import(Module other);
  /*! \return internal container */
  inline ModuleNode* operator->();
  /*! \return internal container */
  inline const ModuleNode* operator->() const;
  /*!
   * \brief Load a module from file.
   * \param file_name The name of the host function module.
   * \param format The format of the file.
   * \note This function won't load the import relationship.
   *  Re-create import relationship by calling Import.
   */
  HERCULES_DLL static Module LoadFromFile(const String& file_name, const String& format = "");
  // refer to the corresponding container.
  using ContainerType = ModuleNode;
  friend class ModuleNode;
};

/*!
 * \brief Base container of module.
 *
 * Please subclass ModuleNode to create a specific runtime module.
 *
 * \code
 *
 *  class MyModuleNode : public ModuleNode {
 *   public:
 *    // implement the interface
 *  };
 *
 *  // use make_object to create a specific
 *  // instace of MyModuleNode.
 *  Module CreateMyModule() {
 *    ObjectPtr<MyModuleNode> n =
 *      hvm::runtime::make_object<MyModuleNode>();
 *    return Module(n);
 *  }
 *
 * \endcode
 */
class HERCULES_DLL ModuleNode : public Object {
 public:
  /*! \brief virtual destructor */
  virtual ~ModuleNode() {
  }
  /*!
   * \return The per module type key.
   * \note This key is used to for serializing custom modules.
   */
  virtual const char* type_key() const = 0;
  /*!
   * \brief Get a PackedFunc from module.
   *
   *  The PackedFunc may not be fully initialized,
   *  there might still be first time running overhead when
   *  executing the function on certain devices.
   *  For benchmarking, use prepare to eliminate
   *
   * \param name the name of the function.
   * \param sptr_to_self The ObjectPtr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   *
   * \note The function will always remain valid.
   *   If the function need resource from the module(e.g. late linking),
   *   it should capture sptr_to_self.
   */
  virtual NativeFunction GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) = 0;
  /*!
   * \brief Save the module to file.
   * \param file_name The file to be saved to.
   * \param format The format of the file.
   */
  virtual void SaveToFile(const String& file_name, const String& format);
  /*!
   * \brief Save the module to binary stream.
   * \param stream The binary stream to save to.
   * \note It is recommended to implement this for device modules,
   *   but not necessarily host modules.
   *   We can use this to do AOT loading of bundled device functions.
   */
  // virtual void SaveToBinary(dmlc::Stream* stream);
  /*!
   * \brief Get the source code of module, when available.
   * \param format Format of the source code, can be empty by default.
   * \return Possible source code when available.
   */
  virtual String GetSource(const String& format = "");
  /*!
   * \brief Get packed function from current module by name.
   *
   * \param name The name of the function.
   * \param query_imports Whether also query dependency modules.
   * \return The result function.
   *  This function will return PackedFunc(nullptr) if function do not exist.
   * \note Implemented in packed_func.cc
   */
  NativeFunction GetFunction(const String& name, bool query_imports = false);
  /*!
   * \brief Import another module into this module.
   * \param other The module to be imported.
   *
   * \note Cyclic dependency is not allowed among modules,
   *  An error will be thrown when cyclic dependency is detected.
   */
  void Import(Module other);
  /*!
   * \brief Get a function from current environment
   *  The environment includes all the imports as well as Global functions.
   *
   * \param name name of the function.
   * \return The corresponding function.
   */
  const NativeFunction* GetFuncFromEnv(const String& name);
  /*! \return The module it imports from */
  const std::vector<Module>& imports() const {
    return imports_;
  }

  // integration with the existing components.
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeModule;
  static constexpr const char* _type_key = "runtime.Module";
  // NOTE: ModuleNode can still be sub-classed
  //
  HERCULES_DECLARE_FINAL_OBJECT_INFO(ModuleNode, Object);

 protected:
  friend class Module;
  friend class ModuleInternal;
  /*! \brief The modules this module depend on */
  std::vector<Module> imports_;

 private:
  /*! \brief Cache used by GetImport */
  std::unordered_map<String, std::shared_ptr<NativeFunction>> import_cache_;
};

/*! \brief namespace for constant symbols */
namespace symbol {
/*! \brief Global variable to store module context. */
constexpr const char* library_module_ctx = "__hercules_module_ctx";
/*! \brief Placeholder for the module's entry function. */
constexpr const char* library_func_array = "__hercules_func_array__";
constexpr const char* library_func_registry = "__hercules_func_registry__";
/*! \brief Placeholder for the module's closures names. */
constexpr const char* library_closures_names = "__hercules_closures_names__";
}  // namespace symbol

// implementations of inline functions.

inline void Module::Import(Module other) {
  return (*this)->Import(other);
}

inline ModuleNode* Module::operator->() {
  return static_cast<ModuleNode*>(get_mutable());
}

inline const ModuleNode* Module::operator->() const {
  return static_cast<const ModuleNode*>(get());
}

inline NativeFunction Module::GetFunction(const String& name, bool query_imports) {
  return (*this)->GetFunction(name, query_imports);
}

}  // namespace runtime
}  // namespace hercules

#endif  // HERCULES_RUNTIME_MODULE_H
