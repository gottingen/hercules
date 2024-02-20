// Copyright 2023 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#include <string>
#include <tuple>

#include "hercules/parser/ast.h"
#include "hercules/parser/cache.h"
#include "hercules/parser/common.h"
#include "hercules/parser/visitors/simplify/simplify.h"
#include "hercules/parser/visitors/typecheck/typecheck.h"

using fmt::format;
using namespace hercules::error;

namespace hercules::ast {

    using namespace types;

    /// Typecheck identifiers. If an identifier is a static variable, evaluate it and
    /// replace it with its value (e.g., a @c IntExpr ). Also ensure that the identifier of
    /// a generic function or a type is fully qualified (e.g., replace `Ptr` with
    /// `Ptr[byte]`).
    void TypecheckVisitor::visit(IdExpr *expr) {
        // Replace identifiers that have been superseded by domination analysis during the
        // simplification
        while (auto s = in(ctx->cache->replacements, expr->value))
            expr->value = s->first;

        auto val = ctx->find(expr->value);
        if (!val) {
            // Handle overloads
            if (in(ctx->cache->overloads, expr->value)) {
                val = ctx->forceFind(getDispatch(expr->value)->ast->name);
            }
            seqassert(val, "cannot find '{}'", expr->value);
        }
        unify(expr->type, ctx->instantiate(val->type));

        if (val->type->isStaticType()) {
            // Evaluate static expression if possible
            expr->staticValue.type = StaticValue::Type(val->type->isStaticType());
            auto s = val->type->getStatic();
            seqassert(!expr->staticValue.evaluated, "expected unevaluated expression: {}",
                      expr->toString());
            if (s && s->expr->staticValue.evaluated) {
                // Replace the identifier with static expression
                if (s->expr->staticValue.type == StaticValue::STRING)
                    resultExpr = transform(N<StringExpr>(s->expr->staticValue.getString()));
                else
                    resultExpr = transform(N<IntExpr>(s->expr->staticValue.getInt()));
            }
            return;
        }

        if (val->isType())
            expr->markType();

        // Realize a type or a function if possible and replace the identifier with the fully
        // typed identifier (e.g., `foo` -> `foo[int]`)
        if (realize(expr->type)) {
            if (!val->isVar())
                expr->value = expr->type->realizedName();
            expr->setDone();
        }
    }

    /// See @c transformDot for details.
    void TypecheckVisitor::visit(DotExpr *expr) {
        // Make sure to unify the current type with the transformed type
        if ((resultExpr = transformDot(expr)))
            unify(expr->type, resultExpr->type);
        if (!expr->type)
            unify(expr->type, ctx->getUnbound());
    }

    /// Find an overload dispatch function for a given overload. If it does not exist and
    /// there is more than one overload, generate it. Dispatch functions ensure that a
    /// function call is being routed to the correct overload even when dealing with partial
    /// functions and decorators.
    /// @example This is how dispatch looks like:
    ///   ```def foo:dispatch(*args, **kwargs):
    ///        return foo(*args, **kwargs)```
    types::FuncTypePtr TypecheckVisitor::getDispatch(const std::string &fn) {
        auto &overloads = ctx->cache->overloads[fn];

        // Single overload: just return it
        if (overloads.size() == 1)
            return ctx->forceFind(overloads.front().name)->type->getFunc();

        // Check if dispatch exists
        for (auto &m: overloads)
            if (endswith(ctx->cache->functions[m.name].ast->name, ":dispatch"))
                return ctx->cache->functions[m.name].type;

        // Dispatch does not exist. Generate it
        auto name = fn + ":dispatch";
        ExprPtr root; // Root function name used for calling
        auto a = ctx->cache->functions[overloads[0].name].ast;
        if (!a->attributes.parentClass.empty())
            root = N<DotExpr>(N<IdExpr>(a->attributes.parentClass),
                              ctx->cache->reverseIdentifierLookup[fn]);
        else
            root = N<IdExpr>(fn);
        root = N<CallExpr>(root, N<StarExpr>(N<IdExpr>("args")),
                           N<KeywordStarExpr>(N<IdExpr>("kwargs")));
        auto ast = N<FunctionStmt>(
                name, nullptr, std::vector<Param>{Param("*args"), Param("**kwargs")},
                N<SuiteStmt>(N<ReturnStmt>(root)), Attr({"autogenerated"}));
        ctx->cache->reverseIdentifierLookup[name] = ctx->cache->reverseIdentifierLookup[fn];

        auto baseType = getFuncTypeBase(2);
        auto typ = std::make_shared<FuncType>(baseType, ast.get());
        typ = std::static_pointer_cast<FuncType>(typ->generalize(ctx->typecheckLevel - 1));
        ctx->add(TypecheckItem::Func, name, typ);

        overloads.insert(overloads.begin(), {name, 0});
        ctx->cache->functions[name].ast = ast;
        ctx->cache->functions[name].type = typ;
        prependStmts->push_back(ast);
        return typ;
    }

    /// Transform a dot expression. Select the best method overload if possible.
    /// @param args (optional) list of class method arguments used to select the best
    ///             overload. nullptr if not available.
    /// @example
    ///   `obj.__class__`   -> `type(obj)`
    ///   `cls.__name__`    -> `"class"` (same for functions)
    ///   `obj.method`      -> `cls.method(obj, ...)` or
    ///                        `cls.method(obj)` if method has `@property` attribute
    ///   @c getClassMember examples:
    ///   `obj.GENERIC`     -> `GENERIC` (IdExpr with generic/static value)
    ///   `optional.member` -> `unwrap(optional).member`
    ///   `pyobj.member`    -> `pyobj._getattr("member")`
    /// @return nullptr if no transformation was made
    /// See @c getClassMember and @c getBestOverload
    ExprPtr TypecheckVisitor::transformDot(DotExpr *expr,
                                           std::vector<CallExpr::Arg> *args) {
        // Special case: obj.__class__
        if (expr->member == "__class__") {
            /// TODO: prevent cls.__class__ and type(cls)
            return transformType(NT<CallExpr>(NT<IdExpr>("type"), expr->expr));
        }

        transform(expr->expr);

        // Special case: fn.__name__
        // Should go before cls.__name__ to allow printing generic functions
        if (expr->expr->type->getFunc() && expr->member == "__name__") {
            return transform(N<StringExpr>(expr->expr->type->prettyString()));
        }
        // Special case: fn.__llvm_name__ or obj.__llvm_name__
        if (expr->member == "__llvm_name__") {
            if (realize(expr->expr->type))
                return transform(N<StringExpr>(expr->expr->type->realizedName()));
            return nullptr;
        }
        // Special case: cls.__name__
        if (expr->expr->isType() && expr->member == "__name__") {
            if (realize(expr->expr->type))
                return transform(N<StringExpr>(expr->expr->type->prettyString()));
            return nullptr;
        }
        // Special case: expr.__is_static__
        if (expr->member == "__is_static__") {
            if (expr->expr->isDone())
                return transform(N<BoolExpr>(expr->expr->isStatic()));
            return nullptr;
        }
        // Special case: cls.__id__
        if (expr->expr->isType() && expr->member == "__id__") {
            if (auto c = realize(expr->expr->type))
                return transform(N<IntExpr>(ctx->cache->classes[c->getClass()->name]
                                                    .realizations[c->getClass()->realizedTypeName()]
                                                    ->id));
            return nullptr;
        }

        // Ensure that the type is known (otherwise wait until it becomes known)
        auto typ = expr->expr->getType()->getClass();
        if (!typ)
            return nullptr;

        // Check if this is a method or member access
        if (ctx->findMethod(typ.get(), expr->member).empty())
            return getClassMember(expr, args);
        auto bestMethod = getBestOverload(expr, args);

        if (args) {
            unify(expr->type, ctx->instantiate(bestMethod, typ));

            // A function is deemed virtual if it is marked as such and
            // if a base class has a RTTI
            bool isVirtual = in(ctx->cache->classes[typ->name].virtuals, expr->member);
            isVirtual &= ctx->cache->classes[typ->name].rtti;
            isVirtual &= !expr->expr->isType();
            if (isVirtual && !bestMethod->ast->attributes.has(Attr::StaticMethod) &&
                !bestMethod->ast->attributes.has(Attr::Property)) {
                // Special case: route the call through a vtable
                if (realize(expr->type)) {
                    auto fn = expr->type->getFunc();
                    auto vid = getRealizationID(typ.get(), fn.get());

                    // Function[Tuple[TArg1, TArg2, ...], TRet]
                    std::vector<ExprPtr> ids;
                    for (auto &t: fn->getArgTypes())
                        ids.push_back(NT<IdExpr>(t->realizedName()));
                    auto fnType = NT<InstantiateExpr>(
                            NT<IdExpr>("Function"),
                            std::vector<ExprPtr>{NT<InstantiateExpr>(NT<IdExpr>(TYPE_TUPLE), ids),
                                                 NT<IdExpr>(fn->getRetType()->realizedName())});
                    // Function[Tuple[TArg1, TArg2, ...],TRet](
                    //    __internal__.class_get_rtti_vtable(expr)[T[VIRTUAL_ID]]
                    // )
                    auto e = N<CallExpr>(
                            fnType,
                            N<IndexExpr>(N<CallExpr>(N<IdExpr>("__internal__.class_get_rtti_vtable:0"),
                                                     expr->expr),
                                         N<IntExpr>(vid)));
                    return transform(e);
                }
            }
        }

        // Check if a method is a static or an instance method and transform accordingly
        if (expr->expr->isType() || args) {
            // Static access: `cls.method`
            ExprPtr e = N<IdExpr>(bestMethod->ast->name);
            unify(e->type, unify(expr->type, ctx->instantiate(bestMethod, typ)));
            return transform(e); // Realize if needed
        } else {
            // Instance access: `obj.method`
            // Transform y.method to a partial call `type(obj).method(args, ...)`
            std::vector<ExprPtr> methodArgs;
            // Do not add self if a method is marked with @staticmethod
            if (!bestMethod->ast->attributes.has(Attr::StaticMethod))
                methodArgs.push_back(expr->expr);
            // If a method is marked with @property, just call it directly
            if (!bestMethod->ast->attributes.has(Attr::Property))
                methodArgs.push_back(N<EllipsisExpr>(EllipsisExpr::PARTIAL));
            auto e = transform(N<CallExpr>(N<IdExpr>(bestMethod->ast->name), methodArgs));
            unify(expr->type, e->type);
            return e;
        }
    }

    /// Select the requested class member.
    /// @param args (optional) list of class method arguments used to select the best
    ///             overload if the member is optional. nullptr if not available.
    /// @example
    ///   `obj.GENERIC`     -> `GENERIC` (IdExpr with generic/static value)
    ///   `optional.member` -> `unwrap(optional).member`
    ///   `pyobj.member`    -> `pyobj._getattr("member")`
    ExprPtr TypecheckVisitor::getClassMember(DotExpr *expr,
                                             std::vector<CallExpr::Arg> *args) {
        auto typ = expr->expr->getType()->getClass();
        seqassert(typ, "not a class");

        // Case: object member access (`obj.member`)
        if (!expr->expr->isType()) {
            if (auto member = ctx->findMember(typ, expr->member)) {
                unify(expr->type, ctx->instantiate(member, typ));
                if (expr->expr->isDone() && realize(expr->type))
                    expr->setDone();
                return nullptr;
            }
        }

        // Case: class variable (`Cls.var`)
        if (auto cls = in(ctx->cache->classes, typ->name))
            if (auto var = in(cls->classVars, expr->member)) {
                return transform(N<IdExpr>(*var));
            }

        // Case: special members
        if (auto mtyp = findSpecialMember(expr->member)) {
            unify(expr->type, mtyp);
            if (expr->expr->isDone() && realize(expr->type))
                expr->setDone();
            return nullptr;
        }

        // Case: object generic access (`obj.T`)
        TypePtr generic = nullptr;
        for (auto &g: typ->generics)
            if (ctx->cache->reverseIdentifierLookup[g.name] == expr->member) {
                generic = g.type;
                break;
            }
        if (generic) {
            unify(expr->type, generic);
            if (!generic->isStaticType()) {
                expr->markType();
            } else {
                expr->staticValue.type = StaticValue::Type(generic->isStaticType());
            }
            if (realize(expr->type)) {
                if (!generic->isStaticType()) {
                    return transform(N<IdExpr>(generic->realizedName()));
                } else if (generic->getStatic()->expr->staticValue.type == StaticValue::STRING) {
                    expr->type = nullptr; // to prevent unify(T, Static[T]) error
                    return transform(
                            N<StringExpr>(generic->getStatic()->expr->staticValue.getString()));
                } else {
                    expr->type = nullptr; // to prevent unify(T, Static[T]) error
                    return transform(N<IntExpr>(generic->getStatic()->expr->staticValue.getInt()));
                }
            }
            return nullptr;
        }

        // Case: transform `optional.member` to `unwrap(optional).member`
        if (typ->is(TYPE_OPTIONAL)) {
            auto dot = N<DotExpr>(transform(N<CallExpr>(N<IdExpr>(FN_UNWRAP), expr->expr)),
                                  expr->member);
            dot->setType(ctx->getUnbound()); // as dot is not transformed
            if (auto d = transformDot(dot.get(), args))
                return d;
            return dot;
        }

        // Case: transform `pyobj.member` to `pyobj._getattr("member")`
        if (typ->is("pyobj")) {
            return transform(
                    N<CallExpr>(N<DotExpr>(expr->expr, "_getattr"), N<StringExpr>(expr->member)));
        }

        // Case: transform `union.m` to `__internal__.get_union_method(union, "m", ...)`
        if (typ->getUnion()) {
            if (!typ->canRealize())
                return nullptr; // delay!
            // bool isMember = false;
            // for (auto &t: typ->getUnion()->getRealizationTypes())
            //   if (ctx->findMethod(t.get(), expr->member).empty())
            return transform(N<CallExpr>(
                    N<IdExpr>("__internal__.union_member:0"),
                    std::vector<CallExpr::Arg>{{"union",  expr->expr},
                                               {"member", N<StringExpr>(expr->member)}}));
        }

        // For debugging purposes:
        // ctx->findMethod(typ.get(), expr->member);
        E(Error::DOT_NO_ATTR, expr, typ->prettyString(), expr->member);
        return nullptr;
    }

    TypePtr TypecheckVisitor::findSpecialMember(const std::string &member) {
        if (member == "__elemsize__")
            return ctx->getType("int");
        if (member == "__atomic__")
            return ctx->getType("bool");
        if (member == "__contents_atomic__")
            return ctx->getType("bool");
        if (member == "__name__")
            return ctx->getType("str");
        return nullptr;
    }

    /// Select the best overloaded function or method.
    /// @param expr    a DotExpr (for methods) or an IdExpr (for overloaded functions)
    /// @param methods List of available methods.
    /// @param args    (optional) list of class method arguments used to select the best
    ///                overload if the member is optional. nullptr if not available.
    FuncTypePtr TypecheckVisitor::getBestOverload(Expr *expr,
                                                  std::vector<CallExpr::Arg> *args) {
        // Prepare the list of method arguments if possible
        std::unique_ptr<std::vector<CallExpr::Arg>> methodArgs;

        if (args) {
            // Case: method overloads (DotExpr)
            bool addSelf = true;
            if (auto dot = expr->getDot()) {
                auto methods =
                        ctx->findMethod(dot->expr->type->getClass().get(), dot->member, false);
                if (!methods.empty() && methods.front()->ast->attributes.has(Attr::StaticMethod))
                    addSelf = false;
            }

            // Case: arguments explicitly provided (by CallExpr)
            if (addSelf && expr->getDot() && !expr->getDot()->expr->isType()) {
                // Add `self` as the first argument
                args->insert(args->begin(), {"", expr->getDot()->expr});
            }
            methodArgs = std::make_unique<std::vector<CallExpr::Arg>>();
            for (auto &a: *args)
                methodArgs->push_back(a);
        } else {
            // Partially deduced type thus far
            auto typeSoFar = expr->getType() ? expr->getType()->getClass() : nullptr;
            if (typeSoFar && typeSoFar->getFunc()) {
                // Case: arguments available from the previous type checking round
                methodArgs = std::make_unique<std::vector<CallExpr::Arg>>();
                if (expr->getDot() && !expr->getDot()->expr->isType()) { // Add `self`
                    auto n = N<NoneExpr>();
                    n->setType(expr->getDot()->expr->type);
                    methodArgs->push_back({"", n});
                }
                for (auto &a: typeSoFar->getFunc()->getArgTypes()) {
                    auto n = N<NoneExpr>();
                    n->setType(a);
                    methodArgs->push_back({"", n});
                }
            }
        }

        bool goDispatch = methodArgs == nullptr;
        if (!goDispatch) {
            std::vector<FuncTypePtr> m;
            // Use the provided arguments to select the best method
            if (auto dot = expr->getDot()) {
                // Case: method overloads (DotExpr)
                auto methods =
                        ctx->findMethod(dot->expr->type->getClass().get(), dot->member, false);
                m = findMatchingMethods(dot->expr->type->getClass(), methods, *methodArgs);
            } else if (auto id = expr->getId()) {
                // Case: function overloads (IdExpr)
                std::vector<types::FuncTypePtr> methods;
                for (auto &m: ctx->cache->overloads[id->value])
                    if (!endswith(m.name, ":dispatch"))
                        methods.push_back(ctx->cache->functions[m.name].type);
                std::reverse(methods.begin(), methods.end());
                m = findMatchingMethods(nullptr, methods, *methodArgs);
            }

            if (m.size() == 1) {
                return m[0];
            } else if (m.size() > 1) {
                for (auto &a: *methodArgs) {
                    if (auto u = a.value->type->getUnbound()) {
                        goDispatch = true;
                    }
                }
                if (!goDispatch)
                    return m[0];
            }
        }

        if (goDispatch) {
            // If overload is ambiguous, route through a dispatch function
            std::string name;
            if (auto dot = expr->getDot()) {
                name = ctx->cache->getMethod(dot->expr->type->getClass(), dot->member);
            } else {
                name = expr->getId()->value;
            }
            return getDispatch(name);
        }

        // Print a nice error message
        std::string argsNice;
        if (methodArgs) {
            std::vector<std::string> a;
            for (auto &t: *methodArgs)
                a.emplace_back(fmt::format("{}", t.value->type->prettyString()));
            argsNice = fmt::format("({})", fmt::join(a, ", "));
        }

        if (auto dot = expr->getDot()) {
            E(Error::DOT_NO_ATTR_ARGS, expr, dot->expr->type->prettyString(), dot->member,
              argsNice);
        } else {
            E(Error::FN_NO_ATTR_ARGS, expr, ctx->cache->rev(expr->getId()->value), argsNice);
        }

        return nullptr;
    }

} // namespace hercules::ast
