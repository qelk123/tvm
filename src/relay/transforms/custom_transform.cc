/*
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
 * \file legalize.cc
 * \brief Converts an expr to another expr. This pass can be used to transform an op based on its
 * shape, dtype or layout to another op or a sequence of ops.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>
#include "../op/memory/on_device.h"
#include "../op/memory/device_copy.h"

namespace tvm {
namespace relay {

namespace custom_transform {

// Call registered FTVMLegalize of an op
// Returns the legalized expression
class transformer : public ExprRewriter {
 public:
  explicit transformer(const Array<VirtualDevice>& device_list)
  {
    for(auto i:device_list)
    {
      device_set_.insert(i);
      device_map_[i->device_type()]=i;
    }
  }

  Expr Rewrite_(const CallNode* call_node, const Expr& post) override {
    // Get the new_call node without any changes to current call node.
    Call new_call = Downcast<Call>(post);
    auto call_op = call_node->op;
    if (call_op.as<OpNode>()) {
      
      if(call_node->op.as<OpNode>()->name=="add"&&device_map_.count(kDLCPU)&&device_map_.count(kDLCUDA))
      {
        auto tmp1 = Call(Op::Get("add"), call_node->args, call_node->attrs );
        auto attrs_l = make_object<OnDeviceAttrs>();
        attrs_l->virtual_device = device_map_[kDLCPU];
        attrs_l->constrain_result = false;
        attrs_l->constrain_body = true;
        auto lhs=Call(OnDeviceOp(), {std::move(tmp1)}, Attrs(std::move(attrs_l)));


        std::vector<Expr> new_arg_for_device;
        for(auto i : call_node->args)
        {
          // if(i.as<ConstantNode>())
          // {
          //   new_arg_for_device.push_back(std::move(i));
          // }
          // else
          {
            new_arg_for_device.push_back(DeviceCopy(i, device_map_[kDLCPU], device_map_[kDLCUDA]));
          }
        }
        auto tmp2 = Call(Op::Get("add"), new_arg_for_device, call_node->attrs );
        auto attrs_r = make_object<OnDeviceAttrs>();
        attrs_r->virtual_device = device_map_[kDLCUDA];
        attrs_r->constrain_result = false;
        attrs_r->constrain_body = true;
        auto rhs=Call(OnDeviceOp(), {std::move(tmp2)}, Attrs(std::move(attrs_r)));


        auto add = Call(Op::Get("add"), {std::move(lhs),std::move(rhs)}, call_node->attrs );
        auto attrs = make_object<OnDeviceAttrs>();
        attrs->virtual_device = device_map_[kDLCPU];
        attrs->constrain_result = false;
        attrs->constrain_body = true;
        auto new_post=Call(OnDeviceOp(), {std::move(add)}, Attrs(std::move(attrs)));
        return new_post;
      }

    }
    return post;

  }
  private:
    std::set<VirtualDevice> device_set_={};
    std::map<DLDeviceType,VirtualDevice> device_map_={};
};

Expr custom_transform(const Expr& expr,const Array<VirtualDevice>& device_list) {
  auto rewriter = transformer(device_list);
  return PostOrderRewrite(expr, &rewriter);
}

}  // namespace legalize

namespace transform {

Pass Custom_transform(const Array<VirtualDevice>& device_list) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::custom_transform::custom_transform(f,device_list));
      };
  return CreateFunctionPass(pass_func, 1, "custom_transform", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.custom_transform").set_body_typed(Custom_transform);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
