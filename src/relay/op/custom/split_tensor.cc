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
 * \file liuyn.cc
 * \brief Property def of liuyn operators.
 */

// #include "split_tensor.h"

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/custom.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
// #include <tvm/topi/custom/tensor_to_val_equal.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../make_op.h"


namespace tvm {
namespace relay {
TVM_REGISTER_NODE_TYPE(CustomSplit_TensorAttrs);
bool Split_Tensor_equalRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 3) << "Expects six types, four for the input and two for the output";
  const auto* data1 = types[0].as<TensorTypeNode>();
  if (data1 == nullptr) {
      ICHECK(types[0].as<IncompleteTypeNode>())
      << "Scanop: expect input type to be TensorType but get " << types[0];
      return false;
  }
  const auto* data2 = types[1].as<TensorTypeNode>();
  if (data2 == nullptr) {
      ICHECK(types[1].as<IncompleteTypeNode>())
      << "Scanop: expect input type to be TensorType but get " << types[1];
      return false;
  }
  Array<IndexExpr> oshape({Any(),data1->shape[1]});
  auto tensor1 = TensorType(oshape, data1->dtype);
  auto tensor2 = TensorType(oshape, data1->dtype);
  reporter->Assign(types[2], TupleType({tensor1, tensor2}));
  return true;
}
//matrix, cur_layer, visited, layer
Expr MakeSplit_Tensor(Expr S_M,Expr S_info) {
  auto attrs = make_object<CustomSplit_TensorAttrs>();
  static const Op& op = Op::Get("custom.split_tensor");
  return Call(op, {S_M,S_info}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.custom._make.split_tensor").set_body_typed(MakeSplit_Tensor);


RELAY_REGISTER_OP("custom.split_tensor")
    .describe(R"code(split tensor using split info)code" TVM_ADD_FILELINE)
    .set_attrs_type<CustomSplit_TensorAttrs>()
    .set_num_inputs(2)
    .add_argument("S_M", "2D Tensor", "Input data1.")
    .add_argument("S_info", "2D Tensor", "Input data2.")
    .set_support_level(1)
    .add_type_rel("split_tensor", Split_Tensor_equalRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
}

}