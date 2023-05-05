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

// #include "bfs_pprocess.h"

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
TVM_REGISTER_NODE_TYPE(CustomBfs_PprocessAttrs);
bool Bfs_Pprocess_equalRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 4) << "Expects six types, four for the input and two for the output";
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
  const auto* data3 = types[2].as<TensorTypeNode>();
  if (data3 == nullptr) {
      ICHECK(types[2].as<IncompleteTypeNode>())
      << "Scanop: expect input type to be TensorType but get " << types[2];
      return false;
  }
  auto tensor1 = TensorType(data2->shape, data2->dtype);
  reporter->Assign(types[3], tensor1);
  return true;
}
//matrix, cur_layer, visited, layer
Expr MakeBfs_Pprocess(Expr V,Expr Pi,Expr i) {
  auto attrs = make_object<CustomBfs_PprocessAttrs>();
  static const Op& op = Op::Get("custom.bfs_pprocess");
  return Call(op, {V,Pi,i}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.custom._make.bfs_pprocess").set_body_typed(MakeBfs_Pprocess);


RELAY_REGISTER_OP("custom.bfs_pprocess")
    .describe(R"code(bfs post process)code" TVM_ADD_FILELINE)
    .set_attrs_type<CustomBfs_PprocessAttrs>()
    .set_num_inputs(3)
    .add_argument("V", "1D Tensor", "Input data1.")
    .add_argument("Pi", "1D Tensor", "Input data2.")
    .add_argument("i", "0D Tensor", "Input data3.")
    .set_support_level(1)
    .add_type_rel("bfs_pprocess", Bfs_Pprocess_equalRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
}
}