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

// #include "tensor_ana.h"

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
TVM_REGISTER_NODE_TYPE(CustomTensor_AnaAttrs);
bool Tensor_Ana_equalRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 2) << "Expects six types, four for the input and two for the output";
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
      ICHECK(types[0].as<IncompleteTypeNode>())
      << "Scanop: expect input type to be TensorType but get " << types[0];
      return false;
  }
  Array<IndexExpr> oshape({2,data->shape[0]});
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}
//matrix, cur_layer, visited, layer
Expr MakeTensor_Ana(Expr S_M) {
  auto attrs = make_object<CustomTensor_AnaAttrs>();
  static const Op& op = Op::Get("custom.tensor_ana");
  return Call(op, {S_M}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.custom._make.tensor_ana").set_body_typed(MakeTensor_Ana);


RELAY_REGISTER_OP("custom.tensor_ana")
    .describe(R"code(analysis the sparse tensor and split to two dynamic shape tensor)code" TVM_ADD_FILELINE)
    .set_attrs_type<CustomTensor_AnaAttrs>()
    .set_num_inputs(1)
    .add_argument("S_M", "2D Tensor", "Input data1.")
    .set_support_level(1)
    .add_type_rel("tensor_ana", Tensor_Ana_equalRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
}

}