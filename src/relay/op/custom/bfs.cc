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

#include "bfs.h"

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/custom.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/custom/bfs.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../make_op.h"


namespace tvm {
namespace relay {
  TVM_REGISTER_NODE_TYPE(CustomBFSAttrs);
  bool CustomBFSRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
    // types: [data, output]
    ICHECK_EQ(types.size(), 5) << "Expects six types, four for the input and two for the output";
    const auto* data = types[1].as<TensorTypeNode>(); //输入的tensor信息
    if (data == nullptr) {
        ICHECK(types[1].as<IncompleteTypeNode>())
        << "Scanop: expect input type to be TensorType but get " << types[0];
        return false;
    }

    auto ele_type = TensorType(data->shape, DataType::Int(32));
    reporter->Assign(types[4],  ele_type);

    return true;
}
//matrix, cur_layer, visited, layer
Expr MakeCustomBFS(Expr matrix, Expr cur_layer, Expr visited, Expr layer) {
  auto attrs = make_object<CustomBFSAttrs>();
  static const Op& op = Op::Get("custom.custombfs");
  return Call(op, {matrix, cur_layer, visited, layer}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.custom._make.custombfs").set_body_typed(MakeCustomBFS);


RELAY_REGISTER_OP("custom.custombfs")
    .describe(R"code(Add bias to an axis of the input.)code" TVM_ADD_FILELINE)
    .set_attrs_type<CustomBFSAttrs>()
    .set_num_inputs(4)
    .add_argument("matrix", "2D Tensor", "Input data.")
    .add_argument("cur_layer", "1D Tensor", "Input data2.")
    .add_argument("visited", "1D Tensor", "Input data3.")
    .add_argument("layer", "int", "layer")
    .set_support_level(1)
    .add_type_rel("CustomBFS", CustomBFSRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
}

}