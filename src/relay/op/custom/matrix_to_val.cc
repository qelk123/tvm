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

#include "matrix_to_val.h"

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/custom.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/custom/matrix_to_val.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../make_op.h"


namespace tvm {
namespace relay {
  TVM_REGISTER_NODE_TYPE(CustomMatrix_to_valAttrs);
  bool Matrix_to_valRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
    // types: [data, output]
    ICHECK_EQ(types.size(), 2) << "Expects six types, four for the input and two for the output";
    const auto* data = types[0].as<TensorTypeNode>(); //输入的tensor信息
    if (data == nullptr) {
        ICHECK(types[0].as<IncompleteTypeNode>())
        << "Scanop: expect input type to be TensorType but get " << types[0];
        return false;
    }

    auto ele_type = TensorType::Scalar(data->dtype);
    reporter->Assign(types[1],  ele_type);

    return true;
}
//matrix, cur_layer, visited, layer
Expr MakeMatrix_to_val(Expr matrix) {
  auto attrs = make_object<CustomMatrix_to_valAttrs>();
  static const Op& op = Op::Get("custom.matrix_to_val");
  return Call(op, {matrix}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.custom._make.matrix_to_val").set_body_typed(MakeMatrix_to_val);


RELAY_REGISTER_OP("custom.matrix_to_val")
    .describe(R"code(Add bias to an axis of the input.)code" TVM_ADD_FILELINE)
    .set_attrs_type<CustomMatrix_to_valAttrs>()
    .set_num_inputs(1)
    .add_argument("matrix", "1D Tensor", "Input data1.")
    // .add_argument("val", "float", "Input data2.")
    .set_support_level(1)
    .add_type_rel("Matrix_to_val", Matrix_to_valRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
}

}