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
 * \brief bias_add op constructions
 * \file nn/bias_add.h
 */
#ifndef TVM_TOPI_CUSTOM_BFS_H_
#define TVM_TOPI_CUSTOM_BFS_H_

#include <tvm/te/operation.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/tags.h>
#include <tvm/topi/transform.h>

#include <string>

namespace tvm {
namespace topi {
namespace custom {

// /*!
//  * \brief Creates an operation that calculates data + bias
//  *
//  * \param data Tensor with shape [batch, in_dim]
//  * \param bias Tensor with shape [batch].
//  * \param axis The axis to add the bias to.
//  * \return Tensor with shape [batch, in_dim]
//  */
// inline tvm::te::Tensor bfs(const tvm::te::Tensor& data, const tvm::te::Tensor& bias,
//                                 int axis) {
//   return add(data, bias);
// }

inline Schedule schedule_bfs(const Target& target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  return s;
}
}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_BIAS_ADD_H_