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
 * \file tvm/relay/attrs/nn.h
 * \brief Auxiliary attributes for nn operators.
 */
#ifndef TVM_RELAY_ATTRS_CUSTOM_H_
#define TVM_RELAY_ATTRS_CUSTOM_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>

#include <string>

namespace tvm {
namespace relay {
    struct CustomBFSAttrs : public tvm::AttrsNode<CustomBFSAttrs>{
      TVM_DECLARE_ATTRS(CustomBFSAttrs ,"relay.attrs.CustomBFSAttrs"){
      }
    };
    struct CustomUpdatePiAttrs : public tvm::AttrsNode<CustomUpdatePiAttrs>{
      TVM_DECLARE_ATTRS(CustomUpdatePiAttrs ,"relay.attrs.CustomUpdatePiAttrs"){
      }
    };
    struct CustomTensor_to_val_equalAttrs : public tvm::AttrsNode<CustomTensor_to_val_equalAttrs>{
      TVM_DECLARE_ATTRS(CustomTensor_to_val_equalAttrs ,"relay.attrs.CustomTensor_to_val_equalAttrs"){
      }
    };
    struct CustomMatrix_to_valAttrs : public tvm::AttrsNode<CustomMatrix_to_valAttrs>{
      TVM_DECLARE_ATTRS(CustomMatrix_to_valAttrs ,"relay.attrs.CustomMatrix_to_valAttrs"){
      }
    };
    struct Customdynamic_output_tensorAttrs : public tvm::AttrsNode<Customdynamic_output_tensorAttrs>{
      TVM_DECLARE_ATTRS(Customdynamic_output_tensorAttrs ,"relay.attrs.Customdynamic_output_tensorAttrs"){
      }
    };
    struct Customsame_output_tensorAttrs : public tvm::AttrsNode<Customsame_output_tensorAttrs>{
      TVM_DECLARE_ATTRS(Customsame_output_tensorAttrs ,"relay.attrs.Customsame_output_tensorAttrs"){
      }
    };
    struct CustomTensor_AnaAttrs : public tvm::AttrsNode<CustomTensor_AnaAttrs>{
      TVM_DECLARE_ATTRS(CustomTensor_AnaAttrs ,"relay.attrs.CustomTensor_AnaAttrs"){
      }
    };
    struct CustomSplit_TensorAttrs : public tvm::AttrsNode<CustomSplit_TensorAttrs>{
      TVM_DECLARE_ATTRS(CustomSplit_TensorAttrs ,"relay.attrs.CustomSplit_TensorAttrs"){
      }
    };
    struct CustomSpmv_CpuAttrs : public tvm::AttrsNode<CustomSpmv_CpuAttrs>{
      TVM_DECLARE_ATTRS(CustomSpmv_CpuAttrs ,"relay.attrs.CustomSpmv_CpuAttrs"){
      }
    };
    struct CustomSpmv_GpuAttrs : public tvm::AttrsNode<CustomSpmv_GpuAttrs>{
      TVM_DECLARE_ATTRS(CustomSpmv_GpuAttrs ,"relay.attrs.CustomSpmv_GpuAttrs"){
      }
    };
    struct CustomConcat_VectorAttrs : public tvm::AttrsNode<CustomConcat_VectorAttrs>{
      TVM_DECLARE_ATTRS(CustomConcat_VectorAttrs ,"relay.attrs.CustomConcat_VectorAttrs"){
      }
    };
    struct CustomBfs_PprocessAttrs : public tvm::AttrsNode<CustomBfs_PprocessAttrs>{
      TVM_DECLARE_ATTRS(CustomBfs_PprocessAttrs ,"relay.attrs.CustomBfs_PprocessAttrs"){
      }
    };
  }
}
#endif