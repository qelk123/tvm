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

namespace custom_device_op_forward {

// Call registered FTVMLegalize of an op
// Returns the legalized expression
class Let_finder : public ExprMutator {
  public:
    explicit Let_finder(){ }

    std::vector<std::pair<std::pair<Let,int>,std::vector<Var>>> Let_need_mutate={};/*pair<Let,int>:root let node in subgraph and the layer of subgraph
                                                                    vector<Var>:all the var this subgraph need                      */

  private:
    Expr VisitExpr_(const VarNode* n)
    {
      std::cout<<"into here!\n";
      if(start_gathering)
        tmp_use_var.push_back(GetRef<Var>(n));
      return ExprMutator::VisitExpr_(n);
    }
    Expr VisitExpr_(const LetNode* n)
    {
      //TODO:deal with the situation about this let node is leaf node in the ast,whose body is not a let,a var for example
      if(n->body.as<LetNode>()){
        /*for function with a caller with on_device(GPU) in the next let node
        Let--var:%0
          |-value:Function
          `-body:Call--on device(GPU)
                      |-Call
                      |-var:(Use)%0
                      `-other vars
        */
        /*for a let node whose value is a callnode with on_device(GPU)*/
        if(n->value.as<FunctionNode>())
        {
          if(n->body.as<LetNode>()&&n->body.as<LetNode>()->value.as<CallNode>())
          {
            auto call=n->body.as<LetNode>()->value.as<CallNode>();
            auto on_op_info = GetOnDeviceProps(call);
            //for normal call
            if(on_op_info.body.defined()&&on_op_info.virtual_device->device_type()==kDLCUDA&&on_op_info.body.as<CallNode>())
            {
              auto child_call = on_op_info.body.as<CallNode>();
              if(child_call&&child_call->op==n->var)
              {
                start_gathering = true;
                tmp_use_var.clear();
                for(size_t i=0;i<child_call->args.size();i++){
                  this->memo_.clear();
                  ExprMutator::Mutate(child_call->args[i]);

                }
                // ExprMutator::Mutate(child_call->op);
                start_gathering = false;
                for(auto i:tmp_use_var)
                {
                  std::cout<<"tmp_use_var:\n"<<PrettyPrint(i)<<"\n";
                }
                std::cout<<"\ninsert:\n";
                std::cout<<PrettyPrint(GetRef<Let>(n));
                Let_need_mutate.push_back({{GetRef<Let>(n),2},tmp_use_var});
                return ExprMutator::Mutate(n->body.as<LetNode>()->body);
              }
            }
            else if(on_op_info.body.defined()&&on_op_info.virtual_device->device_type()==kDLCPU&&on_op_info.body.as<CallNode>())
            {
              auto child_call = on_op_info.body.as<CallNode>();
              if(auto call_in_function = n->value.as<FunctionNode>()->body.as<CallNode>()){
                if(GetDeviceCopyProps(call_in_function).src_virtual_device->device_type()==kDLCUDA&&
                    GetDeviceCopyProps(call_in_function).dst_virtual_device->device_type()==kDLCPU)
                {
                  ICHECK(child_call&&child_call->op==n->var)<<
                      "for device copy op from GPU to HOST,we only support device copy call immediately after the device copy function";
                  start_gathering = true;
                  tmp_use_var.clear();
                  for(size_t i=0;i<child_call->args.size();i++){
                    this->memo_.clear();
                    ExprMutator::Mutate(child_call->args[i]);

                  }
                  for(auto i:tmp_use_var)
                  {
                    std::cout<<"tmp_use_var:\n"<<PrettyPrint(i)<<"\n";
                  }
                  // ExprMutator::Mutate(child_call->op);
                  start_gathering = false;
                  std::cout<<"\ninsert:\n";
                  std::cout<<PrettyPrint(GetRef<Let>(n));
                  Let_need_mutate.push_back({{GetRef<Let>(n),2},tmp_use_var});
                  return ExprMutator::Mutate(n->body.as<LetNode>()->body);
                }
              }
            }

          }
        }
        else if(n->value.as<CallNode>())
        {
          auto on_op_info = GetOnDeviceProps(n->value.as<CallNode>());
          if(on_op_info.body.defined()&&on_op_info.virtual_device->device_type()==kDLCUDA)
          {
            auto child_call = on_op_info.body.as<CallNode>();
            if(child_call&&child_call->op==n->var)
            {
              start_gathering = true;
              tmp_use_var.clear();
              for(size_t i=1;i<child_call->args.size();i++){
                this->memo_.clear();
                ExprMutator::Mutate(child_call->args[i]);
              }
              ExprMutator::Mutate(child_call->op);
              for(auto i:tmp_use_var)
              {
                std::cout<<"tmp_use_var:\n"<<PrettyPrint(i)<<"\n";
              }
              start_gathering = false;
              std::cout<<"\ninsert:\n";
              std::cout<<PrettyPrint(GetRef<Let>(n));
              Let_need_mutate.push_back({{GetRef<Let>(n),1},tmp_use_var});
              return ExprMutator::Mutate(n->body);
            }
          }
          else if(on_op_info.body.defined()&&on_op_info.virtual_device->device_type()==kDLCPU&&
          on_op_info.body.as<TupleGetItemNode>())
          {
            std::cout<<"TupleGetItemNode\n";
            auto child_tuple = on_op_info.body.as<TupleGetItemNode>();
            // if(child_tuple&&child_tuple->tuple==n->var)
            {
              start_gathering = true;
              tmp_use_var.clear();
              // for(size_t i=1;i<child_tuple->args.size();i++){
              this->memo_.clear();
              ExprMutator::Mutate(child_tuple->tuple);
              // }
              for(auto i:tmp_use_var)
              {
                std::cout<<"TupleGetItemNode tmp_use_var:\n"<<PrettyPrint(i)<<"\n";
              }
              start_gathering = false;
              std::cout<<"\ninsert:\n";
              std::cout<<PrettyPrint(GetRef<Let>(n));
              Let_need_mutate.push_back({{GetRef<Let>(n),1},tmp_use_var});
              return ExprMutator::Mutate(n->body);
            }
          }
        }
      }
      return ExprMutator::VisitExpr_(n);
    }

    std::vector<Var> tmp_use_var={};
    bool start_gathering=false;
};

class Let_inserter : public ExprMutator {
  public:
    explicit Let_inserter(std::vector<std::pair<std::pair<Let,int>,std::vector<Var>>> Let_need_mutate):Let_need_insert(Let_need_mutate){ }
  private:
    Expr VisitExpr_(const VarNode* n)
    {
      def_or_use.insert(GetRef<Var>(n));
      return ExprMutator::VisitExpr_(n);
    }
    Expr VisitExpr_(const LetNode* n)
    {
      if(Let_need_insert.size())
      {
        auto first_candidate = Let_need_insert.begin();
        bool can_insert=true;
        for(auto i:first_candidate->second)
        {
          if(std::find(def_or_use.begin(),def_or_use.end(),i)==def_or_use.end()){
            can_insert=false;
            break;
          }
        }
        std::cout<<"can_insert!!\n";
        for(auto i:first_candidate->second)
        {
          std::cout<<PrettyPrint(i)<<"    ";
        }
        std::cout<<"\n"<<def_or_use.size()<<"\n";

        if(can_insert)
        {
          auto  tmp = WithFields(GetRef<Let>(n) , n->var, n->value,{});
          std::cout<<"current let:\n"<<PrettyPrint(n->var)<<"\n";
          std::cout<<"current let:\n"<<PrettyPrint(n->value)<<"\n";
          std::cout<<"insert!!\n";
          auto tmp_let_pair = *Let_need_insert.begin();
          Let_need_insert.erase(Let_need_insert.begin());
          int layer=0;
          auto tmp_let=tmp_let_pair.first.first;
          while(tmp_let.defined()&&layer<tmp_let_pair.first.second)
          {
            this->Mutate(tmp_let->value);
            this->Mutate(tmp_let->var);
            tmp_let=Downcast<Let>(tmp_let->body);
            layer++;
          }
          auto ret=ExprMutator::Mutate(GetRef<Expr>(n));
          ICHECK(tmp_let_pair.first.second<=2)<<"Let_need_insert layer should be 1 or 2";
          if(tmp_let_pair.first.second==2)
          {
            auto child_let = Downcast<Let>(tmp_let_pair.first.first->body);
            ret = WithFields(child_let , child_let->var, child_let->value, ret);
          }
            ret = WithFields(tmp_let_pair.first.first , tmp_let_pair.first.first->var, tmp_let_pair.first.first->value, ret);
          return ret;
        }
      }
      return ExprMutator::VisitExpr_(n);
    }

    std::set<Var> def_or_use={};
    std::vector<std::pair<std::pair<Let,int>,std::vector<Var>>> Let_need_insert={};/*pair<Let,int>:root let node in subgraph and the layer of subgraph
                                                                    vector<Var>:all the var this subgraph need                      */

};



Expr device_op_forward(const Expr& expr) {
  std::cout<<"before_device_op_forward:\n";
  std::cout<<PrettyPrint(Downcast<Function>(expr));
  auto finder = Let_finder();
  auto n_expr =  finder.Mutate(expr);
  std::cout<<"\nafter_finder:\n";
  std::cout<<PrettyPrint(Downcast<Function>(n_expr));
  for(auto i:finder.Let_need_mutate)
  {
    std::cout<<"\nfind let node:\n";
    std::cout<<PrettyPrint(i.first.first)<<"\n";
  }
  auto inserter = Let_inserter(finder.Let_need_mutate);
  auto ret = inserter.Mutate(n_expr);
  std::cout<<"\nafter_device_op_forward:\n";
  std::cout<<PrettyPrint(Downcast<Function>(ret));


  // IRModule tmp_module = IRModule({}, {}, {}, {}, {});
  // std::cout<<"\n------visualizer start-------\n";
  // tmp_module->Add(GlobalVar("main"), Downcast<Function>(ret), true);
  // static auto flower_call = tvm::runtime::Registry::Get("relay.visualizer");
  // ICHECK(flower_call) << "relay.visualizer is not registered.";
  // // std::cout<<"mod print:\n"<<PrettyPrint(context_.module)<<"\n";
  // {
  //   (*flower_call)(tmp_module);
  // }
  // std::cout<<"\n------visualizer end-------\n";

  return ret;
}

}  // namespace legalize

namespace transform {

Pass Custom_device_op_forward() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::custom_device_op_forward::device_op_forward(f));
      };
  return CreateFunctionPass(pass_func, 1, "custom_device_op_forward", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.custom_device_op_forward").set_body_typed(Custom_device_op_forward);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
