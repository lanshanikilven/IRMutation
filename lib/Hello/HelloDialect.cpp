// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace hello;

//===----------------------------------------------------------------------===//
// Hello dialect.
//===----------------------------------------------------------------------===//

#include "Hello/HelloOpsDialect.cpp.inc"

void HelloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Hello/HelloOps.cpp.inc"
      >();
}

void hello::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  hello::ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::Operation *HelloDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
    return builder.create<hello::ConstantOp>(loc, type,
                                      value.cast<mlir::DenseElementsAttr>());
}

//?????????????????????????????????
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<hello::TransposeOp> {
  using OpRewritePattern<hello::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hello::TransposeOp op,
                                PatternRewriter &rewriter) const final {
    mlir::Value transposeInput = op.getOperand();
    hello::TransposeOp transposeInputOp = transposeInput.getDefiningOp<hello::TransposeOp>();
    if (!transposeInputOp) {
      return failure();
    }
  rewriter.replaceOp(op, {transposeInputOp.getOperand()});
  return success();
  }
};

//???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????h.inc???????????????????????????????????????Ops.td????????????????????????
//let hasCanonicalizer = 1;????????????tablegen??????????????????????????????
void TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                              MLIRContext *context) {
  // SimplifyRedundantTranspose ???????????????????????????????????????
  results.insert<SimplifyRedundantTranspose>(context);
}

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}
