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

#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"
#include "Hello/HelloPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace hello;

static mlir::MemRefType convertTensorToMemRef(mlir::TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

// using LoopIterationFn = function_ref<Value(
//     OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs)>;

// static void lowerOpToLoops(Operation *op, ArrayRef<Value> operands,
//                            PatternRewriter &rewriter,
//                            LoopIterationFn processIteration) {
//   auto tensorType = (*op->result_type_begin()).cast<TensorType>();
//   auto loc = op->getLoc();

//   // Insert an allocation and deallocation for the result of this operation.
//   auto memRefType = convertTensorToMemRef(tensorType);
//   auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

//   // Create a nest of affine loops, with one loop per dimension of the shape.
//   // The buildAffineLoopNest function takes a callback that is used to construct
//   // the body of the innermost loop given a builder, a location and a range of
//   // loop induction variables.
//   SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
//   SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
//   buildAffineLoopNest(
//       rewriter, loc, lowerBounds, tensorType.getShape(), steps,
//       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
//         // Call the processing function with the rewriter, the memref operands,
//         // and the loop induction variables. This function will return the value
//         // to store at the current index.
//         Value valueToStore = processIteration(nestedBuilder, operands, ivs);
//         nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
//       });

//   // Replace this operation with the generated alloc.
//   rewriter.replaceOp(op, alloc);
// }

// struct TransposeOpLowering : public mlir::ConversionPattern {
//   TransposeOpLowering(mlir::MLIRContext *ctx)
//       : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, ctx) {}
//   // 匹配和重写函数
//   mlir::LogicalResult
//   matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
//                   mlir::ConversionPatternRewriter &rewriter) const final {
//     auto loc = op->getLoc();
//     // 实现将当前的操作lower到一组仿射循环
//     // memRef是AffineDialect的操作数类型，类似于缓冲区
//     lowerOpToLoops(
//         op, operands, rewriter,
//         [loc](mlir::PatternRewriter &rewriter,
//               ArrayRef<mlir::Value> memRefOperands,
//               ArrayRef<mlir::Value> loopIvs) {
//           // TransposeOpAdaptor 是在ODS框架执行后自动生成的
//           TransposeOpAdaptor transposeAdaptor(memRefOperands);
//           mlir::Value input = transposeAdaptor.getInput();
//           SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
//           return rewriter.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
//         });
//     return success();
//   }
// };


class ConstantOpLowering : public mlir::OpRewritePattern<hello::ConstantOp> {
  using OpRewritePattern<hello::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(hello::ConstantOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::DenseElementsAttr constantValue = op.getValue();
    mlir::Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<mlir::TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    mlir::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
          0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
    }
    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.

    // [4, 3] (1, 2, 3, 4, 5, 6, 7, 8)
    // storeElements(0)
    //   indices = [0]
    //   storeElements(1)
    //     indices = [0, 0]
    //     storeElements(2)
    //       store (const 1) [0, 0]
    //     indices = [0]
    //     indices = [0, 1]
    //     storeElements(2)
    //       store (const 2) [0, 1]
    //  ...
    //
    mlir::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.getValues<mlir::FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::AffineStoreOp>(
            loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc); //这里采用了replaceOp方法，将原来的op替换成新的op，与此对应的还有一个替换函数，名称为replaceOpWithNewOp
    return mlir::success();
  }
};

class PrintOpLowering : public mlir::OpConversionPattern<hello::PrintOp> {
  using OpConversionPattern<hello::PrintOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(hello::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
      // We don't lower "hello.print" in this pass, but we need to update its
      // operands.
      //下面这个函数可以就地更新操作的属性，位置，操作数或者后继者，
      //这个函数内部的机制是，从startRootUpdate 启动，分别用 cancelRootUpdate 和 finalizeRootUpdate 取消或终止。
      rewriter.updateRootInPlace(op,
                                 [&] { op->setOperands(adaptor.getOperands()); });
      return mlir::success();
  }
};


namespace {
class HelloToAffineLowerPass : public mlir::PassWrapper<HelloToAffineLowerPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HelloToAffineLowerPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
      registry.insert<mlir::AffineDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final;
};
}

void HelloToAffineLowerPass::runOnOperation() {

  mlir::ConversionTarget target(getContext()); //这里首先定义了一个转换目标，getContext()函数是Pass基类中的一个纯虚函数，所有自定义的Pass都会继承自该Pass基类
  //这里声明了lower过程中能够合法使用的dialect，需要将非法dialect中的op转换到这些dialect中的合法op
  target.addLegalDialect<mlir::AffineDialect, mlir::BuiltinDialect,
    mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  // 下面这两个组合起来的作用是，需要将HelloDialect中的所有Op都转换到上面的合法dialect中，但是可以保留其中的printOp, 这就是允许属于不同Dialect的op共存的本质
  target.addIllegalDialect<hello::HelloDialect>();
  target.addDynamicallyLegalOp<hello::PrintOp>([](hello::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(),
                           [](mlir::Type type) { return type.isa<mlir::TensorType>(); });
  });
  //注意在MLIR里面，单个op的转换优先级是高于Dialect整体的
  target.addLegalOp<hello::TransposeOp>();
  //至此，我们相当于转换了namespace，但是还没有开始对dialect中的op进行实际的转换过程

 
  //在上面声明完转换目标之后，接下来就需要编写具体的转换规则，将Dialect中的非法op转换成目标Dialect中的合法op
  //这里使用了传统的OpRewritePattern，对应ConstantOpLowering和新的OpConversionPattern，对应PrintOpLowering两种方式来执行转换逻辑
  //OpConversionPattern与OpRewritePattern的不同之处在于，前者需要多接受一个Array<Operands>参数，用于在处理类型转换的时候，对旧类型进行匹配

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ConstantOpLowering, PrintOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    //这里执行部分转换，会保留没有被标记为illegal的operations，转换完成后，隶属于不同dialect的op可以共存在一个大的FuncOp或者ModuleOp中
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> hello::createLowerToAffinePass() {
  return std::make_unique<HelloToAffineLowerPass>();
}
