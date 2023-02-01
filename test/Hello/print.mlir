// RUN: hello-opt %s | FileCheck %s

// CHECK: define void @main()

//func.func @main() {
    //%0 = "hello.constant"() {value = dense<[[4.0, -3.7], [4.0, -5.2]]> : tensor<2x2xf64>} : () -> tensor<2x2xf64>
    //"hello.print"(%0) : (tensor<2x2xf64>) -> ()
    //return
//}

func.func @main(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = hello.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = hello.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  return %1 : tensor<*xf64>
}
