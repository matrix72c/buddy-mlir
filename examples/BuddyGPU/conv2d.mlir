!unit = f32
!lhs = tensor<1x1x5376x2048x!unit>
!rhs = tensor<1x1x3x3x!unit>
!res = tensor<1x1x5374x2046x!unit>

func.func @conv_2d(%arg0: !lhs, %arg1: !rhs) -> !res {
  %0 = tensor.empty() : !res
  %2 = linalg.conv_2d_nchw_fchw ins (%arg0, %arg1: !lhs, !rhs) outs (%0: !res) -> !res
  func.return %2 : !res
}