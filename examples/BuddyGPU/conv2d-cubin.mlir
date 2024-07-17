#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 256)>
module {
  func.func @conv_2d(%arg0: memref<1x1x5376x2048xf32>, %arg1: memref<1x1x3x3xf32>) -> memref<1x1x5374x2046xf32> {
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x5374x2046xf32>
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c1, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c2, %arg13 = %c1) {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = affine.apply #map(%1)
      %3 = affine.apply #map1(%0)
      %subview = memref.subview %arg0[%2, 0, 0, 0] [1, 1, 5376, 2048] [1, 1, 1, 1] : memref<1x1x5376x2048xf32> to memref<1x1x5376x2048xf32, strided<[11010048, 11010048, 2048, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%3, 0, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : memref<1x1x3x3xf32> to memref<1x1x3x3xf32, strided<[9, 9, 3, 1], offset: ?>>
      %subview_1 = memref.subview %alloc[%2, %3, 0, 0] [1, 1, 5374, 2046] [1, 1, 1, 1] : memref<1x1x5374x2046xf32> to memref<1x1x5374x2046xf32, strided<[10995204, 10995204, 2046, 1], offset: ?>>
      linalg.conv_2d_nchw_fchw ins(%subview, %subview_0 : memref<1x1x5376x2048xf32, strided<[11010048, 11010048, 2048, 1], offset: ?>>, memref<1x1x3x3xf32, strided<[9, 9, 3, 1], offset: ?>>) outs(%subview_1 : memref<1x1x5374x2046xf32, strided<[10995204, 10995204, 2046, 1], offset: ?>>)
      gpu.terminator
    }
    return %alloc : memref<1x1x5374x2046xf32>
  }
}

