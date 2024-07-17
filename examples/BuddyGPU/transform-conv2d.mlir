module attributes { transform.with_named_sequence } {
  transform.named_sequence @codegen(
      %arg0: !transform.any_op) {

    // Step 1: Match the conv2d operation inside the input operation.
    %conv2d = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    // Step 2: Tile the conv2d operation using forall loops.
    %tiled_op, %forall_op = transform.structured.tile_using_forall %conv2d tile_sizes [128, 256](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 3: Apply tiling canonicalization and other patterns.
    %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %1 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %1 : !transform.any_op

    // Step 4: Apply loop invariant code motion.
    %all_loops = transform.structured.match interface{LoopLikeInterface}
        in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op
    transform.apply_patterns to %1 {
    transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op

    transform.print %1 : !transform.any_op
    // Step 5: Pad the tiled operation.
    // %padded, %pad, %copy = transform.structured.pad %tiled_op {copy_back_op = "none", pack_paddings = [1, 1, 1], pad_to_multiple_of = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Step 6: Tile using for loops again.
    // %tiled_linalg_op, %loops = transform.structured.tile_using_for %padded [0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 7: Apply vectorization.
    // %14 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // transform.apply_patterns to %14 {
    //   transform.apply_patterns.linalg.tiling_canonicalization
    //   transform.apply_patterns.scf.for_loop_canonicalization
    //   transform.apply_patterns.canonicalization
    // } : !transform.any_op
    // transform.apply_cse to %14 : !transform.any_op
    // %all_loops_2 = transform.structured.match interface{LoopLikeInterface}
    //     in %14
    //     : (!transform.any_op) -> !transform.any_op
    // transform.apply_licm to %all_loops_2 : !transform.any_op
    // transform.apply_patterns to %14 {
    //   transform.apply_patterns.linalg.tiling_canonicalization
    //   transform.apply_patterns.vector.lower_masked_transfers
    // } : !transform.any_op
    // %16 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op

    // transform.apply_patterns to %16 {
    // transform.apply_patterns.linalg.tiling_canonicalization
    // transform.apply_patterns.scf.for_loop_canonicalization
    // transform.apply_patterns.canonicalization
    // } : !transform.any_op
    // transform.apply_cse to %16 : !transform.any_op
    // %all_loops_3 = transform.structured.match interface{LoopLikeInterface}
    //     in %16
    //     : (!transform.any_op) -> !transform.any_op
    // transform.apply_licm to %all_loops_3 : !transform.any_op
    // transform.apply_patterns to %16 {
    //   transform.apply_patterns.linalg.tiling_canonicalization
    //   transform.apply_patterns.vector.lower_masked_transfers
    // } : !transform.any_op

    // Step 8: Bufferization.
    %scf_forall = transform.structured.match ops{["scf.forall"]} attributes{mapping = [#gpu.block<y>, #gpu.block<x>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %alloc_tensor_ops = transform.structured.match ops{["bufferization.alloc_tensor"]} in %scf_forall : (!transform.any_op) -> !transform.any_op
    %buffer, %new_ops = transform.structured.bufferize_to_allocation
      %alloc_tensor_ops {memory_space = 3} : !transform.any_op

    // Step 9: Apply bufferization, elimination of empty tensors, and dead allocation/stores cleanup.
    transform.structured.eliminate_empty_tensors %arg0 : !transform.any_op
    %func_eras = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func_eras {
        transform.apply_patterns.linalg.erase_unnecessary_inputs
      } : !transform.any_op
    
    %17 = transform.bufferization.one_shot_bufferize  %arg0 { bufferize_function_boundaries = true, function_boundary_type_conversion = 1 : i32} : (!transform.any_op) -> !transform.any_op
    %18 = transform.structured.match ops{["func.func"]} in %17 : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %18 : (!transform.any_op) -> ()
    %19 = transform.structured.match ops{["func.func"]} in %17 : (!transform.any_op) -> !transform.any_op

    // Steps 10-12: GPU mapping, barrier elimination, and other optimizations.
    %gpu_launch = transform.gpu.map_forall_to_blocks %19 { generate_gpu_launch } : (!transform.any_op) -> !transform.any_op
    %mapped = transform.gpu.map_nested_forall_to_threads %gpu_launch block_dims = [64, 2, 1] warp_size = 32 : (!transform.any_op) -> !transform.any_op
  %20 = transform.structured.match ops{["func.func"]} in %17 : (!transform.any_op) -> !transform.any_op

  %21 = transform.buddy.eliminate_gpu_barriers %20 : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %17 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.apply_cse to %17 : !transform.any_op
  %all_loops_4 = transform.structured.match interface{LoopLikeInterface}
      in %17
      : (!transform.any_op) -> !transform.any_op
  transform.apply_licm to %all_loops_4 : !transform.any_op
  transform.apply_patterns to %17 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.vector.lower_masked_transfers
  } : !transform.any_op

  // don't use this pass as it hoists shared memory operation out.
  transform.buddy.hoist_static_alloc %21 : (!transform.any_op) -> ()

  transform.apply_patterns to %21 {
    transform.apply_patterns.memref.fold_memref_alias_ops
  } : !transform.any_op
  transform.apply_patterns to %21 {
    transform.apply_patterns.memref.extract_address_computations
  } : !transform.any_op
  transform.apply_patterns to %21 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.apply_cse to %21 : !transform.any_op
  %all_loops_5 = transform.structured.match interface{LoopLikeInterface}
      in %21
      : (!transform.any_op) -> !transform.any_op
  transform.apply_licm to %all_loops_5 : !transform.any_op
  transform.apply_patterns to %21 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.vector.lower_masked_transfers
  } : !transform.any_op

  transform.apply_patterns to %21 {
    transform.apply_patterns.buddy.unroll_vectors_gpu_mma_sync
  } : !transform.any_op

  %22 = transform.structured.match ops{["scf.for"]} in %21 : (!transform.any_op) -> !transform.op<"scf.for">
  transform.buddy.synchronize_loop %22 : (!transform.op<"scf.for">) -> ()

  transform.apply_patterns to %21 {
              transform.apply_patterns.memref.fold_memref_alias_ops
            } : !transform.any_op
  transform.apply_cse to %21 : !transform.any_op
  %23 = transform.structured.hoist_redundant_vector_transfers %21 : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %23 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.apply_cse to %23 : !transform.any_op
  %all_loops_6 = transform.structured.match interface{LoopLikeInterface}
      in %23
      : (!transform.any_op) -> !transform.any_op
  transform.apply_licm to %all_loops_6 : !transform.any_op
  transform.apply_patterns to %23 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.vector.lower_masked_transfers
  } : !transform.any_op

  //transform.memref.erase_dead_alloc_and_stores %23 : (!transform.any_op) -> ()
  transform.buddy.vector.vector_to_mma_conversion %23 {use_mma_sync} : (!transform.any_op) -> ()
  %24 = transform.buddy.eliminate_gpu_barriers %23 : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %24 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.apply_cse to %24 : !transform.any_op
  %all_loops_7 = transform.structured.match interface{LoopLikeInterface}
      in %24
      : (!transform.any_op) -> !transform.any_op
  transform.apply_licm to %all_loops_7 : !transform.any_op
  transform.apply_patterns to %24 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.vector.lower_masked_transfers
  } : !transform.any_op

  %25 = transform.structured.match ops{["gpu.launch"]} in %24 : (!transform.any_op) -> !transform.any_op

  %fwfa = transform.structured.match ops{["memref.alloc"]} in %25 : (!transform.any_op) -> !transform.op<"memref.alloc">

  // transform.print %fwfa : !transform.op<"memref.alloc">

  // This needs to be enabled with hoisting.
  transform.memref.multibuffer %fwfa {factor = 3 : i64, skip_analysis} : (!transform.op<"memref.alloc">) -> !transform.any_op

  transform.apply_patterns to %24 {
    transform.apply_patterns.vector.transfer_to_scf full_unroll = true
  } : !transform.any_op
  transform.apply_patterns to %24 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.apply_cse to %24 : !transform.any_op
  %all_loops_8 = transform.structured.match interface{LoopLikeInterface}
      in %24
      : (!transform.any_op) -> !transform.any_op
  transform.apply_licm to %all_loops_8 : !transform.any_op
  transform.apply_patterns to %24 {
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.vector.lower_masked_transfers
  } : !transform.any_op

  // Don't use nvgpu here as it relies memory space identifiers.
  transform.buddy.create_async_groups %24 {use_mma_sync} : (!transform.any_op) -> ()
  transform.apply_patterns to %24 {
    transform.apply_patterns.linalg.tiling_canonicalization
    //transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
    transform.apply_patterns.memref.fold_memref_alias_ops
  } : !transform.any_op
  %all_loops_9 = transform.structured.match interface{LoopLikeInterface}
      in %24
      : (!transform.any_op) -> !transform.any_op
  transform.apply_licm to %all_loops_9 : !transform.any_op
  transform.apply_cse to %24 : !transform.any_op

  %26 = transform.structured.match ops{["nvgpu.mma.sync"]} in %24 : (!transform.any_op) -> !transform.any_op
  %27 = transform.get_parent_op %26 {deduplicate, op_name = "scf.for"} : (!transform.any_op) -> !transform.any_op
  // %28 = transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %27 {depth = 3 : i64, use_mma_sync, peel_epilogue} : (!transform.any_op) -> !transform.any_op
  %28 = transform.buddy.pipeline_shared_memory_copies %27 {depth = 3 : i64, use_mma_sync, peel_epilogue} : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %24 {
    transform.apply_patterns.vector.lower_masks
  } : !transform.any_op
  transform.apply_patterns to %24 {
    transform.apply_patterns.vector.materialize_masks
  } : !transform.any_op
  transform.apply_patterns to %24 {
    transform.apply_patterns.linalg.tiling_canonicalization
    //transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
    transform.apply_patterns.memref.fold_memref_alias_ops
  } : !transform.any_op

  %all_loops_10 = transform.structured.match interface{LoopLikeInterface}
      in %24
      : (!transform.any_op) -> !transform.any_op
  transform.apply_licm to %all_loops_10 : !transform.any_op
  transform.apply_cse to %24 : !transform.any_op

  transform.yield
}
} // module