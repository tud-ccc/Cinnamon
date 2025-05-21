# TileFirst softmax scheduling

As of 2025-05-21, only the single-DPU scheduling runs end-to-end.
In order to compare with CINM I partially generate and manually edit some files for a softmax that runs on similar configurations as the CINM examples.

1. Do greedy fusion, no scheduling yet:
```shell
just cinm-opt source.mlir --btfl-apply-transforms > par.mlir
```
2. Copy `par.mlir` to `par1.mlir`. Modify it manually: 
- Add accelerator def
```mlir
  %upmem = tilefirst.accelerator #upmem.array<ranks(r : R = 4), dpus(d : D = 64), tasklets(t : T = 16)>
```
- Place intermediate buffers and reduction loops on host.
- Place buffers inside other schedule blocks in `wram(r, d)`
- Change the transform program to 
```mlir

```
2. Copy `par.mlir` to `par2.mlir`. Modify it manually:
- Add accelerator def
```mlir
  %upmem = tilefirst.accelerator #upmem.array<ranks(r : R = 4), dpus(d : D = 64), tasklets(t : T = 16)>
```
- Parallelize the reductions `schedule` using the following scheme:
  - Split the `red R` dimension into a `par L, red R` couple, with kdims `(1,1)` and pdims `(R*D, 1048576 | (R*D))`
  - Add another schedule with no dimensions, that reduces the remaining `(R*D)` elements.
  - Also delete the `Q` dimension.
3. Copy `par2.mlir` to `par3.mlir`. Modify it manually:
- Place the intermediate buffers and the reductions on host.
- Execute the transform program to unwrap reduction schedules:
```shell
 just cinm-opt par3.mlir --btfl-apply-transforms > par4.mlir
```
4. On `par4.mlir`, change the transform program to:
```mlir
    transform.sequence failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      %0 = transform.btfl.find_descendants "btfl.schedule" in %arg1 : (!transform.op<"btfl.block">) -> !transform.op<"btfl.schedule">
      transform.btfl.tile_down %0 dim 0 by factor symbolic<R * D> scheduler hwparallel tsym symbolic<r*d> : !transform.op<"btfl.schedule">
      transform.btfl.simplify_schedule %0
    }
```
This will create an explicit loop for the parallel `R*D` part.
Execute the program:
```shell
 just cinm-opt par4.mlir --btfl-apply-transforms > par5.mlir
```
5. In `par5.mlir`, look at the kernel `sub+exp+sum`. Our goal is to extract parallelism again from 
the reduction. We have to split it open to separate the parallel and reduction part, which were merged
too greedily. Replace the schedule around that kernel with:
```mlir
schedule<(par M) = (1) to (1048576 | (R * D))>
              ins(%buf_4 = %buf : <f32> to host)
              outs(%buf_5 = %tile : <(M), f32> to host) {
  kernel "sub+exp" ins(%arg1 = %buf_4 : <f32>) outs(%arg2 = %buf_5 : <1xf32>) {
    %0 = affine.load %arg2[0] : memref<1xf32>
    %1 = affine.load %arg1[] : memref<f32>
    %2 = arith.subf %0, %1 : f32
    %3 = math.exp %2 : f32
    affine.store %3, %arg2[0] : memref<1xf32>
  }
}
schedule<(red M) = (1) to (1048576 | (R * D))>
        ins(%buf_5 = %tile : <(M), f32> to host)
        outs(%buf_6 = %tile_3 : <1xf32> to host) {
  kernel "sum" ins(%arg2 = %buf_5 : <1xf32>) outs(%arg3 = %buf_6 : <1xf32>) {
    %0 = affine.load %arg2[0] : memref<1xf32>
    %4 = affine.load %arg3[0] : memref<1xf32>
    %5 = arith.addf %0, %4 : f32
    affine.store %5, %arg3[0] : memref<1xf32>
  }
}
```
Replace the transform program with:
```mlir
  transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      transform.btfl.schedule_block %arg1
  }
```


Now IIUC, the remaining schedules represent 1-DPU only parts of the code. 


