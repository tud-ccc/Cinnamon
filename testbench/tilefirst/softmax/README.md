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
3. Copy `par2.mlir` to `par3.mlir`. Modify it manually:
- Place the intermediate buffers and the reductions on host.


