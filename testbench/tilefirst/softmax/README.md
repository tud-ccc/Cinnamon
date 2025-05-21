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
