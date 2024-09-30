; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__tconstant_1xi64 = private constant [1 x i64] [i64 1024], align 64
@__tconstant_256x4xi32 = private constant [256 x [4 x i32]] zeroinitializer, align 64
@__tconstant_2xi64 = private constant [2 x i64] [i64 256, i64 4], align 64
@__tconstant_1024xi32 = private constant [1024 x i32] zeroinitializer, align 64
@__constant_1xi64_7 = private constant [1 x i64] [i64 32768]
@__constant_256x1xf32 = private constant [256 x [1 x float]] zeroinitializer
@__constant_2xi64_8 = private constant [2 x i64] [i64 768, i64 1]
@__constant_1xi64_6 = private constant [1 x i64] [i64 768]
@__constant_48x6xf32 = private constant [48 x [6 x float]] zeroinitializer
@__constant_2xi64_7 = private constant [2 x i64] [i64 48, i64 6]
@__constant_48x1xf32 = private constant [48 x [1 x float]] zeroinitializer
@__constant_2xi64_6 = private constant [2 x i64] [i64 288, i64 1]
@__constant_1xi64_5 = private constant [1 x i64] [i64 1024]
@__constant_256x4xf32 = private constant [256 x [4 x float]] zeroinitializer
@__constant_2xi64_5 = private constant [2 x i64] [i64 256, i64 4]
@__constant_1024xf32 = private constant [1024 x float] zeroinitializer
@__constant_1xi64_4 = private constant [1 x i64] [i64 4096]
@__constant_256x16xf32 = private constant [256 x [16 x float]] zeroinitializer
@__constant_2xi64_4 = private constant [2 x i64] [i64 256, i64 16]
@__constant_4096xf32 = private constant [4096 x float] zeroinitializer
@__constant_1xi64_3 = private constant [1 x i64] [i64 48]
@__constant_8x6xf32 = private constant [8 x [6 x float]] zeroinitializer
@__constant_2xi64_3 = private constant [2 x i64] [i64 8, i64 6]
@__constant_48xf32 = private constant [48 x float] zeroinitializer
@__constant_1xi64_2 = private constant [1 x i64] [i64 256]
@__constant_128x2xf32 = private constant [128 x [2 x float]] zeroinitializer
@__constant_2xi64_2 = private constant [2 x i64] [i64 128, i64 2]
@__constant_256xf32 = private constant [256 x float] zeroinitializer
@__constant_1xi64_1 = private constant [1 x i64] [i64 288]
@__constant_16x18xf32 = private constant [16 x [18 x float]] zeroinitializer
@__constant_2xi64_1 = private constant [2 x i64] [i64 16, i64 18]
@__constant_288xf32 = private constant [288 x float] zeroinitializer
@__constant_xf32_0 = private constant float 0.000000e+00
@__constant_1xi64_0 = private constant [1 x i64] [i64 262144]
@__constant_4096x64xf32 = private constant [4096 x [64 x float]] zeroinitializer
@__constant_4096x2xf32 = private constant [4096 x [2 x float]] zeroinitializer
@__constant_2xi64_0 = private constant [2 x i64] [i64 4096, i64 64]
@__constant_262144xf32 = private constant [262144 x float] zeroinitializer
@__constant_xf32 = private constant float 0xFFF0000000000000
@__constant_1xi64 = private constant [1 x i64] [i64 1048576]
@__constant_4096x256xf32 = private constant [4096 x [256 x float]] zeroinitializer
@__constant_2xi64 = private constant [2 x i64] [i64 4096, i64 256]
@__constant_1048576xf32 = private constant [1048576 x float] zeroinitializer
@dpu_program = private constant [8 x i8] c"forward\00"
@dpu_program_0 = private constant [8 x i8] c"forward\00"
@dpu_program_1 = private constant [8 x i8] c"forward\00"
@dpu_program_2 = private constant [8 x i8] c"forward\00"
@dpu_program_3 = private constant [10 x i8] c"forward_3\00"
@dpu_program_4 = private constant [8 x i8] c"forward\00"
@dpu_program_5 = private constant [8 x i8] c"forward\00"
@dpu_program_6 = private constant [10 x i8] c"forward_6\00"
@dpu_program_7 = private constant [10 x i8] c"forward_3\00"
@dpu_program_8 = private constant [10 x i8] c"forward_8\00"
@dpu_program_9 = private constant [4 x i8] c"mha\00"
@dpu_program_10 = private constant [6 x i8] c"mha_9\00"
@dpu_program_11 = private constant [10 x i8] c"forward_3\00"
@dpu_program_12 = private constant [8 x i8] c"rmsnorm\00"
@dpu_program_13 = private constant [11 x i8] c"rmsnorm_11\00"
@dpu_program_14 = private constant [8 x i8] c"rmsnorm\00"
@dpu_program_15 = private constant [8 x i8] c"softmax\00"
@dpu_program_16 = private constant [11 x i8] c"softmax_13\00"
@dpu_program_17 = private constant [16 x i8] c"rmsnorm_1048576\00"
@dpu_program_18 = private constant [19 x i8] c"rmsnorm_1048576_14\00"
@dpu_program_19 = private constant [16 x i8] c"rmsnorm_1048576\00"
@dpu_program_20 = private constant [16 x i8] c"softmax_1048576\00"
@dpu_program_21 = private constant [19 x i8] c"softmax_1048576_16\00"
@dpu_program_22 = private constant [11 x i8] c"va_1048576\00"
@dpu_program_23 = private constant [15 x i8] c"rmsnorm_262144\00"
@dpu_program_24 = private constant [18 x i8] c"rmsnorm_262144_17\00"
@dpu_program_25 = private constant [15 x i8] c"rmsnorm_262144\00"
@dpu_program_26 = private constant [15 x i8] c"softmax_262144\00"
@dpu_program_27 = private constant [18 x i8] c"softmax_262144_19\00"
@dpu_program_28 = private constant [19 x i8] c"rmsnorm_262144_opt\00"
@dpu_program_29 = private constant [22 x i8] c"rmsnorm_262144_opt_20\00"
@dpu_program_30 = private constant [19 x i8] c"softmax_262144_opt\00"
@dpu_program_31 = private constant [22 x i8] c"softmax_262144_opt_21\00"
@dpu_program_32 = private constant [10 x i8] c"va_262144\00"
@dpu_program_33 = private constant [8 x i8] c"mha_big\00"
@dpu_program_34 = private constant [11 x i8] c"mha_big_22\00"
@dpu_program_35 = private constant [11 x i8] c"mha_big_23\00"
@dpu_program_36 = private constant [11 x i8] c"mha_big_24\00"
@dpu_program_37 = private constant [11 x i8] c"mha_big_25\00"
@dpu_program_38 = private constant [7 x i8] c"test_0\00"
@dpu_program_39 = private constant [7 x i8] c"test_0\00"
@dpu_program_40 = private constant [7 x i8] c"test_0\00"
@dpu_program_41 = private constant [7 x i8] c"test_0\00"
@dpu_program_42 = private constant [7 x i8] c"test_0\00"
@dpu_program_43 = private constant [7 x i8] c"test_0\00"
@dpu_program_44 = private constant [7 x i8] c"test_2\00"

declare ptr @malloc(i64)

define private i64 @scatter_map_13(i64 %0) {
  %2 = mul i64 %0, 16
  %3 = srem i64 %0, 256
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 256
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 256
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -256
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_12(i64 %0) {
  %2 = mul i64 %0, 64
  %3 = srem i64 %0, 256
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 256
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 1024
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -1024
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_11(i64 %0) {
  %2 = mul i64 %0, 8
  %3 = icmp slt i64 %0, 0
  %4 = sub i64 -1, %0
  %5 = select i1 %3, i64 %4, i64 %0
  %6 = sdiv i64 %5, 1024
  %7 = sub i64 -1, %6
  %8 = select i1 %3, i64 %7, i64 %6
  %9 = mul i64 %8, 8192
  %10 = add i64 %2, %9
  %11 = srem i64 %0, 1024
  %12 = icmp slt i64 %11, 0
  %13 = add i64 %11, 1024
  %14 = select i1 %12, i64 %13, i64 %11
  %15 = icmp slt i64 %14, 0
  %16 = sub i64 -1, %14
  %17 = select i1 %15, i64 %16, i64 %14
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, 128
  %22 = add i64 %10, %21
  %23 = sdiv i64 %5, 16
  %24 = sub i64 -1, %23
  %25 = select i1 %3, i64 %24, i64 %23
  %26 = mul i64 %25, -128
  %27 = add i64 %22, %26
  ret i64 %27
}

define private i64 @scatter_map_10(i64 %0) {
  %2 = mul i64 %0, 256
  %3 = icmp slt i64 %0, 0
  %4 = sub i64 -1, %0
  %5 = select i1 %3, i64 %4, i64 %0
  %6 = sdiv i64 %5, 1024
  %7 = sub i64 -1, %6
  %8 = select i1 %3, i64 %7, i64 %6
  %9 = mul i64 %8, 262144
  %10 = add i64 %2, %9
  %11 = srem i64 %0, 1024
  %12 = icmp slt i64 %11, 0
  %13 = add i64 %11, 1024
  %14 = select i1 %12, i64 %13, i64 %11
  %15 = icmp slt i64 %14, 0
  %16 = sub i64 -1, %14
  %17 = select i1 %15, i64 %16, i64 %14
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, 4096
  %22 = add i64 %10, %21
  %23 = sdiv i64 %5, 16
  %24 = sub i64 -1, %23
  %25 = select i1 %3, i64 %24, i64 %23
  %26 = mul i64 %25, -4096
  %27 = add i64 %22, %26
  ret i64 %27
}

define private i64 @scatter_map_9(i64 %0) {
  %2 = mul i64 %0, 1024
  %3 = icmp slt i64 %0, 0
  %4 = sub i64 -1, %0
  %5 = select i1 %3, i64 %4, i64 %0
  %6 = sdiv i64 %5, 1024
  %7 = sub i64 -1, %6
  %8 = select i1 %3, i64 %7, i64 %6
  %9 = mul i64 %8, 1048576
  %10 = add i64 %2, %9
  %11 = srem i64 %0, 1024
  %12 = icmp slt i64 %11, 0
  %13 = add i64 %11, 1024
  %14 = select i1 %12, i64 %13, i64 %11
  %15 = icmp slt i64 %14, 0
  %16 = sub i64 -1, %14
  %17 = select i1 %15, i64 %16, i64 %14
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, 16384
  %22 = add i64 %10, %21
  %23 = sdiv i64 %5, 16
  %24 = sub i64 -1, %23
  %25 = select i1 %3, i64 %24, i64 %23
  %26 = mul i64 %25, -16384
  %27 = add i64 %22, %26
  ret i64 %27
}

define private i64 @scatter_map_8(i64 %0) {
  %2 = mul i64 %0, 8
  %3 = srem i64 %0, 128
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 128
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 128
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -128
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_7(i64 %0) {
  %2 = mul i64 %0, 72
  %3 = icmp slt i64 %0, 0
  %4 = sub i64 -1, %0
  %5 = select i1 %3, i64 %4, i64 %0
  %6 = sdiv i64 %5, 16
  %7 = sub i64 -1, %6
  %8 = select i1 %3, i64 %7, i64 %6
  %9 = mul i64 %8, -1152
  %10 = add i64 %2, %9
  ret i64 %10
}

define private i64 @scatter_map_6(i64 %0) {
  %2 = mul i64 %0, 24
  %3 = icmp slt i64 %0, 0
  %4 = sub i64 -1, %0
  %5 = select i1 %3, i64 %4, i64 %0
  %6 = sdiv i64 %5, 8
  %7 = sub i64 -1, %6
  %8 = select i1 %3, i64 %7, i64 %6
  %9 = mul i64 %8, -192
  %10 = add i64 %2, %9
  ret i64 %10
}

define private i64 @scatter_map_5(i64 %0) {
  %2 = mul i64 %0, 4
  %3 = srem i64 %0, 128
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 128
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 64
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -64
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_4(i64 %0) {
  %2 = mul i64 %0, 1152
  %3 = srem i64 %0, 128
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 128
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 18432
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -18432
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_3(i64 %0) {
  %2 = mul i64 %0, 3072
  %3 = srem i64 %0, 48
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 48
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 8
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 24576
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 8
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -24576
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_2(i64 %0) {
  %2 = mul i64 %0, 24
  %3 = srem i64 %0, 48
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 48
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 8
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 192
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 8
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -192
  %22 = add i64 %14, %21
  ret i64 %22
}

declare void @upmemrt_dpu_free(ptr)

declare void @upmemrt_dpu_gather(ptr, ptr, i64, i64, i64, i64, i64, ptr)

declare void @upmemrt_dpu_launch(ptr)

declare void @upmemrt_dpu_load(ptr, ptr)

define private i64 @scatter_map_1(i64 %0) {
  %2 = mul i64 %0, 4
  %3 = srem i64 %0, 48
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 48
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 8
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 32
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 8
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -32
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_0(i64 %0) {
  ret i64 0
}

declare void @upmemrt_dpu_scatter(ptr, ptr, i64, i64, i64, i64, i64, ptr)

define private i64 @scatter_map(i64 %0) {
  %2 = mul i64 %0, 1152
  %3 = srem i64 %0, 48
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 48
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 8
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 9216
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 8
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -9216
  %22 = add i64 %14, %21
  ret i64 %22
}

declare ptr @upmemrt_dpu_alloc(i32, i32)

define ptr @forward(i64 %0, i64 %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15) {
  %17 = mul i64 %0, 288
  %18 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %18, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, ptr %18, 1
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 0, 2
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 288, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 1, 4, 0
  %24 = getelementptr float, ptr %4, i64 %17
  call void @llvm.memcpy.p0.p0.i64(ptr %18, ptr %24, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %25 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %26 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %27 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %28 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %28, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %28, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 0, 2
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 288, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 1, 3, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 1, 4, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 1, 4, 1
  %36 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %37 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %38 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %39 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %39, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, ptr %39, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 0, 2
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 288, 3, 0
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 1, 3, 1
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 1, 4, 0
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 1, 4, 1
  %47 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %48 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %49 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %50 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %50, 0
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, ptr %50, 1
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 0, 2
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 288, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 1, 3, 1
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 1, 4, 0
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 1, 4, 1
  %58 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %59 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %60 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %61 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %60, 0
  %62 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, ptr %60, 1
  %63 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, i64 0, 2
  %64 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %63, i64 288, 3, 0
  %65 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %64, i64 1, 4, 0
  %66 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %67 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %66, 0
  %68 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, ptr %66, 1
  %69 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %68, i64 0, 2
  %70 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %69, i64 288, 3, 0
  %71 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %70, i64 1, 4, 0
  %72 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %73 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 73728) to i64))
  %74 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 73728) to i64))
  %75 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %76 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %77 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %76, 0
  %78 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, ptr %76, 1
  %79 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %78, i64 0, 2
  %80 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, i64 288, 3, 0
  %81 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %80, i64 1, 3, 1
  %82 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %81, i64 1, 4, 0
  %83 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, i64 1, 4, 1
  %84 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %85 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %86 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %87 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %88 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %89 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %90 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %90, 0
  %92 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %91, ptr %90, 1
  %93 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, i64 0, 2
  %94 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %93, i64 768, 3, 0
  %95 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %94, i64 1, 3, 1
  %96 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, i64 1, 4, 0
  %97 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, i64 1, 4, 1
  %98 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %99 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %100 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %101 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %101, 0
  %103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %102, ptr %101, 1
  %104 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, i64 0, 2
  %105 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %104, i64 768, 3, 0
  %106 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %105, i64 1, 3, 1
  %107 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %106, i64 1, 4, 0
  %108 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %107, i64 1, 4, 1
  %109 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %110 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %111 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %112 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %111, 0
  %113 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %112, ptr %111, 1
  %114 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %113, i64 0, 2
  %115 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %114, i64 768, 3, 0
  %116 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %115, i64 1, 4, 0
  %117 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %118 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %118, 0
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, ptr %118, 1
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 0, 2
  %122 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, i64 288, 3, 0
  %123 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %122, i64 1, 3, 1
  %124 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %123, i64 1, 4, 0
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %124, i64 1, 4, 1
  %126 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %127 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  br label %128

128:                                              ; preds = %528, %16
  %129 = phi i64 [ %537, %528 ], [ 0, %16 ]
  %130 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %536, %528 ], [ %23, %16 ]
  %131 = icmp slt i64 %129, 6
  br i1 %131, label %132, label %538

132:                                              ; preds = %128
  %133 = mul i64 %129, 288
  %134 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, 3, 0
  %135 = mul i64 %134, 1
  %136 = mul i64 %135, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %137 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, 1
  %138 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, 2
  %139 = getelementptr float, ptr %137, i64 %138
  call void @llvm.memcpy.p0.p0.i64(ptr %25, ptr %139, i64 %136, i1 false)
  %140 = getelementptr float, ptr %5, i64 %133
  call void @llvm.memcpy.p0.p0.i64(ptr %26, ptr %140, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %141 = call ptr @rmsnorm(ptr %25, ptr %26)
  call void @llvm.memcpy.p0.p0.i64(ptr %28, ptr %27, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %142

142:                                              ; preds = %161, %132
  %143 = phi i64 [ %180, %161 ], [ 0, %132 ]
  %144 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %170, %161 ], [ %35, %132 ]
  %145 = icmp slt i64 %143, 288
  br i1 %145, label %146, label %181

146:                                              ; preds = %142
  %147 = mul i64 %129, 82944
  %148 = mul i64 %143, 288
  %149 = add i64 %147, %148
  %150 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  br label %151

151:                                              ; preds = %154, %146
  %152 = phi i64 [ %160, %154 ], [ 0, %146 ]
  %153 = icmp slt i64 %152, 288
  br i1 %153, label %154, label %161

154:                                              ; preds = %151
  %155 = add i64 %152, 0
  %156 = getelementptr float, ptr %141, i64 %155
  %157 = load float, ptr %156, align 4
  %158 = add i64 0, %152
  %159 = getelementptr float, ptr %36, i64 %158
  store float %157, ptr %159, align 4
  %160 = add i64 %152, 1
  br label %151

161:                                              ; preds = %151
  call void @upmemrt_dpu_load(ptr %150, ptr @dpu_program)
  %162 = getelementptr inbounds float, ptr %6, i64 %149
  call void @upmemrt_dpu_scatter(ptr %150, ptr %162, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %150, ptr %36, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %150, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %150)
  call void @upmemrt_dpu_gather(ptr %150, ptr %37, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %150)
  %163 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %163, 0
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %164, ptr %163, 1
  %166 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %165, i64 0, 2
  %167 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %166, i64 288, 3, 0
  %168 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %167, i64 1, 3, 1
  %169 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %168, i64 1, 4, 0
  %170 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %169, i64 1, 4, 1
  %171 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, 3, 0
  %172 = mul i64 %171, 1
  %173 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, 3, 1
  %174 = mul i64 %172, %173
  %175 = mul i64 %174, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %176 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, 1
  %177 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, 2
  %178 = getelementptr float, ptr %176, i64 %177
  call void @llvm.memcpy.p0.p0.i64(ptr %163, ptr %178, i64 %175, i1 false)
  %179 = getelementptr float, ptr %163, i64 %143
  call void @llvm.memcpy.p0.p0.i64(ptr %179, ptr %37, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %180 = add i64 %143, 48
  br label %142

181:                                              ; preds = %142
  %182 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %39, ptr %38, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %183

183:                                              ; preds = %202, %181
  %184 = phi i64 [ %221, %202 ], [ 0, %181 ]
  %185 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %211, %202 ], [ %46, %181 ]
  %186 = icmp slt i64 %184, 288
  br i1 %186, label %187, label %222

187:                                              ; preds = %183
  %188 = mul i64 %129, 82944
  %189 = mul i64 %184, 288
  %190 = add i64 %188, %189
  %191 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  br label %192

192:                                              ; preds = %195, %187
  %193 = phi i64 [ %201, %195 ], [ 0, %187 ]
  %194 = icmp slt i64 %193, 288
  br i1 %194, label %195, label %202

195:                                              ; preds = %192
  %196 = add i64 %193, 0
  %197 = getelementptr float, ptr %141, i64 %196
  %198 = load float, ptr %197, align 4
  %199 = add i64 0, %193
  %200 = getelementptr float, ptr %47, i64 %199
  store float %198, ptr %200, align 4
  %201 = add i64 %193, 1
  br label %192

202:                                              ; preds = %192
  call void @upmemrt_dpu_load(ptr %191, ptr @dpu_program_0)
  %203 = getelementptr inbounds float, ptr %7, i64 %190
  call void @upmemrt_dpu_scatter(ptr %191, ptr %203, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %191, ptr %47, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %191, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %191)
  call void @upmemrt_dpu_gather(ptr %191, ptr %48, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %191)
  %204 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %205 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %204, 0
  %206 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %205, ptr %204, 1
  %207 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %206, i64 0, 2
  %208 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %207, i64 288, 3, 0
  %209 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %208, i64 1, 3, 1
  %210 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %209, i64 1, 4, 0
  %211 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %210, i64 1, 4, 1
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, 3, 0
  %213 = mul i64 %212, 1
  %214 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, 3, 1
  %215 = mul i64 %213, %214
  %216 = mul i64 %215, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %217 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, 1
  %218 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, 2
  %219 = getelementptr float, ptr %217, i64 %218
  call void @llvm.memcpy.p0.p0.i64(ptr %204, ptr %219, i64 %216, i1 false)
  %220 = getelementptr float, ptr %204, i64 %184
  call void @llvm.memcpy.p0.p0.i64(ptr %220, ptr %48, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %221 = add i64 %184, 48
  br label %183

222:                                              ; preds = %183
  %223 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %50, ptr %49, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %224

224:                                              ; preds = %243, %222
  %225 = phi i64 [ %262, %243 ], [ 0, %222 ]
  %226 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %252, %243 ], [ %57, %222 ]
  %227 = icmp slt i64 %225, 288
  br i1 %227, label %228, label %263

228:                                              ; preds = %224
  %229 = mul i64 %129, 82944
  %230 = mul i64 %225, 288
  %231 = add i64 %229, %230
  %232 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  br label %233

233:                                              ; preds = %236, %228
  %234 = phi i64 [ %242, %236 ], [ 0, %228 ]
  %235 = icmp slt i64 %234, 288
  br i1 %235, label %236, label %243

236:                                              ; preds = %233
  %237 = add i64 %234, 0
  %238 = getelementptr float, ptr %141, i64 %237
  %239 = load float, ptr %238, align 4
  %240 = add i64 0, %234
  %241 = getelementptr float, ptr %58, i64 %240
  store float %239, ptr %241, align 4
  %242 = add i64 %234, 1
  br label %233

243:                                              ; preds = %233
  call void @upmemrt_dpu_load(ptr %232, ptr @dpu_program_1)
  %244 = getelementptr inbounds float, ptr %8, i64 %231
  call void @upmemrt_dpu_scatter(ptr %232, ptr %244, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %232, ptr %58, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %232, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %232)
  call void @upmemrt_dpu_gather(ptr %232, ptr %59, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %232)
  %245 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %246 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %245, 0
  %247 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %246, ptr %245, 1
  %248 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %247, i64 0, 2
  %249 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %248, i64 288, 3, 0
  %250 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %249, i64 1, 3, 1
  %251 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %250, i64 1, 4, 0
  %252 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %251, i64 1, 4, 1
  %253 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %226, 3, 0
  %254 = mul i64 %253, 1
  %255 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %226, 3, 1
  %256 = mul i64 %254, %255
  %257 = mul i64 %256, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %258 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %226, 1
  %259 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %226, 2
  %260 = getelementptr float, ptr %258, i64 %259
  call void @llvm.memcpy.p0.p0.i64(ptr %245, ptr %260, i64 %257, i1 false)
  %261 = getelementptr float, ptr %245, i64 %225
  call void @llvm.memcpy.p0.p0.i64(ptr %261, ptr %59, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %262 = add i64 %225, 48
  br label %224

263:                                              ; preds = %224
  %264 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %226, 1
  %265 = uitofp i64 %1 to float
  call void @llvm.memcpy.p0.p0.i64(ptr %60, ptr %182, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %66, ptr %223, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %266

266:                                              ; preds = %311, %263
  %267 = phi i64 [ %312, %311 ], [ 0, %263 ]
  %268 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %292, %311 ], [ %65, %263 ]
  %269 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %310, %311 ], [ %71, %263 ]
  %270 = icmp slt i64 %267, 288
  br i1 %270, label %271, label %313

271:                                              ; preds = %266
  %272 = urem i64 %267, 48
  %273 = uitofp i64 %272 to float
  %274 = fdiv float %273, 4.800000e+01
  %275 = call float @llvm.pow.f32(float 1.000000e+04, float %274)
  %276 = fdiv float 1.000000e+00, %275
  %277 = fmul float %265, %276
  %278 = call float @llvm.cos.f32(float %277)
  %279 = call float @llvm.sin.f32(float %277)
  %280 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %281 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %268, 3, 0
  %282 = mul i64 %281, 1
  %283 = mul i64 %282, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %284 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %268, 1
  %285 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %268, 2
  %286 = getelementptr float, ptr %284, i64 %285
  call void @llvm.memcpy.p0.p0.i64(ptr %280, ptr %286, i64 %283, i1 false)
  %287 = call ptr @rot(ptr %280, i64 %267, float %278, float %279)
  %288 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %287, 0
  %289 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %288, ptr %287, 1
  %290 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %289, i64 0, 2
  %291 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %290, i64 288, 3, 0
  %292 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %291, i64 1, 4, 0
  %293 = icmp ult i64 %267, 288
  %294 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  br i1 %293, label %295, label %308

295:                                              ; preds = %271
  %296 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %269, 3, 0
  %297 = mul i64 %296, 1
  %298 = mul i64 %297, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %299 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %269, 1
  %300 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %269, 2
  %301 = getelementptr float, ptr %299, i64 %300
  call void @llvm.memcpy.p0.p0.i64(ptr %294, ptr %301, i64 %298, i1 false)
  %302 = call ptr @rot(ptr %294, i64 %267, float %278, float %279)
  %303 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %302, 0
  %304 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %303, ptr %302, 1
  %305 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %304, i64 0, 2
  %306 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %305, i64 288, 3, 0
  %307 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %306, i64 1, 4, 0
  br label %309

308:                                              ; preds = %271
  br label %309

309:                                              ; preds = %295, %308
  %310 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %269, %308 ], [ %307, %295 ]
  br label %311

311:                                              ; preds = %309
  %312 = add i64 %267, 2
  br label %266

313:                                              ; preds = %266
  %314 = mul i64 %129, 73728
  %315 = mul i64 %1, 288
  %316 = add i64 %314, %315
  %317 = getelementptr float, ptr %3, i64 %316
  call void @llvm.memcpy.p0.p0.i64(ptr %317, ptr %264, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %318 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %269, 3, 0
  %319 = mul i64 %318, 1
  %320 = mul i64 %319, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %321 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %269, 1
  %322 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %269, 2
  %323 = getelementptr float, ptr %321, i64 %322
  %324 = getelementptr float, ptr %2, i64 %316
  call void @llvm.memcpy.p0.p0.i64(ptr %324, ptr %323, i64 %320, i1 false)
  %325 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %268, 3, 0
  %326 = mul i64 %325, 1
  %327 = mul i64 %326, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %328 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %268, 1
  %329 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %268, 2
  %330 = getelementptr float, ptr %328, i64 %329
  call void @llvm.memcpy.p0.p0.i64(ptr %72, ptr %330, i64 %327, i1 false)
  %331 = getelementptr float, ptr %2, i64 %314
  call void @llvm.memcpy.p0.p0.i64(ptr %73, ptr %331, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 73728), i1 false)
  %332 = getelementptr float, ptr %3, i64 %314
  call void @llvm.memcpy.p0.p0.i64(ptr %74, ptr %332, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 73728), i1 false)
  %333 = call ptr @mha(ptr %72, ptr %73, ptr %74, i64 %1)
  call void @llvm.memcpy.p0.p0.i64(ptr %76, ptr %75, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %334

334:                                              ; preds = %353, %313
  %335 = phi i64 [ %372, %353 ], [ 0, %313 ]
  %336 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %362, %353 ], [ %83, %313 ]
  %337 = icmp slt i64 %335, 288
  br i1 %337, label %338, label %373

338:                                              ; preds = %334
  %339 = mul i64 %129, 82944
  %340 = mul i64 %335, 288
  %341 = add i64 %339, %340
  %342 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  br label %343

343:                                              ; preds = %346, %338
  %344 = phi i64 [ %352, %346 ], [ 0, %338 ]
  %345 = icmp slt i64 %344, 288
  br i1 %345, label %346, label %353

346:                                              ; preds = %343
  %347 = add i64 %344, 0
  %348 = getelementptr float, ptr %333, i64 %347
  %349 = load float, ptr %348, align 4
  %350 = add i64 0, %344
  %351 = getelementptr float, ptr %84, i64 %350
  store float %349, ptr %351, align 4
  %352 = add i64 %344, 1
  br label %343

353:                                              ; preds = %343
  call void @upmemrt_dpu_load(ptr %342, ptr @dpu_program_2)
  %354 = getelementptr inbounds float, ptr %9, i64 %341
  call void @upmemrt_dpu_scatter(ptr %342, ptr %354, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %342, ptr %84, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %342, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %342)
  call void @upmemrt_dpu_gather(ptr %342, ptr %85, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %342)
  %355 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %356 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %355, 0
  %357 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %356, ptr %355, 1
  %358 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %357, i64 0, 2
  %359 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %358, i64 288, 3, 0
  %360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %359, i64 1, 3, 1
  %361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %360, i64 1, 4, 0
  %362 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, i64 1, 4, 1
  %363 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 3, 0
  %364 = mul i64 %363, 1
  %365 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 3, 1
  %366 = mul i64 %364, %365
  %367 = mul i64 %366, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %368 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 1
  %369 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 2
  %370 = getelementptr float, ptr %368, i64 %369
  call void @llvm.memcpy.p0.p0.i64(ptr %355, ptr %370, i64 %367, i1 false)
  %371 = getelementptr float, ptr %355, i64 %335
  call void @llvm.memcpy.p0.p0.i64(ptr %371, ptr %85, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %372 = add i64 %335, 48
  br label %334

373:                                              ; preds = %334
  %374 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 1
  %375 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  call void @upmemrt_dpu_load(ptr %375, ptr @dpu_program_3)
  call void @upmemrt_dpu_scatter(ptr %375, ptr %137, i64 4, i64 288, i64 6, i64 192, i64 0, ptr @scatter_map_2)
  call void @upmemrt_dpu_scatter(ptr %375, ptr %374, i64 4, i64 288, i64 6, i64 192, i64 192, ptr @scatter_map_2)
  call void @upmemrt_dpu_scatter(ptr %375, ptr @__constant_48x6xf32, i64 4, i64 288, i64 6, i64 192, i64 384, ptr @scatter_map_2)
  call void @upmemrt_dpu_launch(ptr %375)
  call void @upmemrt_dpu_gather(ptr %375, ptr %86, i64 4, i64 288, i64 6, i64 192, i64 384, ptr @scatter_map_2)
  call void @upmemrt_dpu_free(ptr %375)
  call void @llvm.memcpy.p0.p0.i64(ptr %87, ptr %86, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %376 = getelementptr float, ptr %13, i64 %133
  call void @llvm.memcpy.p0.p0.i64(ptr %88, ptr %376, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %377 = call ptr @rmsnorm(ptr %87, ptr %88)
  call void @llvm.memcpy.p0.p0.i64(ptr %90, ptr %89, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %378

378:                                              ; preds = %397, %373
  %379 = phi i64 [ %416, %397 ], [ 0, %373 ]
  %380 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %406, %397 ], [ %97, %373 ]
  %381 = icmp slt i64 %379, 768
  br i1 %381, label %382, label %417

382:                                              ; preds = %378
  %383 = mul i64 %129, 221184
  %384 = mul i64 %379, 288
  %385 = add i64 %383, %384
  %386 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  br label %387

387:                                              ; preds = %390, %382
  %388 = phi i64 [ %396, %390 ], [ 0, %382 ]
  %389 = icmp slt i64 %388, 288
  br i1 %389, label %390, label %397

390:                                              ; preds = %387
  %391 = add i64 %388, 0
  %392 = getelementptr float, ptr %377, i64 %391
  %393 = load float, ptr %392, align 4
  %394 = add i64 0, %388
  %395 = getelementptr float, ptr %98, i64 %394
  store float %393, ptr %395, align 4
  %396 = add i64 %388, 1
  br label %387

397:                                              ; preds = %387
  call void @upmemrt_dpu_load(ptr %386, ptr @dpu_program_4)
  %398 = getelementptr inbounds float, ptr %10, i64 %385
  call void @upmemrt_dpu_scatter(ptr %386, ptr %398, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %386, ptr %98, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %386, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %386)
  call void @upmemrt_dpu_gather(ptr %386, ptr %99, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %386)
  %399 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %400 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %399, 0
  %401 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %400, ptr %399, 1
  %402 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %401, i64 0, 2
  %403 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %402, i64 768, 3, 0
  %404 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %403, i64 1, 3, 1
  %405 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %404, i64 1, 4, 0
  %406 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %405, i64 1, 4, 1
  %407 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %380, 3, 0
  %408 = mul i64 %407, 1
  %409 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %380, 3, 1
  %410 = mul i64 %408, %409
  %411 = mul i64 %410, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %412 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %380, 1
  %413 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %380, 2
  %414 = getelementptr float, ptr %412, i64 %413
  call void @llvm.memcpy.p0.p0.i64(ptr %399, ptr %414, i64 %411, i1 false)
  %415 = getelementptr float, ptr %399, i64 %379
  call void @llvm.memcpy.p0.p0.i64(ptr %415, ptr %99, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %416 = add i64 %379, 48
  br label %378

417:                                              ; preds = %378
  %418 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %380, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %101, ptr %100, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %419

419:                                              ; preds = %438, %417
  %420 = phi i64 [ %457, %438 ], [ 0, %417 ]
  %421 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %447, %438 ], [ %108, %417 ]
  %422 = icmp slt i64 %420, 768
  br i1 %422, label %423, label %458

423:                                              ; preds = %419
  %424 = mul i64 %129, 221184
  %425 = mul i64 %420, 288
  %426 = add i64 %424, %425
  %427 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  br label %428

428:                                              ; preds = %431, %423
  %429 = phi i64 [ %437, %431 ], [ 0, %423 ]
  %430 = icmp slt i64 %429, 288
  br i1 %430, label %431, label %438

431:                                              ; preds = %428
  %432 = add i64 %429, 0
  %433 = getelementptr float, ptr %377, i64 %432
  %434 = load float, ptr %433, align 4
  %435 = add i64 0, %429
  %436 = getelementptr float, ptr %109, i64 %435
  store float %434, ptr %436, align 4
  %437 = add i64 %429, 1
  br label %428

438:                                              ; preds = %428
  call void @upmemrt_dpu_load(ptr %427, ptr @dpu_program_5)
  %439 = getelementptr inbounds float, ptr %12, i64 %426
  call void @upmemrt_dpu_scatter(ptr %427, ptr %439, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %427, ptr %109, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %427, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %427)
  call void @upmemrt_dpu_gather(ptr %427, ptr %110, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %427)
  %440 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %441 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %440, 0
  %442 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %441, ptr %440, 1
  %443 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %442, i64 0, 2
  %444 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %443, i64 768, 3, 0
  %445 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %444, i64 1, 3, 1
  %446 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %445, i64 1, 4, 0
  %447 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %446, i64 1, 4, 1
  %448 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %421, 3, 0
  %449 = mul i64 %448, 1
  %450 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %421, 3, 1
  %451 = mul i64 %449, %450
  %452 = mul i64 %451, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %453 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %421, 1
  %454 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %421, 2
  %455 = getelementptr float, ptr %453, i64 %454
  call void @llvm.memcpy.p0.p0.i64(ptr %440, ptr %455, i64 %452, i1 false)
  %456 = getelementptr float, ptr %440, i64 %420
  call void @llvm.memcpy.p0.p0.i64(ptr %456, ptr %110, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %457 = add i64 %420, 48
  br label %419

458:                                              ; preds = %419
  %459 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %421, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %111, ptr %418, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %460

460:                                              ; preds = %464, %458
  %461 = phi i64 [ %486, %464 ], [ 0, %458 ]
  %462 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %479, %464 ], [ %116, %458 ]
  %463 = icmp slt i64 %461, 768
  br i1 %463, label %464, label %487

464:                                              ; preds = %460
  %465 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %462, 1
  %466 = getelementptr float, ptr %465, i64 %461
  %467 = load float, ptr %466, align 4
  %468 = getelementptr float, ptr %459, i64 %461
  %469 = load float, ptr %468, align 4
  %470 = call float @llvm.exp.f32(float %467)
  %471 = fadd float %470, 1.000000e+00
  %472 = fdiv float 1.000000e+00, %471
  %473 = fmul float %469, %472
  %474 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64))
  %475 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %474, 0
  %476 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %475, ptr %474, 1
  %477 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %476, i64 0, 2
  %478 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %477, i64 768, 3, 0
  %479 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %478, i64 1, 4, 0
  %480 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %462, 3, 0
  %481 = mul i64 %480, 1
  %482 = mul i64 %481, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %483 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %462, 2
  %484 = getelementptr float, ptr %465, i64 %483
  call void @llvm.memcpy.p0.p0.i64(ptr %474, ptr %484, i64 %482, i1 false)
  %485 = getelementptr float, ptr %474, i64 %461
  store float %473, ptr %485, align 4
  %486 = add i64 %461, 1
  br label %460

487:                                              ; preds = %460
  %488 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %462, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %118, ptr %117, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %489

489:                                              ; preds = %508, %487
  %490 = phi i64 [ %527, %508 ], [ 0, %487 ]
  %491 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %517, %508 ], [ %125, %487 ]
  %492 = icmp slt i64 %490, 288
  br i1 %492, label %493, label %528

493:                                              ; preds = %489
  %494 = mul i64 %129, 221184
  %495 = mul i64 %490, 768
  %496 = add i64 %494, %495
  %497 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  br label %498

498:                                              ; preds = %501, %493
  %499 = phi i64 [ %507, %501 ], [ 0, %493 ]
  %500 = icmp slt i64 %499, 768
  br i1 %500, label %501, label %508

501:                                              ; preds = %498
  %502 = add i64 %499, 0
  %503 = getelementptr float, ptr %488, i64 %502
  %504 = load float, ptr %503, align 4
  %505 = add i64 0, %499
  %506 = getelementptr float, ptr %126, i64 %505
  store float %504, ptr %506, align 4
  %507 = add i64 %499, 1
  br label %498

508:                                              ; preds = %498
  call void @upmemrt_dpu_load(ptr %497, ptr @dpu_program_6)
  %509 = getelementptr inbounds float, ptr %11, i64 %496
  call void @upmemrt_dpu_scatter(ptr %497, ptr %509, i64 4, i64 36864, i64 768, i64 24576, i64 0, ptr @scatter_map_3)
  call void @upmemrt_dpu_scatter(ptr %497, ptr %126, i64 4, i64 768, i64 16, i64 24576, i64 24576, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %497, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 49152, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %497)
  call void @upmemrt_dpu_gather(ptr %497, ptr %127, i64 4, i64 48, i64 1, i64 32, i64 49152, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %497)
  %510 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %511 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %510, 0
  %512 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %511, ptr %510, 1
  %513 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %512, i64 0, 2
  %514 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %513, i64 288, 3, 0
  %515 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %514, i64 1, 3, 1
  %516 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %515, i64 1, 4, 0
  %517 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %516, i64 1, 4, 1
  %518 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %491, 3, 0
  %519 = mul i64 %518, 1
  %520 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %491, 3, 1
  %521 = mul i64 %519, %520
  %522 = mul i64 %521, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %523 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %491, 1
  %524 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %491, 2
  %525 = getelementptr float, ptr %523, i64 %524
  call void @llvm.memcpy.p0.p0.i64(ptr %510, ptr %525, i64 %522, i1 false)
  %526 = getelementptr float, ptr %510, i64 %490
  call void @llvm.memcpy.p0.p0.i64(ptr %526, ptr %127, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %527 = add i64 %490, 48
  br label %489

528:                                              ; preds = %489
  %529 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %491, 1
  %530 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6)
  call void @upmemrt_dpu_load(ptr %530, ptr @dpu_program_7)
  call void @upmemrt_dpu_scatter(ptr %530, ptr %137, i64 4, i64 288, i64 6, i64 192, i64 0, ptr @scatter_map_2)
  call void @upmemrt_dpu_scatter(ptr %530, ptr %529, i64 4, i64 288, i64 6, i64 192, i64 192, ptr @scatter_map_2)
  call void @upmemrt_dpu_scatter(ptr %530, ptr @__constant_48x6xf32, i64 4, i64 288, i64 6, i64 192, i64 384, ptr @scatter_map_2)
  call void @upmemrt_dpu_launch(ptr %530)
  %531 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  call void @upmemrt_dpu_gather(ptr %530, ptr %531, i64 4, i64 288, i64 6, i64 192, i64 384, ptr @scatter_map_2)
  %532 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %531, 0
  %533 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %532, ptr %531, 1
  %534 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %533, i64 0, 2
  %535 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %534, i64 288, 3, 0
  %536 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %535, i64 1, 4, 0
  call void @upmemrt_dpu_free(ptr %530)
  %537 = add i64 %129, 1
  br label %128

538:                                              ; preds = %128
  %539 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %540 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, 3, 0
  %541 = mul i64 %540, 1
  %542 = mul i64 %541, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %543 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, 1
  %544 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, 2
  %545 = getelementptr float, ptr %543, i64 %544
  call void @llvm.memcpy.p0.p0.i64(ptr %539, ptr %545, i64 %542, i1 false)
  %546 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %546, ptr %14, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %547 = call ptr @rmsnorm(ptr %539, ptr %546)
  %548 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64))
  br label %549

549:                                              ; preds = %561, %538
  %550 = phi i64 [ %562, %561 ], [ 0, %538 ]
  %551 = icmp slt i64 %550, 32768
  br i1 %551, label %552, label %563

552:                                              ; preds = %549
  br label %553

553:                                              ; preds = %556, %552
  %554 = phi i64 [ %560, %556 ], [ 0, %552 ]
  %555 = icmp slt i64 %554, 288
  br i1 %555, label %556, label %561

556:                                              ; preds = %553
  %557 = mul i64 %550, 288
  %558 = add i64 %557, %554
  %559 = getelementptr float, ptr %548, i64 %558
  store float 0.000000e+00, ptr %559, align 4
  %560 = add i64 %554, 1
  br label %553

561:                                              ; preds = %553
  %562 = add i64 %550, 1
  br label %549

563:                                              ; preds = %549
  call void @llvm.memcpy.p0.p0.i64(ptr %548, ptr %15, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9216000), i1 false)
  %564 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64))
  %565 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64))
  %566 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %565, 0
  %567 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %566, ptr %565, 1
  %568 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %567, i64 0, 2
  %569 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %568, i64 32768, 3, 0
  %570 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %569, i64 1, 3, 1
  %571 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %570, i64 1, 4, 0
  %572 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %571, i64 1, 4, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %565, ptr %564, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 32768), i1 false)
  %573 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %574 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  br label %575

575:                                              ; preds = %592, %563
  %576 = phi i64 [ %611, %592 ], [ 0, %563 ]
  %577 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %601, %592 ], [ %572, %563 ]
  %578 = icmp slt i64 %576, 32768
  br i1 %578, label %579, label %612

579:                                              ; preds = %575
  %580 = mul i64 %576, 288
  %581 = call ptr @upmemrt_dpu_alloc(i32 2, i32 8)
  br label %582

582:                                              ; preds = %585, %579
  %583 = phi i64 [ %591, %585 ], [ 0, %579 ]
  %584 = icmp slt i64 %583, 288
  br i1 %584, label %585, label %592

585:                                              ; preds = %582
  %586 = add i64 %583, 0
  %587 = getelementptr float, ptr %547, i64 %586
  %588 = load float, ptr %587, align 4
  %589 = add i64 0, %583
  %590 = getelementptr float, ptr %573, i64 %589
  store float %588, ptr %590, align 4
  %591 = add i64 %583, 1
  br label %582

592:                                              ; preds = %582
  call void @upmemrt_dpu_load(ptr %581, ptr @dpu_program_8)
  %593 = getelementptr inbounds float, ptr %548, i64 %580
  call void @upmemrt_dpu_scatter(ptr %581, ptr %593, i64 4, i64 73728, i64 288, i64 18432, i64 0, ptr @scatter_map_4)
  call void @upmemrt_dpu_scatter(ptr %581, ptr %573, i64 4, i64 288, i64 1, i64 18432, i64 18432, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %581, ptr @__constant_256x1xf32, i64 4, i64 256, i64 1, i64 64, i64 36864, ptr @scatter_map_5)
  call void @upmemrt_dpu_launch(ptr %581)
  call void @upmemrt_dpu_gather(ptr %581, ptr %574, i64 4, i64 256, i64 1, i64 64, i64 36864, ptr @scatter_map_5)
  call void @upmemrt_dpu_free(ptr %581)
  %594 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64))
  %595 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %594, 0
  %596 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %595, ptr %594, 1
  %597 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %596, i64 0, 2
  %598 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %597, i64 32768, 3, 0
  %599 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %598, i64 1, 3, 1
  %600 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %599, i64 1, 4, 0
  %601 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %600, i64 1, 4, 1
  %602 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %577, 3, 0
  %603 = mul i64 %602, 1
  %604 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %577, 3, 1
  %605 = mul i64 %603, %604
  %606 = mul i64 %605, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %607 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %577, 1
  %608 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %577, 2
  %609 = getelementptr float, ptr %607, i64 %608
  call void @llvm.memcpy.p0.p0.i64(ptr %594, ptr %609, i64 %606, i1 false)
  %610 = getelementptr float, ptr %594, i64 %576
  call void @llvm.memcpy.p0.p0.i64(ptr %610, ptr %574, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 256), i1 false)
  %611 = add i64 %576, 256
  br label %575

612:                                              ; preds = %575
  %613 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %577, 0
  ret ptr %613
}

define ptr @rot(ptr %0, i64 %1, float %2, float %3) {
  %5 = add i64 %1, 1
  %6 = getelementptr float, ptr %0, i64 %1
  %7 = load float, ptr %6, align 4
  %8 = getelementptr float, ptr %0, i64 %5
  %9 = load float, ptr %8, align 4
  %10 = fmul float %7, %2
  %11 = fmul float %9, %3
  %12 = fsub float %10, %11
  %13 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %13, ptr %0, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %14 = getelementptr float, ptr %13, i64 %1
  store float %12, ptr %14, align 4
  %15 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %15, ptr %13, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %16 = getelementptr float, ptr %15, i64 %1
  store float %12, ptr %16, align 4
  ret ptr %15
}

define ptr @mha(ptr %0, ptr %1, ptr %2, i64 %3) {
  %5 = add i64 %3, 1
  %6 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  br label %7

7:                                                ; preds = %10, %4
  %8 = phi i64 [ %12, %10 ], [ 0, %4 ]
  %9 = icmp slt i64 %8, 256
  br i1 %9, label %10, label %13

10:                                               ; preds = %7
  %11 = getelementptr float, ptr %6, i64 %8
  store float 0xFFF0000000000000, ptr %11, align 4
  %12 = add i64 %8, 1
  br label %7

13:                                               ; preds = %7
  %14 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %15 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %15, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 0, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 288, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %15, ptr %14, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %21 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %21, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, ptr %21, 1
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 0, 2
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 256, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 1, 4, 0
  %27 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %28 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %29 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %30 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %31 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %32 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %33 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  %34 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %35 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %35, 0
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, ptr %35, 1
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, i64 0, 2
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 48, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 1, 4, 0
  %41 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  %42 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64))
  %43 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  br label %44

44:                                               ; preds = %132, %13
  %45 = phi i64 [ %152, %132 ], [ 0, %13 ]
  %46 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %138, %132 ], [ %20, %13 ]
  %47 = icmp slt i64 %45, 6
  br i1 %47, label %48, label %153

48:                                               ; preds = %44
  %49 = mul i64 %45, 48
  call void @llvm.memcpy.p0.p0.i64(ptr %21, ptr %6, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 256), i1 false)
  br label %50

50:                                               ; preds = %69, %48
  %51 = phi i64 [ %89, %69 ], [ 0, %48 ]
  %52 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %81, %69 ], [ %26, %48 ]
  %53 = icmp slt i64 %51, %5
  br i1 %53, label %54, label %90

54:                                               ; preds = %50
  %55 = mul i64 %51, 288
  %56 = add i64 %55, %49
  %57 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1)
  call void @upmemrt_dpu_load(ptr %57, ptr @dpu_program_9)
  %58 = getelementptr float, ptr %0, i64 %49
  call void @llvm.memcpy.p0.p0.i64(ptr %27, ptr %58, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  call void @upmemrt_dpu_scatter(ptr %57, ptr %27, i64 4, i64 48, i64 6, i64 192, i64 0, ptr @scatter_map_6)
  %59 = getelementptr float, ptr %1, i64 %56
  call void @llvm.memcpy.p0.p0.i64(ptr %28, ptr %59, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  call void @upmemrt_dpu_scatter(ptr %57, ptr %28, i64 4, i64 48, i64 6, i64 192, i64 192, ptr @scatter_map_6)
  call void @upmemrt_dpu_scatter(ptr %57, ptr @__constant_8x6xf32, i64 4, i64 48, i64 6, i64 192, i64 384, ptr @scatter_map_6)
  call void @upmemrt_dpu_launch(ptr %57)
  call void @upmemrt_dpu_gather(ptr %57, ptr %29, i64 4, i64 48, i64 6, i64 192, i64 384, ptr @scatter_map_6)
  call void @upmemrt_dpu_free(ptr %57)
  call void @llvm.memcpy.p0.p0.i64(ptr %31, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %60

60:                                               ; preds = %63, %54
  %61 = phi i64 [ %68, %63 ], [ 0, %54 ]
  %62 = icmp slt i64 %61, 48
  br i1 %62, label %63, label %69

63:                                               ; preds = %60
  %64 = getelementptr float, ptr %29, i64 %61
  %65 = load float, ptr %64, align 4
  %66 = load float, ptr %31, align 4
  %67 = fadd float %65, %66
  store float %67, ptr %31, align 4
  %68 = add i64 %61, 1
  br label %60

69:                                               ; preds = %60
  %70 = load float, ptr %31, align 4
  store float %70, ptr %30, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %32, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %71 = load float, ptr %30, align 4
  %72 = load float, ptr %32, align 4
  %73 = fadd float %71, %72
  store float %73, ptr %32, align 4
  %74 = load float, ptr %32, align 4
  %75 = fdiv float %74, 0x401BB67AE0000000
  %76 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  %77 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %76, 0
  %78 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %77, ptr %76, 1
  %79 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %78, i64 0, 2
  %80 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %79, i64 256, 3, 0
  %81 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %80, i64 1, 4, 0
  %82 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 3, 0
  %83 = mul i64 %82, 1
  %84 = mul i64 %83, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %85 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 1
  %86 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 2
  %87 = getelementptr float, ptr %85, i64 %86
  call void @llvm.memcpy.p0.p0.i64(ptr %76, ptr %87, i64 %84, i1 false)
  %88 = getelementptr float, ptr %76, i64 %51
  store float %75, ptr %88, align 4
  %89 = add i64 %51, 1
  br label %50

90:                                               ; preds = %50
  %91 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 3, 0
  %92 = mul i64 %91, 1
  %93 = mul i64 %92, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %94 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 1
  %95 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 2
  %96 = getelementptr float, ptr %94, i64 %95
  call void @llvm.memcpy.p0.p0.i64(ptr %33, ptr %96, i64 %93, i1 false)
  %97 = call ptr @softmax(ptr %33)
  br label %98

98:                                               ; preds = %101, %90
  %99 = phi i64 [ %103, %101 ], [ 0, %90 ]
  %100 = icmp slt i64 %99, 48
  br i1 %100, label %101, label %104

101:                                              ; preds = %98
  %102 = getelementptr float, ptr %34, i64 %99
  store float 0.000000e+00, ptr %102, align 4
  %103 = add i64 %99, 1
  br label %98

104:                                              ; preds = %98
  call void @llvm.memcpy.p0.p0.i64(ptr %35, ptr %34, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  br label %105

105:                                              ; preds = %109, %104
  %106 = phi i64 [ %131, %109 ], [ 0, %104 ]
  %107 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %130, %109 ], [ %40, %104 ]
  %108 = icmp slt i64 %106, %5
  br i1 %108, label %109, label %132

109:                                              ; preds = %105
  %110 = mul i64 %106, 288
  %111 = add i64 %110, %49
  %112 = getelementptr float, ptr %97, i64 %106
  %113 = load float, ptr %112, align 4
  %114 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1)
  call void @upmemrt_dpu_load(ptr %114, ptr @dpu_program_10)
  %115 = getelementptr float, ptr %2, i64 %111
  call void @llvm.memcpy.p0.p0.i64(ptr %41, ptr %115, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  call void @upmemrt_dpu_scatter(ptr %114, ptr %41, i64 4, i64 48, i64 6, i64 192, i64 0, ptr @scatter_map_6)
  store float %113, ptr %42, align 4
  %116 = getelementptr float, ptr %42, i32 1
  store float %113, ptr %116, align 4
  %117 = getelementptr float, ptr %42, i32 2
  store float %113, ptr %117, align 4
  %118 = getelementptr float, ptr %42, i32 3
  store float %113, ptr %118, align 4
  %119 = getelementptr float, ptr %42, i32 4
  store float %113, ptr %119, align 4
  %120 = getelementptr float, ptr %42, i32 5
  store float %113, ptr %120, align 4
  %121 = getelementptr float, ptr %42, i32 6
  store float %113, ptr %121, align 4
  %122 = getelementptr float, ptr %42, i32 7
  store float %113, ptr %122, align 4
  call void @upmemrt_dpu_scatter(ptr %114, ptr %42, i64 4, i64 8, i64 1, i64 32, i64 192, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %114, ptr @__constant_8x6xf32, i64 4, i64 48, i64 6, i64 192, i64 224, ptr @scatter_map_6)
  call void @upmemrt_dpu_launch(ptr %114)
  call void @upmemrt_dpu_gather(ptr %114, ptr %43, i64 4, i64 48, i64 6, i64 192, i64 224, ptr @scatter_map_6)
  call void @upmemrt_dpu_free(ptr %114)
  %123 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1)
  call void @upmemrt_dpu_load(ptr %123, ptr @dpu_program_11)
  %124 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %107, 1
  call void @upmemrt_dpu_scatter(ptr %123, ptr %124, i64 4, i64 48, i64 6, i64 192, i64 0, ptr @scatter_map_6)
  call void @upmemrt_dpu_scatter(ptr %123, ptr %43, i64 4, i64 48, i64 6, i64 192, i64 192, ptr @scatter_map_6)
  call void @upmemrt_dpu_scatter(ptr %123, ptr @__constant_8x6xf32, i64 4, i64 48, i64 6, i64 192, i64 384, ptr @scatter_map_6)
  call void @upmemrt_dpu_launch(ptr %123)
  %125 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64))
  call void @upmemrt_dpu_gather(ptr %123, ptr %125, i64 4, i64 48, i64 6, i64 192, i64 384, ptr @scatter_map_6)
  %126 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %125, 0
  %127 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %126, ptr %125, 1
  %128 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %127, i64 0, 2
  %129 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %128, i64 48, 3, 0
  %130 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %129, i64 1, 4, 0
  call void @upmemrt_dpu_free(ptr %123)
  %131 = add i64 %106, 1
  br label %105

132:                                              ; preds = %105
  %133 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  %134 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %133, 0
  %135 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %134, ptr %133, 1
  %136 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %135, i64 0, 2
  %137 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %136, i64 288, 3, 0
  %138 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %137, i64 1, 4, 0
  %139 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 3, 0
  %140 = mul i64 %139, 1
  %141 = mul i64 %140, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %142 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 1
  %143 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 2
  %144 = getelementptr float, ptr %142, i64 %143
  call void @llvm.memcpy.p0.p0.i64(ptr %133, ptr %144, i64 %141, i1 false)
  %145 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %107, 3, 0
  %146 = mul i64 %145, 1
  %147 = mul i64 %146, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %148 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %107, 1
  %149 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %107, 2
  %150 = getelementptr float, ptr %148, i64 %149
  %151 = getelementptr float, ptr %133, i64 %49
  call void @llvm.memcpy.p0.p0.i64(ptr %151, ptr %150, i64 %147, i1 false)
  %152 = add i64 %45, 1
  br label %44

153:                                              ; preds = %44
  %154 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, 0
  ret ptr %154
}

define ptr @rmsnorm(ptr %0, ptr %1) {
  %3 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1)
  call void @upmemrt_dpu_load(ptr %3, ptr @dpu_program_12)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 288, i64 18, i64 1152, i64 0, ptr @scatter_map_7)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 288, i64 18, i64 1152, i64 1152, ptr @scatter_map_7)
  call void @upmemrt_dpu_scatter(ptr %3, ptr @__constant_16x18xf32, i64 4, i64 288, i64 18, i64 1152, i64 2304, ptr @scatter_map_7)
  call void @upmemrt_dpu_launch(ptr %3)
  %4 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  call void @upmemrt_dpu_gather(ptr %3, ptr %4, i64 4, i64 288, i64 18, i64 1152, i64 2304, ptr @scatter_map_7)
  call void @upmemrt_dpu_free(ptr %3)
  %5 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %6 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %7

7:                                                ; preds = %10, %2
  %8 = phi i64 [ %15, %10 ], [ 0, %2 ]
  %9 = icmp slt i64 %8, 288
  br i1 %9, label %10, label %16

10:                                               ; preds = %7
  %11 = getelementptr float, ptr %4, i64 %8
  %12 = load float, ptr %11, align 4
  %13 = load float, ptr %6, align 4
  %14 = fadd float %12, %13
  store float %14, ptr %6, align 4
  %15 = add i64 %8, 1
  br label %7

16:                                               ; preds = %7
  %17 = load float, ptr %6, align 4
  store float %17, ptr %5, align 4
  %18 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %18, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %19 = load float, ptr %5, align 4
  %20 = load float, ptr %18, align 4
  %21 = fadd float %19, %20
  store float %21, ptr %18, align 4
  %22 = load float, ptr %18, align 4
  %23 = fdiv float %22, 2.880000e+02
  %24 = fadd float %23, 0x3EE4F8B580000000
  %25 = call float @llvm.sqrt.f32(float %24)
  %26 = fdiv float 1.000000e+00, %25
  %27 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1)
  call void @upmemrt_dpu_load(ptr %27, ptr @dpu_program_13)
  call void @upmemrt_dpu_scatter(ptr %27, ptr %0, i64 4, i64 288, i64 18, i64 1152, i64 0, ptr @scatter_map_7)
  %28 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %26, ptr %28, align 4
  %29 = getelementptr float, ptr %28, i32 1
  store float %26, ptr %29, align 4
  %30 = getelementptr float, ptr %28, i32 2
  store float %26, ptr %30, align 4
  %31 = getelementptr float, ptr %28, i32 3
  store float %26, ptr %31, align 4
  %32 = getelementptr float, ptr %28, i32 4
  store float %26, ptr %32, align 4
  %33 = getelementptr float, ptr %28, i32 5
  store float %26, ptr %33, align 4
  %34 = getelementptr float, ptr %28, i32 6
  store float %26, ptr %34, align 4
  %35 = getelementptr float, ptr %28, i32 7
  store float %26, ptr %35, align 4
  %36 = getelementptr float, ptr %28, i32 8
  store float %26, ptr %36, align 4
  %37 = getelementptr float, ptr %28, i32 9
  store float %26, ptr %37, align 4
  %38 = getelementptr float, ptr %28, i32 10
  store float %26, ptr %38, align 4
  %39 = getelementptr float, ptr %28, i32 11
  store float %26, ptr %39, align 4
  %40 = getelementptr float, ptr %28, i32 12
  store float %26, ptr %40, align 4
  %41 = getelementptr float, ptr %28, i32 13
  store float %26, ptr %41, align 4
  %42 = getelementptr float, ptr %28, i32 14
  store float %26, ptr %42, align 4
  %43 = getelementptr float, ptr %28, i32 15
  store float %26, ptr %43, align 4
  call void @upmemrt_dpu_scatter(ptr %27, ptr %28, i64 4, i64 16, i64 1, i64 64, i64 1152, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %27, ptr @__constant_16x18xf32, i64 4, i64 288, i64 18, i64 1152, i64 1216, ptr @scatter_map_7)
  call void @upmemrt_dpu_launch(ptr %27)
  %44 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  call void @upmemrt_dpu_gather(ptr %27, ptr %44, i64 4, i64 288, i64 18, i64 1152, i64 1216, ptr @scatter_map_7)
  call void @upmemrt_dpu_free(ptr %27)
  %45 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1)
  call void @upmemrt_dpu_load(ptr %45, ptr @dpu_program_14)
  call void @upmemrt_dpu_scatter(ptr %45, ptr %44, i64 4, i64 288, i64 18, i64 1152, i64 0, ptr @scatter_map_7)
  call void @upmemrt_dpu_scatter(ptr %45, ptr %1, i64 4, i64 288, i64 18, i64 1152, i64 1152, ptr @scatter_map_7)
  call void @upmemrt_dpu_scatter(ptr %45, ptr @__constant_16x18xf32, i64 4, i64 288, i64 18, i64 1152, i64 2304, ptr @scatter_map_7)
  call void @upmemrt_dpu_launch(ptr %45)
  %46 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64))
  call void @upmemrt_dpu_gather(ptr %45, ptr %46, i64 4, i64 288, i64 18, i64 1152, i64 2304, ptr @scatter_map_7)
  call void @upmemrt_dpu_free(ptr %45)
  ret ptr %46
}

define ptr @softmax(ptr %0) {
  %2 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %3 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %3, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %4

4:                                                ; preds = %7, %1
  %5 = phi i64 [ %12, %7 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 256
  br i1 %6, label %7, label %13

7:                                                ; preds = %4
  %8 = getelementptr float, ptr %0, i64 %5
  %9 = load float, ptr %8, align 4
  %10 = load float, ptr %3, align 4
  %11 = call float @llvm.maximum.f32(float %9, float %10)
  store float %11, ptr %3, align 4
  %12 = add i64 %5, 1
  br label %4

13:                                               ; preds = %4
  %14 = load float, ptr %3, align 4
  store float %14, ptr %2, align 4
  %15 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %15, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %16 = load float, ptr %2, align 4
  %17 = load float, ptr %15, align 4
  %18 = call float @llvm.maximum.f32(float %16, float %17)
  store float %18, ptr %15, align 4
  %19 = load float, ptr %15, align 4
  %20 = call ptr @upmemrt_dpu_alloc(i32 1, i32 8)
  call void @upmemrt_dpu_load(ptr %20, ptr @dpu_program_15)
  call void @upmemrt_dpu_scatter(ptr %20, ptr %0, i64 4, i64 256, i64 2, i64 128, i64 0, ptr @scatter_map_8)
  %21 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %19, ptr %21, align 4
  %22 = getelementptr float, ptr %21, i32 1
  store float %19, ptr %22, align 4
  %23 = getelementptr float, ptr %21, i32 2
  store float %19, ptr %23, align 4
  %24 = getelementptr float, ptr %21, i32 3
  store float %19, ptr %24, align 4
  %25 = getelementptr float, ptr %21, i32 4
  store float %19, ptr %25, align 4
  %26 = getelementptr float, ptr %21, i32 5
  store float %19, ptr %26, align 4
  %27 = getelementptr float, ptr %21, i32 6
  store float %19, ptr %27, align 4
  %28 = getelementptr float, ptr %21, i32 7
  store float %19, ptr %28, align 4
  %29 = getelementptr float, ptr %21, i32 8
  store float %19, ptr %29, align 4
  %30 = getelementptr float, ptr %21, i32 9
  store float %19, ptr %30, align 4
  %31 = getelementptr float, ptr %21, i32 10
  store float %19, ptr %31, align 4
  %32 = getelementptr float, ptr %21, i32 11
  store float %19, ptr %32, align 4
  %33 = getelementptr float, ptr %21, i32 12
  store float %19, ptr %33, align 4
  %34 = getelementptr float, ptr %21, i32 13
  store float %19, ptr %34, align 4
  %35 = getelementptr float, ptr %21, i32 14
  store float %19, ptr %35, align 4
  %36 = getelementptr float, ptr %21, i32 15
  store float %19, ptr %36, align 4
  call void @upmemrt_dpu_scatter(ptr %20, ptr %21, i64 4, i64 16, i64 0, i64 64, i64 128, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %20, ptr @__constant_128x2xf32, i64 4, i64 256, i64 2, i64 128, i64 192, ptr @scatter_map_8)
  call void @upmemrt_dpu_launch(ptr %20)
  %37 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  call void @upmemrt_dpu_gather(ptr %20, ptr %37, i64 4, i64 256, i64 2, i64 128, i64 192, ptr @scatter_map_8)
  call void @upmemrt_dpu_free(ptr %20)
  %38 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  br label %39

39:                                               ; preds = %42, %13
  %40 = phi i64 [ %47, %42 ], [ 0, %13 ]
  %41 = icmp slt i64 %40, 256
  br i1 %41, label %42, label %48

42:                                               ; preds = %39
  %43 = getelementptr float, ptr %37, i64 %40
  %44 = load float, ptr %43, align 4
  %45 = call float @llvm.exp.f32(float %44)
  %46 = getelementptr float, ptr %38, i64 %40
  store float %45, ptr %46, align 4
  %47 = add i64 %40, 1
  br label %39

48:                                               ; preds = %39
  %49 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %50 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %50, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %51

51:                                               ; preds = %54, %48
  %52 = phi i64 [ %59, %54 ], [ 0, %48 ]
  %53 = icmp slt i64 %52, 256
  br i1 %53, label %54, label %60

54:                                               ; preds = %51
  %55 = getelementptr float, ptr %38, i64 %52
  %56 = load float, ptr %55, align 4
  %57 = load float, ptr %50, align 4
  %58 = fadd float %56, %57
  store float %58, ptr %50, align 4
  %59 = add i64 %52, 1
  br label %51

60:                                               ; preds = %51
  %61 = load float, ptr %50, align 4
  store float %61, ptr %49, align 4
  %62 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %62, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %63 = load float, ptr %49, align 4
  %64 = load float, ptr %62, align 4
  %65 = fadd float %63, %64
  store float %65, ptr %62, align 4
  %66 = load float, ptr %62, align 4
  %67 = call ptr @upmemrt_dpu_alloc(i32 1, i32 8)
  call void @upmemrt_dpu_load(ptr %67, ptr @dpu_program_16)
  call void @upmemrt_dpu_scatter(ptr %67, ptr %38, i64 4, i64 256, i64 2, i64 128, i64 0, ptr @scatter_map_8)
  %68 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %66, ptr %68, align 4
  %69 = getelementptr float, ptr %68, i32 1
  store float %66, ptr %69, align 4
  %70 = getelementptr float, ptr %68, i32 2
  store float %66, ptr %70, align 4
  %71 = getelementptr float, ptr %68, i32 3
  store float %66, ptr %71, align 4
  %72 = getelementptr float, ptr %68, i32 4
  store float %66, ptr %72, align 4
  %73 = getelementptr float, ptr %68, i32 5
  store float %66, ptr %73, align 4
  %74 = getelementptr float, ptr %68, i32 6
  store float %66, ptr %74, align 4
  %75 = getelementptr float, ptr %68, i32 7
  store float %66, ptr %75, align 4
  %76 = getelementptr float, ptr %68, i32 8
  store float %66, ptr %76, align 4
  %77 = getelementptr float, ptr %68, i32 9
  store float %66, ptr %77, align 4
  %78 = getelementptr float, ptr %68, i32 10
  store float %66, ptr %78, align 4
  %79 = getelementptr float, ptr %68, i32 11
  store float %66, ptr %79, align 4
  %80 = getelementptr float, ptr %68, i32 12
  store float %66, ptr %80, align 4
  %81 = getelementptr float, ptr %68, i32 13
  store float %66, ptr %81, align 4
  %82 = getelementptr float, ptr %68, i32 14
  store float %66, ptr %82, align 4
  %83 = getelementptr float, ptr %68, i32 15
  store float %66, ptr %83, align 4
  call void @upmemrt_dpu_scatter(ptr %67, ptr %68, i64 4, i64 16, i64 0, i64 64, i64 128, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %67, ptr @__constant_128x2xf32, i64 4, i64 256, i64 2, i64 128, i64 192, ptr @scatter_map_8)
  call void @upmemrt_dpu_launch(ptr %67)
  %84 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  call void @upmemrt_dpu_gather(ptr %67, ptr %84, i64 4, i64 256, i64 2, i64 128, i64 192, ptr @scatter_map_8)
  call void @upmemrt_dpu_free(ptr %67)
  ret ptr %84
}

define ptr @rmsnorm_1048576(ptr %0, ptr %1) {
  %3 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %3, ptr @dpu_program_17)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 1048576, i64 256, i64 16384, i64 0, ptr @scatter_map_9)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 1048576, i64 256, i64 16384, i64 16384, ptr @scatter_map_9)
  call void @upmemrt_dpu_scatter(ptr %3, ptr @__constant_4096x256xf32, i64 4, i64 1048576, i64 256, i64 16384, i64 32768, ptr @scatter_map_9)
  call void @upmemrt_dpu_launch(ptr %3)
  %4 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  call void @upmemrt_dpu_gather(ptr %3, ptr %4, i64 4, i64 1048576, i64 256, i64 16384, i64 32768, ptr @scatter_map_9)
  call void @upmemrt_dpu_free(ptr %3)
  %5 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %6 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  br label %7

7:                                                ; preds = %22, %2
  %8 = phi i64 [ %25, %22 ], [ 0, %2 ]
  %9 = icmp slt i64 %8, 1024
  br i1 %9, label %10, label %26

10:                                               ; preds = %7
  %11 = mul i64 %8, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %12

12:                                               ; preds = %15, %10
  %13 = phi i64 [ %21, %15 ], [ 0, %10 ]
  %14 = icmp slt i64 %13, 1024
  br i1 %14, label %15, label %22

15:                                               ; preds = %12
  %16 = add i64 %11, %13
  %17 = getelementptr float, ptr %4, i64 %16
  %18 = load float, ptr %17, align 4
  %19 = load float, ptr %6, align 4
  %20 = fadd float %18, %19
  store float %20, ptr %6, align 4
  %21 = add i64 %13, 1
  br label %12

22:                                               ; preds = %12
  %23 = load float, ptr %6, align 4
  %24 = getelementptr float, ptr %5, i64 %8
  store float %23, ptr %24, align 4
  %25 = add i64 %8, 1
  br label %7

26:                                               ; preds = %7
  %27 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %27, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %28

28:                                               ; preds = %31, %26
  %29 = phi i64 [ %36, %31 ], [ 0, %26 ]
  %30 = icmp slt i64 %29, 1024
  br i1 %30, label %31, label %37

31:                                               ; preds = %28
  %32 = getelementptr float, ptr %5, i64 %29
  %33 = load float, ptr %32, align 4
  %34 = load float, ptr %27, align 4
  %35 = fadd float %33, %34
  store float %35, ptr %27, align 4
  %36 = add i64 %29, 1
  br label %28

37:                                               ; preds = %28
  %38 = load float, ptr %27, align 4
  %39 = fdiv float %38, 0x4130000000000000
  %40 = fadd float %39, 0x3EE4F8B580000000
  %41 = call float @llvm.sqrt.f32(float %40)
  %42 = fdiv float 1.000000e+00, %41
  %43 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %43, ptr @dpu_program_18)
  call void @upmemrt_dpu_scatter(ptr %43, ptr %0, i64 4, i64 1048576, i64 256, i64 16384, i64 0, ptr @scatter_map_9)
  %44 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %42, ptr %44, align 4
  %45 = getelementptr float, ptr %44, i32 1
  store float %42, ptr %45, align 4
  %46 = getelementptr float, ptr %44, i32 2
  store float %42, ptr %46, align 4
  %47 = getelementptr float, ptr %44, i32 3
  store float %42, ptr %47, align 4
  %48 = getelementptr float, ptr %44, i32 4
  store float %42, ptr %48, align 4
  %49 = getelementptr float, ptr %44, i32 5
  store float %42, ptr %49, align 4
  %50 = getelementptr float, ptr %44, i32 6
  store float %42, ptr %50, align 4
  %51 = getelementptr float, ptr %44, i32 7
  store float %42, ptr %51, align 4
  %52 = getelementptr float, ptr %44, i32 8
  store float %42, ptr %52, align 4
  %53 = getelementptr float, ptr %44, i32 9
  store float %42, ptr %53, align 4
  %54 = getelementptr float, ptr %44, i32 10
  store float %42, ptr %54, align 4
  %55 = getelementptr float, ptr %44, i32 11
  store float %42, ptr %55, align 4
  %56 = getelementptr float, ptr %44, i32 12
  store float %42, ptr %56, align 4
  %57 = getelementptr float, ptr %44, i32 13
  store float %42, ptr %57, align 4
  %58 = getelementptr float, ptr %44, i32 14
  store float %42, ptr %58, align 4
  %59 = getelementptr float, ptr %44, i32 15
  store float %42, ptr %59, align 4
  call void @upmemrt_dpu_scatter(ptr %43, ptr %44, i64 4, i64 16, i64 0, i64 64, i64 16384, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %43, ptr @__constant_4096x256xf32, i64 4, i64 1048576, i64 256, i64 16384, i64 16448, ptr @scatter_map_9)
  call void @upmemrt_dpu_launch(ptr %43)
  %60 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  call void @upmemrt_dpu_gather(ptr %43, ptr %60, i64 4, i64 1048576, i64 256, i64 16384, i64 16448, ptr @scatter_map_9)
  call void @upmemrt_dpu_free(ptr %43)
  %61 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %61, ptr @dpu_program_19)
  call void @upmemrt_dpu_scatter(ptr %61, ptr %60, i64 4, i64 1048576, i64 256, i64 16384, i64 0, ptr @scatter_map_9)
  call void @upmemrt_dpu_scatter(ptr %61, ptr %1, i64 4, i64 1048576, i64 256, i64 16384, i64 16384, ptr @scatter_map_9)
  call void @upmemrt_dpu_scatter(ptr %61, ptr @__constant_4096x256xf32, i64 4, i64 1048576, i64 256, i64 16384, i64 32768, ptr @scatter_map_9)
  call void @upmemrt_dpu_launch(ptr %61)
  %62 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  call void @upmemrt_dpu_gather(ptr %61, ptr %62, i64 4, i64 1048576, i64 256, i64 16384, i64 32768, ptr @scatter_map_9)
  call void @upmemrt_dpu_free(ptr %61)
  ret ptr %62
}

define ptr @softmax_1048576(ptr %0) {
  %2 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %3 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  br label %4

4:                                                ; preds = %19, %1
  %5 = phi i64 [ %22, %19 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 1024
  br i1 %6, label %7, label %23

7:                                                ; preds = %4
  %8 = mul i64 %5, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %3, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %9

9:                                                ; preds = %12, %7
  %10 = phi i64 [ %18, %12 ], [ 0, %7 ]
  %11 = icmp slt i64 %10, 1024
  br i1 %11, label %12, label %19

12:                                               ; preds = %9
  %13 = add i64 %8, %10
  %14 = getelementptr float, ptr %0, i64 %13
  %15 = load float, ptr %14, align 4
  %16 = load float, ptr %3, align 4
  %17 = call float @llvm.maximum.f32(float %15, float %16)
  store float %17, ptr %3, align 4
  %18 = add i64 %10, 1
  br label %9

19:                                               ; preds = %9
  %20 = load float, ptr %3, align 4
  %21 = getelementptr float, ptr %2, i64 %5
  store float %20, ptr %21, align 4
  %22 = add i64 %5, 1
  br label %4

23:                                               ; preds = %4
  %24 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %24, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %25

25:                                               ; preds = %28, %23
  %26 = phi i64 [ %33, %28 ], [ 0, %23 ]
  %27 = icmp slt i64 %26, 1024
  br i1 %27, label %28, label %34

28:                                               ; preds = %25
  %29 = getelementptr float, ptr %2, i64 %26
  %30 = load float, ptr %29, align 4
  %31 = load float, ptr %24, align 4
  %32 = call float @llvm.maximum.f32(float %30, float %31)
  store float %32, ptr %24, align 4
  %33 = add i64 %26, 1
  br label %25

34:                                               ; preds = %25
  %35 = load float, ptr %24, align 4
  %36 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %36, ptr @dpu_program_20)
  call void @upmemrt_dpu_scatter(ptr %36, ptr %0, i64 4, i64 1048576, i64 256, i64 16384, i64 0, ptr @scatter_map_9)
  %37 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %35, ptr %37, align 4
  %38 = getelementptr float, ptr %37, i32 1
  store float %35, ptr %38, align 4
  %39 = getelementptr float, ptr %37, i32 2
  store float %35, ptr %39, align 4
  %40 = getelementptr float, ptr %37, i32 3
  store float %35, ptr %40, align 4
  %41 = getelementptr float, ptr %37, i32 4
  store float %35, ptr %41, align 4
  %42 = getelementptr float, ptr %37, i32 5
  store float %35, ptr %42, align 4
  %43 = getelementptr float, ptr %37, i32 6
  store float %35, ptr %43, align 4
  %44 = getelementptr float, ptr %37, i32 7
  store float %35, ptr %44, align 4
  %45 = getelementptr float, ptr %37, i32 8
  store float %35, ptr %45, align 4
  %46 = getelementptr float, ptr %37, i32 9
  store float %35, ptr %46, align 4
  %47 = getelementptr float, ptr %37, i32 10
  store float %35, ptr %47, align 4
  %48 = getelementptr float, ptr %37, i32 11
  store float %35, ptr %48, align 4
  %49 = getelementptr float, ptr %37, i32 12
  store float %35, ptr %49, align 4
  %50 = getelementptr float, ptr %37, i32 13
  store float %35, ptr %50, align 4
  %51 = getelementptr float, ptr %37, i32 14
  store float %35, ptr %51, align 4
  %52 = getelementptr float, ptr %37, i32 15
  store float %35, ptr %52, align 4
  call void @upmemrt_dpu_scatter(ptr %36, ptr %37, i64 4, i64 16, i64 0, i64 64, i64 16384, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %36, ptr @__constant_4096x256xf32, i64 4, i64 1048576, i64 256, i64 16384, i64 16448, ptr @scatter_map_9)
  call void @upmemrt_dpu_launch(ptr %36)
  %53 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  call void @upmemrt_dpu_gather(ptr %36, ptr %53, i64 4, i64 1048576, i64 256, i64 16384, i64 16448, ptr @scatter_map_9)
  call void @upmemrt_dpu_free(ptr %36)
  %54 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  br label %55

55:                                               ; preds = %58, %34
  %56 = phi i64 [ %63, %58 ], [ 0, %34 ]
  %57 = icmp slt i64 %56, 1048576
  br i1 %57, label %58, label %64

58:                                               ; preds = %55
  %59 = getelementptr float, ptr %53, i64 %56
  %60 = load float, ptr %59, align 4
  %61 = call float @llvm.exp.f32(float %60)
  %62 = getelementptr float, ptr %54, i64 %56
  store float %61, ptr %62, align 4
  %63 = add i64 %56, 1
  br label %55

64:                                               ; preds = %55
  %65 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %66 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  br label %67

67:                                               ; preds = %82, %64
  %68 = phi i64 [ %85, %82 ], [ 0, %64 ]
  %69 = icmp slt i64 %68, 1024
  br i1 %69, label %70, label %86

70:                                               ; preds = %67
  %71 = mul i64 %68, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %66, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %72

72:                                               ; preds = %75, %70
  %73 = phi i64 [ %81, %75 ], [ 0, %70 ]
  %74 = icmp slt i64 %73, 1024
  br i1 %74, label %75, label %82

75:                                               ; preds = %72
  %76 = add i64 %71, %73
  %77 = getelementptr float, ptr %54, i64 %76
  %78 = load float, ptr %77, align 4
  %79 = load float, ptr %66, align 4
  %80 = fadd float %78, %79
  store float %80, ptr %66, align 4
  %81 = add i64 %73, 1
  br label %72

82:                                               ; preds = %72
  %83 = load float, ptr %66, align 4
  %84 = getelementptr float, ptr %65, i64 %68
  store float %83, ptr %84, align 4
  %85 = add i64 %68, 1
  br label %67

86:                                               ; preds = %67
  %87 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %87, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %88

88:                                               ; preds = %91, %86
  %89 = phi i64 [ %96, %91 ], [ 0, %86 ]
  %90 = icmp slt i64 %89, 1024
  br i1 %90, label %91, label %97

91:                                               ; preds = %88
  %92 = getelementptr float, ptr %65, i64 %89
  %93 = load float, ptr %92, align 4
  %94 = load float, ptr %87, align 4
  %95 = fadd float %93, %94
  store float %95, ptr %87, align 4
  %96 = add i64 %89, 1
  br label %88

97:                                               ; preds = %88
  %98 = load float, ptr %87, align 4
  %99 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %99, ptr @dpu_program_21)
  call void @upmemrt_dpu_scatter(ptr %99, ptr %54, i64 4, i64 1048576, i64 256, i64 16384, i64 0, ptr @scatter_map_9)
  %100 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %98, ptr %100, align 4
  %101 = getelementptr float, ptr %100, i32 1
  store float %98, ptr %101, align 4
  %102 = getelementptr float, ptr %100, i32 2
  store float %98, ptr %102, align 4
  %103 = getelementptr float, ptr %100, i32 3
  store float %98, ptr %103, align 4
  %104 = getelementptr float, ptr %100, i32 4
  store float %98, ptr %104, align 4
  %105 = getelementptr float, ptr %100, i32 5
  store float %98, ptr %105, align 4
  %106 = getelementptr float, ptr %100, i32 6
  store float %98, ptr %106, align 4
  %107 = getelementptr float, ptr %100, i32 7
  store float %98, ptr %107, align 4
  %108 = getelementptr float, ptr %100, i32 8
  store float %98, ptr %108, align 4
  %109 = getelementptr float, ptr %100, i32 9
  store float %98, ptr %109, align 4
  %110 = getelementptr float, ptr %100, i32 10
  store float %98, ptr %110, align 4
  %111 = getelementptr float, ptr %100, i32 11
  store float %98, ptr %111, align 4
  %112 = getelementptr float, ptr %100, i32 12
  store float %98, ptr %112, align 4
  %113 = getelementptr float, ptr %100, i32 13
  store float %98, ptr %113, align 4
  %114 = getelementptr float, ptr %100, i32 14
  store float %98, ptr %114, align 4
  %115 = getelementptr float, ptr %100, i32 15
  store float %98, ptr %115, align 4
  call void @upmemrt_dpu_scatter(ptr %99, ptr %100, i64 4, i64 16, i64 0, i64 64, i64 16384, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %99, ptr @__constant_4096x256xf32, i64 4, i64 1048576, i64 256, i64 16384, i64 16448, ptr @scatter_map_9)
  call void @upmemrt_dpu_launch(ptr %99)
  %116 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  call void @upmemrt_dpu_gather(ptr %99, ptr %116, i64 4, i64 1048576, i64 256, i64 16384, i64 16448, ptr @scatter_map_9)
  call void @upmemrt_dpu_free(ptr %99)
  ret ptr %116
}

define ptr @va_1048576(ptr %0, ptr %1) {
  %3 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %3, ptr @dpu_program_22)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 1048576, i64 256, i64 16384, i64 0, ptr @scatter_map_9)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %1, i64 4, i64 1048576, i64 256, i64 16384, i64 16384, ptr @scatter_map_9)
  call void @upmemrt_dpu_scatter(ptr %3, ptr @__constant_4096x256xf32, i64 4, i64 1048576, i64 256, i64 16384, i64 32768, ptr @scatter_map_9)
  call void @upmemrt_dpu_launch(ptr %3)
  %4 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  call void @upmemrt_dpu_gather(ptr %3, ptr %4, i64 4, i64 1048576, i64 256, i64 16384, i64 32768, ptr @scatter_map_9)
  call void @upmemrt_dpu_free(ptr %3)
  ret ptr %4
}

define ptr @rmsnorm_262144(ptr %0, ptr %1) {
  %3 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %3, ptr @dpu_program_23)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 4096, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %3, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 8192, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %3)
  %4 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %3, ptr %4, i64 4, i64 262144, i64 64, i64 4096, i64 8192, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %3)
  %5 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  %6 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  br label %7

7:                                                ; preds = %22, %2
  %8 = phi i64 [ %25, %22 ], [ 0, %2 ]
  %9 = icmp slt i64 %8, 256
  br i1 %9, label %10, label %26

10:                                               ; preds = %7
  %11 = mul i64 %8, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %12

12:                                               ; preds = %15, %10
  %13 = phi i64 [ %21, %15 ], [ 0, %10 ]
  %14 = icmp slt i64 %13, 1024
  br i1 %14, label %15, label %22

15:                                               ; preds = %12
  %16 = add i64 %11, %13
  %17 = getelementptr float, ptr %4, i64 %16
  %18 = load float, ptr %17, align 4
  %19 = load float, ptr %6, align 4
  %20 = fadd float %18, %19
  store float %20, ptr %6, align 4
  %21 = add i64 %13, 1
  br label %12

22:                                               ; preds = %12
  %23 = load float, ptr %6, align 4
  %24 = getelementptr float, ptr %5, i64 %8
  store float %23, ptr %24, align 4
  %25 = add i64 %8, 1
  br label %7

26:                                               ; preds = %7
  %27 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %27, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %28

28:                                               ; preds = %31, %26
  %29 = phi i64 [ %36, %31 ], [ 0, %26 ]
  %30 = icmp slt i64 %29, 256
  br i1 %30, label %31, label %37

31:                                               ; preds = %28
  %32 = getelementptr float, ptr %5, i64 %29
  %33 = load float, ptr %32, align 4
  %34 = load float, ptr %27, align 4
  %35 = fadd float %33, %34
  store float %35, ptr %27, align 4
  %36 = add i64 %29, 1
  br label %28

37:                                               ; preds = %28
  %38 = load float, ptr %27, align 4
  %39 = fdiv float %38, 2.621440e+05
  %40 = fadd float %39, 0x3EE4F8B580000000
  %41 = call float @llvm.sqrt.f32(float %40)
  %42 = fdiv float 1.000000e+00, %41
  %43 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %43, ptr @dpu_program_24)
  call void @upmemrt_dpu_scatter(ptr %43, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  %44 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %42, ptr %44, align 4
  %45 = getelementptr float, ptr %44, i32 1
  store float %42, ptr %45, align 4
  %46 = getelementptr float, ptr %44, i32 2
  store float %42, ptr %46, align 4
  %47 = getelementptr float, ptr %44, i32 3
  store float %42, ptr %47, align 4
  %48 = getelementptr float, ptr %44, i32 4
  store float %42, ptr %48, align 4
  %49 = getelementptr float, ptr %44, i32 5
  store float %42, ptr %49, align 4
  %50 = getelementptr float, ptr %44, i32 6
  store float %42, ptr %50, align 4
  %51 = getelementptr float, ptr %44, i32 7
  store float %42, ptr %51, align 4
  %52 = getelementptr float, ptr %44, i32 8
  store float %42, ptr %52, align 4
  %53 = getelementptr float, ptr %44, i32 9
  store float %42, ptr %53, align 4
  %54 = getelementptr float, ptr %44, i32 10
  store float %42, ptr %54, align 4
  %55 = getelementptr float, ptr %44, i32 11
  store float %42, ptr %55, align 4
  %56 = getelementptr float, ptr %44, i32 12
  store float %42, ptr %56, align 4
  %57 = getelementptr float, ptr %44, i32 13
  store float %42, ptr %57, align 4
  %58 = getelementptr float, ptr %44, i32 14
  store float %42, ptr %58, align 4
  %59 = getelementptr float, ptr %44, i32 15
  store float %42, ptr %59, align 4
  call void @upmemrt_dpu_scatter(ptr %43, ptr %44, i64 4, i64 16, i64 0, i64 64, i64 4096, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %43, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %43)
  %60 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %43, ptr %60, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %43)
  %61 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %61, ptr @dpu_program_25)
  call void @upmemrt_dpu_scatter(ptr %61, ptr %60, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %61, ptr %1, i64 4, i64 262144, i64 64, i64 4096, i64 4096, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %61, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 8192, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %61)
  %62 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %61, ptr %62, i64 4, i64 262144, i64 64, i64 4096, i64 8192, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %61)
  ret ptr %62
}

define ptr @softmax_262144(ptr %0) {
  %2 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  %3 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  br label %4

4:                                                ; preds = %19, %1
  %5 = phi i64 [ %22, %19 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 256
  br i1 %6, label %7, label %23

7:                                                ; preds = %4
  %8 = mul i64 %5, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %3, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %9

9:                                                ; preds = %12, %7
  %10 = phi i64 [ %18, %12 ], [ 0, %7 ]
  %11 = icmp slt i64 %10, 1024
  br i1 %11, label %12, label %19

12:                                               ; preds = %9
  %13 = add i64 %8, %10
  %14 = getelementptr float, ptr %0, i64 %13
  %15 = load float, ptr %14, align 4
  %16 = load float, ptr %3, align 4
  %17 = call float @llvm.maximum.f32(float %15, float %16)
  store float %17, ptr %3, align 4
  %18 = add i64 %10, 1
  br label %9

19:                                               ; preds = %9
  %20 = load float, ptr %3, align 4
  %21 = getelementptr float, ptr %2, i64 %5
  store float %20, ptr %21, align 4
  %22 = add i64 %5, 1
  br label %4

23:                                               ; preds = %4
  %24 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %24, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %25

25:                                               ; preds = %28, %23
  %26 = phi i64 [ %33, %28 ], [ 0, %23 ]
  %27 = icmp slt i64 %26, 256
  br i1 %27, label %28, label %34

28:                                               ; preds = %25
  %29 = getelementptr float, ptr %2, i64 %26
  %30 = load float, ptr %29, align 4
  %31 = load float, ptr %24, align 4
  %32 = call float @llvm.maximum.f32(float %30, float %31)
  store float %32, ptr %24, align 4
  %33 = add i64 %26, 1
  br label %25

34:                                               ; preds = %25
  %35 = load float, ptr %24, align 4
  %36 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %36, ptr @dpu_program_26)
  call void @upmemrt_dpu_scatter(ptr %36, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  %37 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %35, ptr %37, align 4
  %38 = getelementptr float, ptr %37, i32 1
  store float %35, ptr %38, align 4
  %39 = getelementptr float, ptr %37, i32 2
  store float %35, ptr %39, align 4
  %40 = getelementptr float, ptr %37, i32 3
  store float %35, ptr %40, align 4
  %41 = getelementptr float, ptr %37, i32 4
  store float %35, ptr %41, align 4
  %42 = getelementptr float, ptr %37, i32 5
  store float %35, ptr %42, align 4
  %43 = getelementptr float, ptr %37, i32 6
  store float %35, ptr %43, align 4
  %44 = getelementptr float, ptr %37, i32 7
  store float %35, ptr %44, align 4
  %45 = getelementptr float, ptr %37, i32 8
  store float %35, ptr %45, align 4
  %46 = getelementptr float, ptr %37, i32 9
  store float %35, ptr %46, align 4
  %47 = getelementptr float, ptr %37, i32 10
  store float %35, ptr %47, align 4
  %48 = getelementptr float, ptr %37, i32 11
  store float %35, ptr %48, align 4
  %49 = getelementptr float, ptr %37, i32 12
  store float %35, ptr %49, align 4
  %50 = getelementptr float, ptr %37, i32 13
  store float %35, ptr %50, align 4
  %51 = getelementptr float, ptr %37, i32 14
  store float %35, ptr %51, align 4
  %52 = getelementptr float, ptr %37, i32 15
  store float %35, ptr %52, align 4
  call void @upmemrt_dpu_scatter(ptr %36, ptr %37, i64 4, i64 16, i64 0, i64 64, i64 4096, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %36, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %36)
  %53 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %36, ptr %53, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %36)
  %54 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  br label %55

55:                                               ; preds = %58, %34
  %56 = phi i64 [ %63, %58 ], [ 0, %34 ]
  %57 = icmp slt i64 %56, 262144
  br i1 %57, label %58, label %64

58:                                               ; preds = %55
  %59 = getelementptr float, ptr %53, i64 %56
  %60 = load float, ptr %59, align 4
  %61 = call float @llvm.exp.f32(float %60)
  %62 = getelementptr float, ptr %54, i64 %56
  store float %61, ptr %62, align 4
  %63 = add i64 %56, 1
  br label %55

64:                                               ; preds = %55
  %65 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64))
  %66 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  br label %67

67:                                               ; preds = %82, %64
  %68 = phi i64 [ %85, %82 ], [ 0, %64 ]
  %69 = icmp slt i64 %68, 256
  br i1 %69, label %70, label %86

70:                                               ; preds = %67
  %71 = mul i64 %68, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %66, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %72

72:                                               ; preds = %75, %70
  %73 = phi i64 [ %81, %75 ], [ 0, %70 ]
  %74 = icmp slt i64 %73, 1024
  br i1 %74, label %75, label %82

75:                                               ; preds = %72
  %76 = add i64 %71, %73
  %77 = getelementptr float, ptr %54, i64 %76
  %78 = load float, ptr %77, align 4
  %79 = load float, ptr %66, align 4
  %80 = fadd float %78, %79
  store float %80, ptr %66, align 4
  %81 = add i64 %73, 1
  br label %72

82:                                               ; preds = %72
  %83 = load float, ptr %66, align 4
  %84 = getelementptr float, ptr %65, i64 %68
  store float %83, ptr %84, align 4
  %85 = add i64 %68, 1
  br label %67

86:                                               ; preds = %67
  %87 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  call void @llvm.memcpy.p0.p0.i64(ptr %87, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %88

88:                                               ; preds = %91, %86
  %89 = phi i64 [ %96, %91 ], [ 0, %86 ]
  %90 = icmp slt i64 %89, 256
  br i1 %90, label %91, label %97

91:                                               ; preds = %88
  %92 = getelementptr float, ptr %65, i64 %89
  %93 = load float, ptr %92, align 4
  %94 = load float, ptr %87, align 4
  %95 = fadd float %93, %94
  store float %95, ptr %87, align 4
  %96 = add i64 %89, 1
  br label %88

97:                                               ; preds = %88
  %98 = load float, ptr %87, align 4
  %99 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %99, ptr @dpu_program_27)
  call void @upmemrt_dpu_scatter(ptr %99, ptr %54, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  %100 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %98, ptr %100, align 4
  %101 = getelementptr float, ptr %100, i32 1
  store float %98, ptr %101, align 4
  %102 = getelementptr float, ptr %100, i32 2
  store float %98, ptr %102, align 4
  %103 = getelementptr float, ptr %100, i32 3
  store float %98, ptr %103, align 4
  %104 = getelementptr float, ptr %100, i32 4
  store float %98, ptr %104, align 4
  %105 = getelementptr float, ptr %100, i32 5
  store float %98, ptr %105, align 4
  %106 = getelementptr float, ptr %100, i32 6
  store float %98, ptr %106, align 4
  %107 = getelementptr float, ptr %100, i32 7
  store float %98, ptr %107, align 4
  %108 = getelementptr float, ptr %100, i32 8
  store float %98, ptr %108, align 4
  %109 = getelementptr float, ptr %100, i32 9
  store float %98, ptr %109, align 4
  %110 = getelementptr float, ptr %100, i32 10
  store float %98, ptr %110, align 4
  %111 = getelementptr float, ptr %100, i32 11
  store float %98, ptr %111, align 4
  %112 = getelementptr float, ptr %100, i32 12
  store float %98, ptr %112, align 4
  %113 = getelementptr float, ptr %100, i32 13
  store float %98, ptr %113, align 4
  %114 = getelementptr float, ptr %100, i32 14
  store float %98, ptr %114, align 4
  %115 = getelementptr float, ptr %100, i32 15
  store float %98, ptr %115, align 4
  call void @upmemrt_dpu_scatter(ptr %99, ptr %100, i64 4, i64 16, i64 0, i64 64, i64 4096, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %99, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %99)
  %116 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %99, ptr %116, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %99)
  ret ptr %116
}

define ptr @rmsnorm_262144_opt(ptr %0, ptr %1) {
  %3 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %3, ptr @dpu_program_28)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 4096, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %3)
  %4 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8192) to i64))
  call void @upmemrt_dpu_gather(ptr %3, ptr %4, i64 4, i64 8192, i64 2, i64 128, i64 8192, ptr @scatter_map_11)
  call void @upmemrt_dpu_free(ptr %3)
  br label %5

5:                                                ; preds = %9, %2
  %6 = phi i64 [ %19, %9 ], [ 0, %2 ]
  %7 = phi float [ %18, %9 ], [ 0.000000e+00, %2 ]
  %8 = icmp slt i64 %6, 4096
  br i1 %8, label %9, label %20

9:                                                ; preds = %5
  %10 = mul i64 %6, 2
  %11 = add i64 %10, 0
  %12 = getelementptr float, ptr %4, i64 %11
  %13 = load float, ptr %12, align 4
  %14 = add i64 %10, 1
  %15 = getelementptr float, ptr %4, i64 %14
  %16 = load float, ptr %15, align 4
  %17 = fadd float %13, %16
  %18 = fadd float %17, %7
  %19 = add i64 %6, 1
  br label %5

20:                                               ; preds = %5
  %21 = fdiv float %7, 2.621440e+05
  %22 = fadd float %21, 0x3EE4F8B580000000
  %23 = call float @llvm.sqrt.f32(float %22)
  %24 = fdiv float 1.000000e+00, %23
  %25 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %24, ptr %25, align 4
  %26 = getelementptr float, ptr %25, i32 1
  store float %24, ptr %26, align 4
  %27 = getelementptr float, ptr %25, i32 2
  store float %24, ptr %27, align 4
  %28 = getelementptr float, ptr %25, i32 3
  store float %24, ptr %28, align 4
  %29 = getelementptr float, ptr %25, i32 4
  store float %24, ptr %29, align 4
  %30 = getelementptr float, ptr %25, i32 5
  store float %24, ptr %30, align 4
  %31 = getelementptr float, ptr %25, i32 6
  store float %24, ptr %31, align 4
  %32 = getelementptr float, ptr %25, i32 7
  store float %24, ptr %32, align 4
  %33 = getelementptr float, ptr %25, i32 8
  store float %24, ptr %33, align 4
  %34 = getelementptr float, ptr %25, i32 9
  store float %24, ptr %34, align 4
  %35 = getelementptr float, ptr %25, i32 10
  store float %24, ptr %35, align 4
  %36 = getelementptr float, ptr %25, i32 11
  store float %24, ptr %36, align 4
  %37 = getelementptr float, ptr %25, i32 12
  store float %24, ptr %37, align 4
  %38 = getelementptr float, ptr %25, i32 13
  store float %24, ptr %38, align 4
  %39 = getelementptr float, ptr %25, i32 14
  store float %24, ptr %39, align 4
  %40 = getelementptr float, ptr %25, i32 15
  store float %24, ptr %40, align 4
  %41 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %41, ptr @dpu_program_29)
  call void @upmemrt_dpu_scatter(ptr %41, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %41, ptr %1, i64 4, i64 262144, i64 64, i64 4096, i64 4096, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %41, ptr %25, i64 4, i64 16, i64 0, i64 64, i64 8192, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %41, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 8256, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %41)
  %42 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %41, ptr %42, i64 4, i64 262144, i64 64, i64 4096, i64 8256, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %41)
  ret ptr %42
}

define ptr @softmax_262144_opt(ptr %0) {
  br label %2

2:                                                ; preds = %6, %1
  %3 = phi i64 [ %10, %6 ], [ 0, %1 ]
  %4 = phi float [ %9, %6 ], [ 0xFFF0000000000000, %1 ]
  %5 = icmp slt i64 %3, 262144
  br i1 %5, label %6, label %11

6:                                                ; preds = %2
  %7 = getelementptr float, ptr %0, i64 %3
  %8 = load float, ptr %7, align 4
  %9 = call float @llvm.maximum.f32(float %8, float %4)
  %10 = add i64 %3, 1
  br label %2

11:                                               ; preds = %2
  %12 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %4, ptr %12, align 4
  %13 = getelementptr float, ptr %12, i32 1
  store float %4, ptr %13, align 4
  %14 = getelementptr float, ptr %12, i32 2
  store float %4, ptr %14, align 4
  %15 = getelementptr float, ptr %12, i32 3
  store float %4, ptr %15, align 4
  %16 = getelementptr float, ptr %12, i32 4
  store float %4, ptr %16, align 4
  %17 = getelementptr float, ptr %12, i32 5
  store float %4, ptr %17, align 4
  %18 = getelementptr float, ptr %12, i32 6
  store float %4, ptr %18, align 4
  %19 = getelementptr float, ptr %12, i32 7
  store float %4, ptr %19, align 4
  %20 = getelementptr float, ptr %12, i32 8
  store float %4, ptr %20, align 4
  %21 = getelementptr float, ptr %12, i32 9
  store float %4, ptr %21, align 4
  %22 = getelementptr float, ptr %12, i32 10
  store float %4, ptr %22, align 4
  %23 = getelementptr float, ptr %12, i32 11
  store float %4, ptr %23, align 4
  %24 = getelementptr float, ptr %12, i32 12
  store float %4, ptr %24, align 4
  %25 = getelementptr float, ptr %12, i32 13
  store float %4, ptr %25, align 4
  %26 = getelementptr float, ptr %12, i32 14
  store float %4, ptr %26, align 4
  %27 = getelementptr float, ptr %12, i32 15
  store float %4, ptr %27, align 4
  %28 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %28, ptr @dpu_program_30)
  call void @upmemrt_dpu_scatter(ptr %28, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %28, ptr %12, i64 4, i64 16, i64 0, i64 64, i64 4096, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %28, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %28, ptr @__constant_4096x2xf32, i64 4, i64 8192, i64 2, i64 128, i64 8256, ptr @scatter_map_11)
  call void @upmemrt_dpu_launch(ptr %28)
  %29 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8192) to i64))
  call void @upmemrt_dpu_gather(ptr %28, ptr %29, i64 4, i64 8192, i64 2, i64 128, i64 8256, ptr @scatter_map_11)
  br label %30

30:                                               ; preds = %34, %11
  %31 = phi i64 [ %44, %34 ], [ 0, %11 ]
  %32 = phi float [ %43, %34 ], [ 0.000000e+00, %11 ]
  %33 = icmp slt i64 %31, 4096
  br i1 %33, label %34, label %45

34:                                               ; preds = %30
  %35 = mul i64 %31, 2
  %36 = add i64 %35, 0
  %37 = getelementptr float, ptr %29, i64 %36
  %38 = load float, ptr %37, align 4
  %39 = add i64 %35, 1
  %40 = getelementptr float, ptr %29, i64 %39
  %41 = load float, ptr %40, align 4
  %42 = fadd float %38, %41
  %43 = fadd float %42, %32
  %44 = add i64 %31, 1
  br label %30

45:                                               ; preds = %30
  %46 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  store float %32, ptr %46, align 4
  %47 = getelementptr float, ptr %46, i32 1
  store float %32, ptr %47, align 4
  %48 = getelementptr float, ptr %46, i32 2
  store float %32, ptr %48, align 4
  %49 = getelementptr float, ptr %46, i32 3
  store float %32, ptr %49, align 4
  %50 = getelementptr float, ptr %46, i32 4
  store float %32, ptr %50, align 4
  %51 = getelementptr float, ptr %46, i32 5
  store float %32, ptr %51, align 4
  %52 = getelementptr float, ptr %46, i32 6
  store float %32, ptr %52, align 4
  %53 = getelementptr float, ptr %46, i32 7
  store float %32, ptr %53, align 4
  %54 = getelementptr float, ptr %46, i32 8
  store float %32, ptr %54, align 4
  %55 = getelementptr float, ptr %46, i32 9
  store float %32, ptr %55, align 4
  %56 = getelementptr float, ptr %46, i32 10
  store float %32, ptr %56, align 4
  %57 = getelementptr float, ptr %46, i32 11
  store float %32, ptr %57, align 4
  %58 = getelementptr float, ptr %46, i32 12
  store float %32, ptr %58, align 4
  %59 = getelementptr float, ptr %46, i32 13
  store float %32, ptr %59, align 4
  %60 = getelementptr float, ptr %46, i32 14
  store float %32, ptr %60, align 4
  %61 = getelementptr float, ptr %46, i32 15
  store float %32, ptr %61, align 4
  %62 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %28, ptr %62, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %28)
  %63 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %63, ptr @dpu_program_31)
  call void @upmemrt_dpu_scatter(ptr %63, ptr %62, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %63, ptr %46, i64 4, i64 16, i64 0, i64 64, i64 4096, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %63, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %63)
  %64 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %63, ptr %64, i64 4, i64 262144, i64 64, i64 4096, i64 4160, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %63)
  ret ptr %64
}

define ptr @va_262144(ptr %0, ptr %1) {
  %3 = call ptr @upmemrt_dpu_alloc(i32 4, i32 64)
  call void @upmemrt_dpu_load(ptr %3, ptr @dpu_program_32)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 262144, i64 64, i64 4096, i64 0, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %1, i64 4, i64 262144, i64 64, i64 4096, i64 4096, ptr @scatter_map_10)
  call void @upmemrt_dpu_scatter(ptr %3, ptr @__constant_4096x64xf32, i64 4, i64 262144, i64 64, i64 4096, i64 8192, ptr @scatter_map_10)
  call void @upmemrt_dpu_launch(ptr %3)
  %4 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 262144) to i64))
  call void @upmemrt_dpu_gather(ptr %3, ptr %4, i64 4, i64 262144, i64 64, i64 4096, i64 8192, ptr @scatter_map_10)
  call void @upmemrt_dpu_free(ptr %3)
  ret ptr %4
}

define ptr @mha_big(ptr %0, ptr %1, ptr %2, i64 %3) {
  %5 = add i64 %3, 1
  %6 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  br label %7

7:                                                ; preds = %10, %4
  %8 = phi i64 [ %12, %10 ], [ 0, %4 ]
  %9 = icmp slt i64 %8, 1024
  br i1 %9, label %10, label %13

10:                                               ; preds = %7
  %11 = getelementptr float, ptr %6, i64 %8
  store float 0xFFF0000000000000, ptr %11, align 4
  %12 = add i64 %8, 1
  br label %7

13:                                               ; preds = %7
  %14 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64))
  %15 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64))
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %15, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 0, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 32768, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %15, ptr %14, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 32768), i1 false)
  %21 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %21, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, ptr %21, 1
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 0, 2
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 1024, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 1, 4, 0
  %27 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4096) to i64))
  %28 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4096) to i64))
  %29 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4096) to i64))
  %30 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4) to i64))
  %31 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %32 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %33 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %34 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %35 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %36 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  %37 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %38 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %39 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %40 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %41 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %42 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  %43 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %44 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4096) to i64))
  %45 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4096) to i64))
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %45, 0
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, ptr %45, 1
  %48 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, i64 0, 2
  %49 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, i64 4096, 3, 0
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %49, i64 1, 4, 0
  %51 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4096) to i64))
  %52 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64))
  %53 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4096) to i64))
  br label %54

54:                                               ; preds = %233, %13
  %55 = phi i64 [ %253, %233 ], [ 0, %13 ]
  %56 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %239, %233 ], [ %20, %13 ]
  %57 = icmp slt i64 %55, 8
  br i1 %57, label %58, label %254

58:                                               ; preds = %54
  %59 = mul i64 %55, 4096
  call void @llvm.memcpy.p0.p0.i64(ptr %21, ptr %6, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 1024), i1 false)
  br label %60

60:                                               ; preds = %99, %58
  %61 = phi i64 [ %115, %99 ], [ 0, %58 ]
  %62 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %107, %99 ], [ %26, %58 ]
  %63 = icmp slt i64 %61, %5
  br i1 %63, label %64, label %116

64:                                               ; preds = %60
  %65 = mul i64 %61, 32768
  %66 = add i64 %65, %59
  %67 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  call void @upmemrt_dpu_load(ptr %67, ptr @dpu_program_33)
  %68 = getelementptr float, ptr %0, i64 %59
  call void @llvm.memcpy.p0.p0.i64(ptr %27, ptr %68, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 4096), i1 false)
  call void @upmemrt_dpu_scatter(ptr %67, ptr %27, i64 4, i64 4096, i64 16, i64 1024, i64 0, ptr @scatter_map_12)
  %69 = getelementptr float, ptr %1, i64 %66
  call void @llvm.memcpy.p0.p0.i64(ptr %28, ptr %69, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 4096), i1 false)
  call void @upmemrt_dpu_scatter(ptr %67, ptr %28, i64 4, i64 4096, i64 16, i64 1024, i64 1024, ptr @scatter_map_12)
  call void @upmemrt_dpu_scatter(ptr %67, ptr @__constant_256x16xf32, i64 4, i64 4096, i64 16, i64 1024, i64 2048, ptr @scatter_map_12)
  call void @upmemrt_dpu_launch(ptr %67)
  call void @upmemrt_dpu_gather(ptr %67, ptr %29, i64 4, i64 4096, i64 16, i64 1024, i64 2048, ptr @scatter_map_12)
  call void @upmemrt_dpu_free(ptr %67)
  br label %70

70:                                               ; preds = %85, %64
  %71 = phi i64 [ %88, %85 ], [ 0, %64 ]
  %72 = icmp slt i64 %71, 4
  br i1 %72, label %73, label %89

73:                                               ; preds = %70
  %74 = mul i64 %71, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %31, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %75

75:                                               ; preds = %78, %73
  %76 = phi i64 [ %84, %78 ], [ 0, %73 ]
  %77 = icmp slt i64 %76, 1024
  br i1 %77, label %78, label %85

78:                                               ; preds = %75
  %79 = add i64 %74, %76
  %80 = getelementptr float, ptr %29, i64 %79
  %81 = load float, ptr %80, align 4
  %82 = load float, ptr %31, align 4
  %83 = fadd float %81, %82
  store float %83, ptr %31, align 4
  %84 = add i64 %76, 1
  br label %75

85:                                               ; preds = %75
  %86 = load float, ptr %31, align 4
  %87 = getelementptr float, ptr %30, i64 %71
  store float %86, ptr %87, align 4
  %88 = add i64 %71, 1
  br label %70

89:                                               ; preds = %70
  call void @llvm.memcpy.p0.p0.i64(ptr %32, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %90

90:                                               ; preds = %93, %89
  %91 = phi i64 [ %98, %93 ], [ 0, %89 ]
  %92 = icmp slt i64 %91, 4
  br i1 %92, label %93, label %99

93:                                               ; preds = %90
  %94 = getelementptr float, ptr %30, i64 %91
  %95 = load float, ptr %94, align 4
  %96 = load float, ptr %32, align 4
  %97 = fadd float %95, %96
  store float %97, ptr %32, align 4
  %98 = add i64 %91, 1
  br label %90

99:                                               ; preds = %90
  %100 = load float, ptr %32, align 4
  %101 = fdiv float %100, 0x401BB67AE0000000
  %102 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %103 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %102, 0
  %104 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %103, ptr %102, 1
  %105 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %104, i64 0, 2
  %106 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %105, i64 1024, 3, 0
  %107 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %106, i64 1, 4, 0
  %108 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 3, 0
  %109 = mul i64 %108, 1
  %110 = mul i64 %109, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %111 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 1
  %112 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 2
  %113 = getelementptr float, ptr %111, i64 %112
  call void @llvm.memcpy.p0.p0.i64(ptr %102, ptr %113, i64 %110, i1 false)
  %114 = getelementptr float, ptr %102, i64 %61
  store float %101, ptr %114, align 4
  %115 = add i64 %61, 1
  br label %60

116:                                              ; preds = %60
  call void @llvm.memcpy.p0.p0.i64(ptr %34, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %117

117:                                              ; preds = %120, %116
  %118 = phi i64 [ %126, %120 ], [ 0, %116 ]
  %119 = icmp slt i64 %118, 1024
  br i1 %119, label %120, label %127

120:                                              ; preds = %117
  %121 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 1
  %122 = getelementptr float, ptr %121, i64 %118
  %123 = load float, ptr %122, align 4
  %124 = load float, ptr %34, align 4
  %125 = call float @llvm.maximum.f32(float %123, float %124)
  store float %125, ptr %34, align 4
  %126 = add i64 %118, 1
  br label %117

127:                                              ; preds = %117
  %128 = load float, ptr %34, align 4
  store float %128, ptr %33, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %35, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %129 = load float, ptr %33, align 4
  %130 = load float, ptr %35, align 4
  %131 = call float @llvm.maximum.f32(float %129, float %130)
  store float %131, ptr %35, align 4
  %132 = load float, ptr %35, align 4
  %133 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  call void @upmemrt_dpu_load(ptr %133, ptr @dpu_program_34)
  %134 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 1
  call void @upmemrt_dpu_scatter(ptr %133, ptr %134, i64 4, i64 1024, i64 4, i64 256, i64 0, ptr @scatter_map_13)
  store float %132, ptr %36, align 4
  %135 = getelementptr float, ptr %36, i32 1
  store float %132, ptr %135, align 4
  %136 = getelementptr float, ptr %36, i32 2
  store float %132, ptr %136, align 4
  %137 = getelementptr float, ptr %36, i32 3
  store float %132, ptr %137, align 4
  %138 = getelementptr float, ptr %36, i32 4
  store float %132, ptr %138, align 4
  %139 = getelementptr float, ptr %36, i32 5
  store float %132, ptr %139, align 4
  %140 = getelementptr float, ptr %36, i32 6
  store float %132, ptr %140, align 4
  %141 = getelementptr float, ptr %36, i32 7
  store float %132, ptr %141, align 4
  %142 = getelementptr float, ptr %36, i32 8
  store float %132, ptr %142, align 4
  %143 = getelementptr float, ptr %36, i32 9
  store float %132, ptr %143, align 4
  %144 = getelementptr float, ptr %36, i32 10
  store float %132, ptr %144, align 4
  %145 = getelementptr float, ptr %36, i32 11
  store float %132, ptr %145, align 4
  %146 = getelementptr float, ptr %36, i32 12
  store float %132, ptr %146, align 4
  %147 = getelementptr float, ptr %36, i32 13
  store float %132, ptr %147, align 4
  %148 = getelementptr float, ptr %36, i32 14
  store float %132, ptr %148, align 4
  %149 = getelementptr float, ptr %36, i32 15
  store float %132, ptr %149, align 4
  call void @upmemrt_dpu_scatter(ptr %133, ptr %36, i64 4, i64 16, i64 0, i64 64, i64 256, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %133, ptr @__constant_256x4xf32, i64 4, i64 1024, i64 4, i64 256, i64 320, ptr @scatter_map_13)
  call void @upmemrt_dpu_launch(ptr %133)
  call void @upmemrt_dpu_gather(ptr %133, ptr %37, i64 4, i64 1024, i64 4, i64 256, i64 320, ptr @scatter_map_13)
  call void @upmemrt_dpu_free(ptr %133)
  br label %150

150:                                              ; preds = %153, %127
  %151 = phi i64 [ %158, %153 ], [ 0, %127 ]
  %152 = icmp slt i64 %151, 1024
  br i1 %152, label %153, label %159

153:                                              ; preds = %150
  %154 = getelementptr float, ptr %37, i64 %151
  %155 = load float, ptr %154, align 4
  %156 = call float @llvm.exp.f32(float %155)
  %157 = getelementptr float, ptr %38, i64 %151
  store float %156, ptr %157, align 4
  %158 = add i64 %151, 1
  br label %150

159:                                              ; preds = %150
  call void @llvm.memcpy.p0.p0.i64(ptr %40, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %160

160:                                              ; preds = %163, %159
  %161 = phi i64 [ %168, %163 ], [ 0, %159 ]
  %162 = icmp slt i64 %161, 1024
  br i1 %162, label %163, label %169

163:                                              ; preds = %160
  %164 = getelementptr float, ptr %38, i64 %161
  %165 = load float, ptr %164, align 4
  %166 = load float, ptr %40, align 4
  %167 = fadd float %165, %166
  store float %167, ptr %40, align 4
  %168 = add i64 %161, 1
  br label %160

169:                                              ; preds = %160
  %170 = load float, ptr %40, align 4
  store float %170, ptr %39, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %41, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %171 = load float, ptr %39, align 4
  %172 = load float, ptr %41, align 4
  %173 = fadd float %171, %172
  store float %173, ptr %41, align 4
  %174 = load float, ptr %41, align 4
  %175 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  call void @upmemrt_dpu_load(ptr %175, ptr @dpu_program_35)
  call void @upmemrt_dpu_scatter(ptr %175, ptr %38, i64 4, i64 1024, i64 4, i64 256, i64 0, ptr @scatter_map_13)
  store float %174, ptr %42, align 4
  %176 = getelementptr float, ptr %42, i32 1
  store float %174, ptr %176, align 4
  %177 = getelementptr float, ptr %42, i32 2
  store float %174, ptr %177, align 4
  %178 = getelementptr float, ptr %42, i32 3
  store float %174, ptr %178, align 4
  %179 = getelementptr float, ptr %42, i32 4
  store float %174, ptr %179, align 4
  %180 = getelementptr float, ptr %42, i32 5
  store float %174, ptr %180, align 4
  %181 = getelementptr float, ptr %42, i32 6
  store float %174, ptr %181, align 4
  %182 = getelementptr float, ptr %42, i32 7
  store float %174, ptr %182, align 4
  %183 = getelementptr float, ptr %42, i32 8
  store float %174, ptr %183, align 4
  %184 = getelementptr float, ptr %42, i32 9
  store float %174, ptr %184, align 4
  %185 = getelementptr float, ptr %42, i32 10
  store float %174, ptr %185, align 4
  %186 = getelementptr float, ptr %42, i32 11
  store float %174, ptr %186, align 4
  %187 = getelementptr float, ptr %42, i32 12
  store float %174, ptr %187, align 4
  %188 = getelementptr float, ptr %42, i32 13
  store float %174, ptr %188, align 4
  %189 = getelementptr float, ptr %42, i32 14
  store float %174, ptr %189, align 4
  %190 = getelementptr float, ptr %42, i32 15
  store float %174, ptr %190, align 4
  call void @upmemrt_dpu_scatter(ptr %175, ptr %42, i64 4, i64 16, i64 0, i64 64, i64 256, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %175, ptr @__constant_256x4xf32, i64 4, i64 1024, i64 4, i64 256, i64 320, ptr @scatter_map_13)
  call void @upmemrt_dpu_launch(ptr %175)
  call void @upmemrt_dpu_gather(ptr %175, ptr %43, i64 4, i64 1024, i64 4, i64 256, i64 320, ptr @scatter_map_13)
  call void @upmemrt_dpu_free(ptr %175)
  br label %191

191:                                              ; preds = %194, %169
  %192 = phi i64 [ %196, %194 ], [ 0, %169 ]
  %193 = icmp slt i64 %192, 4096
  br i1 %193, label %194, label %197

194:                                              ; preds = %191
  %195 = getelementptr float, ptr %44, i64 %192
  store float 0.000000e+00, ptr %195, align 4
  %196 = add i64 %192, 1
  br label %191

197:                                              ; preds = %191
  call void @llvm.memcpy.p0.p0.i64(ptr %45, ptr %44, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 4096), i1 false)
  br label %198

198:                                              ; preds = %202, %197
  %199 = phi i64 [ %232, %202 ], [ 0, %197 ]
  %200 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %231, %202 ], [ %50, %197 ]
  %201 = icmp slt i64 %199, %5
  br i1 %201, label %202, label %233

202:                                              ; preds = %198
  %203 = mul i64 %199, 32768
  %204 = add i64 %203, %59
  %205 = getelementptr float, ptr %43, i64 %199
  %206 = load float, ptr %205, align 4
  %207 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  call void @upmemrt_dpu_load(ptr %207, ptr @dpu_program_36)
  %208 = getelementptr float, ptr %2, i64 %204
  call void @llvm.memcpy.p0.p0.i64(ptr %51, ptr %208, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 4096), i1 false)
  call void @upmemrt_dpu_scatter(ptr %207, ptr %51, i64 4, i64 4096, i64 16, i64 1024, i64 0, ptr @scatter_map_12)
  store float %206, ptr %52, align 4
  %209 = getelementptr float, ptr %52, i32 1
  store float %206, ptr %209, align 4
  %210 = getelementptr float, ptr %52, i32 2
  store float %206, ptr %210, align 4
  %211 = getelementptr float, ptr %52, i32 3
  store float %206, ptr %211, align 4
  %212 = getelementptr float, ptr %52, i32 4
  store float %206, ptr %212, align 4
  %213 = getelementptr float, ptr %52, i32 5
  store float %206, ptr %213, align 4
  %214 = getelementptr float, ptr %52, i32 6
  store float %206, ptr %214, align 4
  %215 = getelementptr float, ptr %52, i32 7
  store float %206, ptr %215, align 4
  %216 = getelementptr float, ptr %52, i32 8
  store float %206, ptr %216, align 4
  %217 = getelementptr float, ptr %52, i32 9
  store float %206, ptr %217, align 4
  %218 = getelementptr float, ptr %52, i32 10
  store float %206, ptr %218, align 4
  %219 = getelementptr float, ptr %52, i32 11
  store float %206, ptr %219, align 4
  %220 = getelementptr float, ptr %52, i32 12
  store float %206, ptr %220, align 4
  %221 = getelementptr float, ptr %52, i32 13
  store float %206, ptr %221, align 4
  %222 = getelementptr float, ptr %52, i32 14
  store float %206, ptr %222, align 4
  %223 = getelementptr float, ptr %52, i32 15
  store float %206, ptr %223, align 4
  call void @upmemrt_dpu_scatter(ptr %207, ptr %52, i64 4, i64 16, i64 0, i64 64, i64 1024, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %207, ptr @__constant_256x16xf32, i64 4, i64 4096, i64 16, i64 1024, i64 1088, ptr @scatter_map_12)
  call void @upmemrt_dpu_launch(ptr %207)
  call void @upmemrt_dpu_gather(ptr %207, ptr %53, i64 4, i64 4096, i64 16, i64 1024, i64 1088, ptr @scatter_map_12)
  call void @upmemrt_dpu_free(ptr %207)
  %224 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  call void @upmemrt_dpu_load(ptr %224, ptr @dpu_program_37)
  %225 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %200, 1
  call void @upmemrt_dpu_scatter(ptr %224, ptr %225, i64 4, i64 4096, i64 16, i64 1024, i64 0, ptr @scatter_map_12)
  call void @upmemrt_dpu_scatter(ptr %224, ptr %53, i64 4, i64 4096, i64 16, i64 1024, i64 1024, ptr @scatter_map_12)
  call void @upmemrt_dpu_scatter(ptr %224, ptr @__constant_256x16xf32, i64 4, i64 4096, i64 16, i64 1024, i64 2048, ptr @scatter_map_12)
  call void @upmemrt_dpu_launch(ptr %224)
  %226 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4096) to i64))
  call void @upmemrt_dpu_gather(ptr %224, ptr %226, i64 4, i64 4096, i64 16, i64 1024, i64 2048, ptr @scatter_map_12)
  %227 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %226, 0
  %228 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %227, ptr %226, 1
  %229 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %228, i64 0, 2
  %230 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %229, i64 4096, 3, 0
  %231 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %230, i64 1, 4, 0
  call void @upmemrt_dpu_free(ptr %224)
  %232 = add i64 %199, 1
  br label %198

233:                                              ; preds = %198
  %234 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64))
  %235 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %234, 0
  %236 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %235, ptr %234, 1
  %237 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %236, i64 0, 2
  %238 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %237, i64 32768, 3, 0
  %239 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %238, i64 1, 4, 0
  %240 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 3, 0
  %241 = mul i64 %240, 1
  %242 = mul i64 %241, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %243 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 1
  %244 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 2
  %245 = getelementptr float, ptr %243, i64 %244
  call void @llvm.memcpy.p0.p0.i64(ptr %234, ptr %245, i64 %242, i1 false)
  %246 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %200, 3, 0
  %247 = mul i64 %246, 1
  %248 = mul i64 %247, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %249 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %200, 1
  %250 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %200, 2
  %251 = getelementptr float, ptr %249, i64 %250
  %252 = getelementptr float, ptr %234, i64 %59
  call void @llvm.memcpy.p0.p0.i64(ptr %252, ptr %251, i64 %248, i1 false)
  %253 = add i64 %55, 1
  br label %54

254:                                              ; preds = %54
  %255 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 0
  ret ptr %255
}

define void @test_0(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  %6 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  %7 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  call void @upmemrt_dpu_load(ptr %5, ptr @dpu_program_38)
  call void @upmemrt_dpu_load(ptr %6, ptr @dpu_program_39)
  call void @upmemrt_dpu_load(ptr %7, ptr @dpu_program_40)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 0, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %1, i64 4, i64 1024, i64 4, i64 256, i64 256, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %6, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 0, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %6, ptr %2, i64 4, i64 1024, i64 4, i64 256, i64 256, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %6, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %7, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 0, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %7, ptr %3, i64 4, i64 1024, i64 4, i64 256, i64 256, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %7, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  call void @upmemrt_dpu_launch(ptr %5)
  call void @upmemrt_dpu_launch(ptr %6)
  call void @upmemrt_dpu_launch(ptr %7)
  %8 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %5, ptr %8, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  %9 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %6, ptr %9, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  %10 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %7, ptr %10, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  call void @upmemrt_dpu_free(ptr %5)
  call void @upmemrt_dpu_free(ptr %6)
  call void @upmemrt_dpu_free(ptr %7)
  ret void
}

define void @test_1(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  call void @upmemrt_dpu_load(ptr %5, ptr @dpu_program_41)
  call void @upmemrt_dpu_load(ptr %5, ptr @dpu_program_42)
  call void @upmemrt_dpu_load(ptr %5, ptr @dpu_program_43)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 0, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %1, i64 4, i64 1024, i64 4, i64 256, i64 256, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 768, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %2, i64 4, i64 1024, i64 4, i64 256, i64 1024, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 1280, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 1536, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %3, i64 4, i64 1024, i64 4, i64 256, i64 1792, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 2048, ptr @scatter_map_13)
  call void @upmemrt_dpu_launch(ptr %5)
  call void @upmemrt_dpu_launch(ptr %5)
  call void @upmemrt_dpu_launch(ptr %5)
  %6 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %5, ptr %6, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  %7 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %5, ptr %7, i64 4, i64 1024, i64 4, i64 256, i64 1280, ptr @scatter_map_13)
  %8 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %5, ptr %8, i64 4, i64 1024, i64 4, i64 256, i64 2048, ptr @scatter_map_13)
  call void @upmemrt_dpu_free(ptr %5)
  ret void
}

define void @test_2(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = call ptr @upmemrt_dpu_alloc(i32 1, i32 16)
  call void @upmemrt_dpu_load(ptr %5, ptr @dpu_program_44)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 0, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %1, i64 4, i64 1024, i64 4, i64 256, i64 256, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 768, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %2, i64 4, i64 1024, i64 4, i64 256, i64 1024, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 1280, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %0, i64 4, i64 1024, i64 4, i64 256, i64 1536, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr %3, i64 4, i64 1024, i64 4, i64 256, i64 1792, ptr @scatter_map_13)
  call void @upmemrt_dpu_scatter(ptr %5, ptr @__tconstant_256x4xi32, i64 4, i64 1024, i64 4, i64 256, i64 2048, ptr @scatter_map_13)
  call void @upmemrt_dpu_launch(ptr %5)
  %6 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %5, ptr %6, i64 4, i64 1024, i64 4, i64 256, i64 512, ptr @scatter_map_13)
  %7 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %5, ptr %7, i64 4, i64 1024, i64 4, i64 256, i64 1280, ptr @scatter_map_13)
  %8 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1024) to i64))
  call void @upmemrt_dpu_gather(ptr %5, ptr %8, i64 4, i64 1024, i64 4, i64 256, i64 2048, ptr @scatter_map_13)
  call void @upmemrt_dpu_free(ptr %5)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.pow.f32(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.cos.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sin.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
