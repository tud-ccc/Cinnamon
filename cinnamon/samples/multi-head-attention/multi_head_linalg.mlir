#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {torch.debug_module_name = "MultiHeadAttention"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x4x8xf64>, %arg1: tensor<1x4x8xf64>, %arg2: tensor<1x4x8xf64>) -> tensor<?x?x8xf64> {
    %cst = arith.constant dense<[[-0.25489680995233127, 1.4905381844142227, 1.2333979674831674, -0.35834049378971178, 0.80108256741189476, 0.50584012696518876, 0.55371378319427, -2.5997339353591729], [-0.50209984651186645, -0.027840017855643433, 0.8513323555119795, 0.027412119471579781, -0.60017830224094304, -0.049254384678932325, 0.73371347469520853, -0.64617619890621703], [-0.087664434080351059, -0.030398684790371545, -0.49141150290760766, 0.36140736222931558, 0.81457455439189896, 0.24865024620397036, 0.24350960087840601, 0.10851793920787658], [-1.042945511117747, -0.17279985553224869, -0.17728793863851508, -2.2601854476903172, 1.636901925038742, -0.075576617218713629, 1.5367149045612882, -0.40369850099826077], [0.49747022602493718, 1.3255820077032283, 1.5439643008598614, 2.5871386633837639, 1.3105349287142647, 0.27959638352274824, 0.61775048426591339, 0.79691551288837836], [-0.77421860138926502, -1.1397143654038289, 0.33003841333333328, 0.55678327458720589, -1.5911798286787084, -1.3807045828599027, 0.72007162706236394, -0.68298684053315006], [0.55062607267262775, -0.32545127135341556, -0.66866612002681713, -0.91050769018119659, 1.3592290941045935, 0.24160153097371206, -2.1125279454665264, 0.60558418000446512], [0.9035609385543123, -0.16867208518032833, 0.43686062271062498, 0.65501117138462706, -0.044031904853287014, 0.20000871685438107, 0.85675179410546398, -0.34605802169972805]]> : tensor<8x8xf64>
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 0xFFF0000000000000 : f64
    %false = arith.constant false
    %cst_2 = arith.constant 4.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %cst_3 = arith.constant dense<[1, 2, 4, 16]> : tensor<4xi64>
    %cst_4 = arith.constant dense<[1, 4, 32]> : tensor<3xi64>
    %cst_5 = arith.constant dense<[1.9591273023515805, -0.76925432129478521, 1.4460098912223047, 0.87236907567916732, -0.078867630070645608, 1.0560086388829639, -0.76442623771310636, 0.85249265216921066]> : tensor<8xf64>
    %cst_6 = arith.constant dense<[[0.30494463129193888, 0.94280904116947961, -0.91734794195222069, 0.35920242588001611, 1.6935617898854043, 2.4258224394653474, 0.2927760319497249, 1.8944759694393118], [-1.5071401228466226, 0.75507682930027453, 1.3387348028060215, 1.2760170257111605, 1.4585985844165876, 0.50933016137883114, -0.10218769082725609, -1.0123694223502757], [0.90171047782726398, -0.0076525029524271227, 1.8756642387168074, -0.23934074571704222, 0.3837105561949346, -0.56937993389589814, 1.3676419284127004, -0.21791872330083731], [0.71205267114742787, 1.6068106852343726, -1.5711056919878785, 0.86556272609861872, -0.83783802708223431, 0.845870131405968, 0.26104025514660478, 0.34906632766705181], [-0.22117003800045798, 0.2255237070102788, -1.7781485080797796, 0.57550167553121989, -0.79313332982308649, -0.6129511149743615, 1.1875165911596732, 0.28457943267634989], [-1.9985861439119823, -1.4250796761949958, -0.15852603348013672, 0.57592746815077445, -0.33958826214712645, -0.26324528975701461, 0.93254632701932771, -0.3756901441268668], [0.14612030678456864, 1.0246240593910672, 0.098771806048013874, -1.000817266404936, -0.60663437459487191, 0.20025851756382057, -0.85464816456717874, 0.49630994109844562], [-0.55313804889098162, -0.085306824019477687, 0.21120927268263009, 0.043587069073455317, 1.0043005636924738, -1.4016376470109972, 1.1260972452033375, -0.34828649268689321]]> : tensor<8x8xf64>
    %cst_7 = arith.constant dense<[0.54296136018194807, -1.0655239718807912, -1.3039516048318485, 1.7375909545591339, 0.08035308094973731, -0.50146390884008163, 0.49754564872954715, -0.4302509747852104]> : tensor<8xf64>
    %cst_8 = arith.constant dense<[[-3.2745803646348355, 0.65248905520009715, 0.15617091794494942, -0.28222841270461368, 0.46254743720879443, -2.0102450850351143, 0.13696134511118821, -0.3844238490005038], [-0.26764466102581347, 0.89829214853740646, -1.3227178818578345, 0.67152795803272591, -1.4782537197123458, -0.53418319006126602, -0.53264224294361062, -0.71951829256088418], [-0.61983320569654243, -0.44682372570033058, 0.51018646794433298, -1.1601319762721174, -0.38927626687239519, 0.69050825763957824, 0.25282183370383798, 1.6515500401699486], [0.73467608648500227, -1.3254998370516855, 0.77747048768703997, -0.99209580611069414, 1.967004461630032, 0.82086473654095337, 0.010556864252676915, -1.4334991893606857], [-1.8668813374385476, -0.97840504649908133, 0.48752921900359025, -0.13534845307613205, 1.8480080449067406, 0.16397248232129177, 0.52018050335943944, 0.18802042943190167], [-0.6104793785034015, 0.61141788549059506, 0.015868451435762675, -0.44458225635865872, 1.0912128888284023, -0.41527334926344484, -0.36877544705798077, -0.0094507518756244621], [-0.22682591876290031, -0.36441249535386039, -1.785314230669468, 0.073744561435228506, -0.034619626749076537, 0.65467788320271392, -0.53490369529918291, -0.088186757384451753], [3.1591602166810762, -0.74657034717502047, -0.077158944960388054, 1.1193578499461052, -0.54073386177290905, -0.78158713059051599, -0.53276336797034007, 0.69710568282322682]]> : tensor<8x8xf64>
    %cst_9 = arith.constant dense<[0.6484746362865389, 1.0022416475846243, 0.36168406373596756, 1.9785626165749564, -0.2578416128343648, -1.1182668091028038, 1.2215481389628127, 1.1013617841487044]> : tensor<8xf64>
    %cst_10 = arith.constant dense<[[0.27068375000087652, -0.73933180592443426, 2.0044274982272805, 1.0176876776412989, -0.70087130168552381, -1.0399172274179889, 0.72709408980141332, 0.375019820908287], [0.55382724069286371, -0.43773556392708901, 0.65291860277040115, -0.0056239814224785306, -1.6229841299370333, 0.48825489286816759, -1.5820231331110164, -1.5582057336230082], [-0.27544506385248291, 1.3196602518037448, -1.4997875449839682, -1.4711132414715955, -0.56646430068112774, -0.69730212495288624, -0.34750486315599266, -0.1076581749404019], [-0.030806890725352858, -0.64125774709171557, 1.619248012359533, 0.86453267610858298, -0.26689695897534627, -1.2675066552116518, -1.2110063637113659, 0.59649203711590959], [1.0098515568536799, 0.93862398490284437, -0.20445243125348256, 0.33291408970799513, 1.3760925252590823, -0.53399645404891671, -1.0236284219244958, -0.19862259737214363], [-0.20484509572308496, 0.68380417619107425, -0.34215019857694284, 1.5046745310560581, 1.1523106281688544, -0.17887566442219646, 1.7281141384178951, 1.0954171695049293], [0.41828693848462273, -0.361373588038918, -0.27536322724853607, -0.16122444978698394, -0.1135167954143006, -1.3669626594784634, -0.68082932389970385, 2.0986160244856849], [-1.2544718581096572, 0.24383336590821339, -0.16322228877529896, -0.20519288829269963, -1.7420532381913565, -0.3558298888387571, -0.48831074543420799, 0.20771346733161014]]> : tensor<8x8xf64>
    %cst_11 = arith.constant dense<[-1.7126765050096819, 0.20674540281667528, -0.98789345045794108, -0.19254166976196188, -1.0717191617191222, -0.057183463740861099, 0.0017195693778868822, -2.7890312769829357]> : tensor<8xf64>
    %0 = tensor.empty() : tensor<8x8xf64>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<8x8xf64>) outs(%0 : tensor<8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<8x8xf64>
    %2 = tensor.empty() : tensor<1x4x8xf64>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x4x8xf64>) outs(%2 : tensor<1x4x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x4x8xf64>
    %4 = tensor.empty() : tensor<1x8x8xf64>
    %5 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<8x8xf64>) outs(%4 : tensor<1x8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x8x8xf64>
    %6 = linalg.fill ins(%cst_0 : f64) outs(%2 : tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
    %7 = linalg.batch_matmul ins(%3, %5 : tensor<1x4x8xf64>, tensor<1x8x8xf64>) outs(%6 : tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
    %8 = linalg.generic {indexing_maps = [#map2, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %cst_5 : tensor<1x4x8xf64>, tensor<8xf64>) outs(%2 : tensor<1x4x8xf64>) {
    ^bb0(%in: f64, %in_32: f64, %out: f64):
      %49 = arith.addf %in, %in_32 : f64
      linalg.yield %49 : f64
    } -> tensor<1x4x8xf64>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_6 : tensor<8x8xf64>) outs(%0 : tensor<8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<8x8xf64>
    %10 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x4x8xf64>) outs(%2 : tensor<1x4x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x4x8xf64>
    %11 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<8x8xf64>) outs(%4 : tensor<1x8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x8x8xf64>
    %12 = linalg.batch_matmul ins(%10, %11 : tensor<1x4x8xf64>, tensor<1x8x8xf64>) outs(%6 : tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
    %13 = linalg.generic {indexing_maps = [#map2, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %cst_7 : tensor<1x4x8xf64>, tensor<8xf64>) outs(%2 : tensor<1x4x8xf64>) {
    ^bb0(%in: f64, %in_32: f64, %out: f64):
      %49 = arith.addf %in, %in_32 : f64
      linalg.yield %49 : f64
    } -> tensor<1x4x8xf64>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_8 : tensor<8x8xf64>) outs(%0 : tensor<8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<8x8xf64>
    %15 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x4x8xf64>) outs(%2 : tensor<1x4x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x4x8xf64>
    %16 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<8x8xf64>) outs(%4 : tensor<1x8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x8x8xf64>
    %17 = linalg.batch_matmul ins(%15, %16 : tensor<1x4x8xf64>, tensor<1x8x8xf64>) outs(%6 : tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
    %18 = linalg.generic {indexing_maps = [#map2, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %cst_9 : tensor<1x4x8xf64>, tensor<8xf64>) outs(%2 : tensor<1x4x8xf64>) {
    ^bb0(%in: f64, %in_32: f64, %out: f64):
      %49 = arith.addf %in, %in_32 : f64
      linalg.yield %49 : f64
    } -> tensor<1x4x8xf64>
    %expanded = tensor.expand_shape %8 [[0], [1], [2, 3]] output_shape [1, 4, 2, 4] : tensor<1x4x8xf64> into tensor<1x4x2x4xf64>
    %19 = tensor.empty() : tensor<1x2x4x4xf64>
    %20 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x4x2x4xf64>) outs(%19 : tensor<1x2x4x4xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x2x4x4xf64>
    %cast = tensor.cast %20 : tensor<1x2x4x4xf64> to tensor<?x?x?x?xf64>
    cf.assert %false, "mismatching contracting dimension"
    cf.assert %false, "mismatching contracting dimension"
    %cast_12 = tensor.cast %cast : tensor<?x?x?x?xf64> to tensor<2x4x?x?xf64>
    %collapsed = tensor.collapse_shape %cast_12 [[0], [1], [2, 3]] : tensor<2x4x?x?xf64> into tensor<2x4x?xf64>
    %expanded_13 = tensor.expand_shape %13 [[0], [1], [2, 3]] output_shape [1, 4, 2, 4] : tensor<1x4x8xf64> into tensor<1x4x2x4xf64>
    %21 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_13 : tensor<1x4x2x4xf64>) outs(%19 : tensor<1x2x4x4xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x2x4x4xf64>
    %cast_14 = tensor.cast %21 : tensor<1x2x4x4xf64> to tensor<?x?x?x?xf64>
    cf.assert %false, "mismatching contracting dimension"
    cf.assert %false, "mismatching contracting dimension"
    %cast_15 = tensor.cast %cast_14 : tensor<?x?x?x?xf64> to tensor<2x4x?x?xf64>
    %collapsed_16 = tensor.collapse_shape %cast_15 [[0], [1], [2, 3]] : tensor<2x4x?x?xf64> into tensor<2x4x?xf64>
    %expanded_17 = tensor.expand_shape %18 [[0], [1], [2, 3]] output_shape [1, 4, 2, 4] : tensor<1x4x8xf64> into tensor<1x4x2x4xf64>
    %22 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_17 : tensor<1x4x2x4xf64>) outs(%19 : tensor<1x2x4x4xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x2x4x4xf64>
    %cast_18 = tensor.cast %22 : tensor<1x2x4x4xf64> to tensor<?x?x?x?xf64>
    cf.assert %false, "mismatching contracting dimension"
    cf.assert %false, "mismatching contracting dimension"
    %cast_19 = tensor.cast %cast_18 : tensor<?x?x?x?xf64> to tensor<2x4x?x?xf64>
    %collapsed_20 = tensor.collapse_shape %cast_19 [[0], [1], [2, 3]] : tensor<2x4x?x?xf64> into tensor<2x4x?xf64>
    %23 = tensor.empty() : tensor<2x16x4xf64>
    %cast_21 = tensor.cast %collapsed_16 : tensor<2x4x?xf64> to tensor<2x4x16xf64>
    %24 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cast_21 : tensor<2x4x16xf64>) outs(%23 : tensor<2x16x4xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<2x16x4xf64>
    %25 = tensor.empty() : tensor<2x4x4xf64>
    %26 = linalg.fill ins(%cst_0 : f64) outs(%25 : tensor<2x4x4xf64>) -> tensor<2x4x4xf64>
    %cast_22 = tensor.cast %collapsed : tensor<2x4x?xf64> to tensor<2x4x16xf64>
    %27 = linalg.batch_matmul ins(%cast_22, %24 : tensor<2x4x16xf64>, tensor<2x16x4xf64>) outs(%26 : tensor<2x4x4xf64>) -> tensor<2x4x4xf64>
    %28 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<2x4x4xf64>) outs(%25 : tensor<2x4x4xf64>) {
    ^bb0(%in: f64, %out: f64):
      %49 = arith.divf %in, %cst_2 : f64
      linalg.yield %49 : f64
    } -> tensor<2x4x4xf64>
    %29 = tensor.empty() : tensor<2x4xi64>
    %30 = linalg.fill ins(%c0_i64 : i64) outs(%29 : tensor<2x4xi64>) -> tensor<2x4xi64>
    %31 = tensor.empty() : tensor<2x4xf64>
    %32 = linalg.fill ins(%cst_1 : f64) outs(%31 : tensor<2x4xf64>) -> tensor<2x4xf64>
    %33:2 = linalg.generic {indexing_maps = [#map3, #map9, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28 : tensor<2x4x4xf64>) outs(%32, %30 : tensor<2x4xf64>, tensor<2x4xi64>) {
    ^bb0(%in: f64, %out: f64, %out_32: i64):
      %49 = linalg.index 2 : index
      %50 = arith.index_cast %49 : index to i64
      %51 = arith.maximumf %in, %out : f64
      %52 = arith.cmpf ogt, %in, %out : f64
      %53 = arith.select %52, %50, %out_32 : i64
      linalg.yield %51, %53 : f64, i64
    } -> (tensor<2x4xf64>, tensor<2x4xi64>)
    %cast_23 = tensor.cast %33#0 : tensor<2x4xf64> to tensor<?x?xf64>
    %expanded_24 = tensor.expand_shape %cast_23 [[0], [1, 2]] output_shape [%c2, %c4, 1] : tensor<?x?xf64> into tensor<?x?x1xf64>
    %34 = linalg.generic {indexing_maps = [#map3, #map10, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %expanded_24 : tensor<2x4x4xf64>, tensor<?x?x1xf64>) outs(%25 : tensor<2x4x4xf64>) {
    ^bb0(%in: f64, %in_32: f64, %out: f64):
      %49 = arith.subf %in, %in_32 : f64
      linalg.yield %49 : f64
    } -> tensor<2x4x4xf64>
    %35 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34 : tensor<2x4x4xf64>) outs(%25 : tensor<2x4x4xf64>) {
    ^bb0(%in: f64, %out: f64):
      %49 = math.exp %in : f64
      linalg.yield %49 : f64
    } -> tensor<2x4x4xf64>
    %36 = tensor.empty() : tensor<2x4x1xf64>
    %37 = linalg.fill ins(%cst_0 : f64) outs(%36 : tensor<2x4x1xf64>) -> tensor<2x4x1xf64>
    %38 = linalg.generic {indexing_maps = [#map3, #map10], iterator_types = ["parallel", "parallel", "reduction"]} ins(%35 : tensor<2x4x4xf64>) outs(%37 : tensor<2x4x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %49 = arith.addf %in, %out : f64
      linalg.yield %49 : f64
    } -> tensor<2x4x1xf64>
    %39 = linalg.generic {indexing_maps = [#map3, #map10, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %38 : tensor<2x4x4xf64>, tensor<2x4x1xf64>) outs(%25 : tensor<2x4x4xf64>) {
    ^bb0(%in: f64, %in_32: f64, %out: f64):
      %49 = arith.divf %in, %in_32 : f64
      linalg.yield %49 : f64
    } -> tensor<2x4x4xf64>
    %40 = tensor.empty() : tensor<2x4x16xf64>
    %41 = linalg.fill ins(%cst_0 : f64) outs(%40 : tensor<2x4x16xf64>) -> tensor<2x4x16xf64>
    %cast_25 = tensor.cast %collapsed_20 : tensor<2x4x?xf64> to tensor<2x4x16xf64>
    %42 = linalg.batch_matmul ins(%39, %cast_25 : tensor<2x4x4xf64>, tensor<2x4x16xf64>) outs(%41 : tensor<2x4x16xf64>) -> tensor<2x4x16xf64>
    %cast_26 = tensor.cast %42 : tensor<2x4x16xf64> to tensor<?x?x?xf64>
    %reshape = tensor.reshape %cast_26(%cst_3) : (tensor<?x?x?xf64>, tensor<4xi64>) -> tensor<?x?x?x?xf64>
    %43 = tensor.empty() : tensor<1x4x2x16xf64>
    %cast_27 = tensor.cast %reshape : tensor<?x?x?x?xf64> to tensor<1x2x4x16xf64>
    %44 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cast_27 : tensor<1x2x4x16xf64>) outs(%43 : tensor<1x4x2x16xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x4x2x16xf64>
    %cast_28 = tensor.cast %44 : tensor<1x4x2x16xf64> to tensor<?x?x?x?xf64>
    %reshape_29 = tensor.reshape %cast_28(%cst_4) : (tensor<?x?x?x?xf64>, tensor<3xi64>) -> tensor<?x?x?xf64>
    %45 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_10 : tensor<8x8xf64>) outs(%0 : tensor<8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<8x8xf64>
    cf.assert %false, "mismatching contracting dimension"
    %46 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45 : tensor<8x8xf64>) outs(%4 : tensor<1x8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<1x8x8xf64>
    %cast_30 = tensor.cast %reshape_29 : tensor<?x?x?xf64> to tensor<1x4x8xf64>
    %47 = linalg.batch_matmul ins(%cast_30, %46 : tensor<1x4x8xf64>, tensor<1x8x8xf64>) outs(%6 : tensor<1x4x8xf64>) -> tensor<1x4x8xf64>
    %48 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %cst_11 : tensor<1x4x8xf64>, tensor<8xf64>) outs(%2 : tensor<1x4x8xf64>) {
    ^bb0(%in: f64, %in_32: f64, %out: f64):
      %49 = arith.addf %in, %in_32 : f64
      linalg.yield %49 : f64
    } -> tensor<1x4x8xf64>
    %cast_31 = tensor.cast %48 : tensor<1x4x8xf64> to tensor<?x?x8xf64>
    return %cast_31 : tensor<?x?x8xf64>
  }
}