


## exp tags which have a @results handle with bayesian results
## TODO redo the measurements for red_256MB... The solutions I found explode the stack
## Also do we have a solution for when we pick a config that ends up failing compilation?
source_dirs = {
  "prim_red_cinm2_CA": "cycle accurate",
  "prim_red_cinm2_CA_200ms": "cycle accurate (timeout 200ms)",
  "prim_red_cinm2_fast": "\"fast\"",
  "prim_red_cinm2_hybrid200": "hybrid (timeout 200ms)",
  "prim_red_cinm2_hybrid400": "hybrid (timeout 400ms)",
}


## TODO for each of those tags:
## run 

