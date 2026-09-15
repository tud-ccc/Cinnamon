"""Python access to a Cinnamon build.

`cinnamon.paths` locates what a build installed -- the tools, the runtime
libraries and headers, the benchmark suites -- and needs nothing beyond the
standard library. `cinnamon.torch_backend` compiles PyTorch modules through
those tools and needs torch and torch-mlir, which is why they are an extra
(`pip install cinnamon[torch]`) rather than a dependency of the package.
"""
