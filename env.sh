#!/usr/bin/env bash
if [ -n "${ZSH_VERSION-}" ]; then
  eval '_script_source="${(%):-%x}"'
else
  _script_source="${BASH_SOURCE:-$0}"
fi
_CINM_ROOT=$(cd "$(dirname "${_script_source}")" && pwd)
_add_to_path() {
  local var_name="$1"
  local new_path="$2"
  [ -d "$new_path" ] || return 0
  eval "local current_value=\"\${$var_name-}\""
  if [ -z "$current_value" ]; then
    eval "$var_name=\"$new_path\""
    return 0
  fi
  case ":$current_value:" in
    *":$new_path:"*) return 0 ;;
  esac
  eval "$var_name=\"$new_path:$current_value\""
}
_add_to_path PATH "${_CINM_ROOT}/build/bin"
_add_to_path PATH "${_CINM_ROOT}/third-party/llvm/build/bin"
_add_to_path PATH "${_CINM_ROOT}/third-party/torch-mlir/install/bin"
_add_to_path LD_LIBRARY_PATH "${_CINM_ROOT}/build/lib"
_add_to_path LD_LIBRARY_PATH "${_CINM_ROOT}/third-party/llvm/build/lib"
_add_to_path LD_LIBRARY_PATH "${_CINM_ROOT}/third-party/torch-mlir/install/lib"
_add_to_path PYTHONPATH "${_CINM_ROOT}"
_add_to_path PYTHONPATH "${_CINM_ROOT}/third-party/torch-mlir/install/python_packages"
_add_to_path PYTHONPATH "${_CINM_ROOT}/third-party/torch-mlir/install/python_packages/torch_mlir"
export CINM_ROOT="${_CINM_ROOT}"
export PATH
export LD_LIBRARY_PATH
export PYTHONPATH
unset -f _add_to_path
unset _script_source
unset _CINM_ROOT
_existing_ld_library_path="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="$HOME/Cinnamon/third-party/llvm/build/lib${_existing_ld_library_path:+:$_existing_ld_library_path}"
