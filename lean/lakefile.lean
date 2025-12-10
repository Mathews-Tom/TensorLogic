import Lake
open Lake DSL

package «tensorlogic» {
  -- Lean 4 project configuration for TensorLogic verification
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «TensorLogic» {
  -- Main TensorLogic theorem library
  roots := #[`TensorLogic]
}
