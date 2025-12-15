import Lake
open Lake DSL

package «tensorlogic» where
  -- Lean 4 project configuration for TensorLogic verification
  -- Using standard library only (no mathlib dependency)

@[default_target]
lean_lib «TensorLogic» where
  -- Main TensorLogic theorem library
  roots := #[`TensorLogic]
