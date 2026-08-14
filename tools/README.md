# Tool Layout

`tools/` keeps thin compatibility entry points for established command paths. New code should use the categorized modules below.

## Model

- `model/inspect_onnx.py`: inspect ONNX structure and operators.
- `model/benchmark_onnx.py`: run ONNX Runtime CPU reference benchmarks.
- `model/analyze_npu_feasibility.py`: estimate NPU deployment feasibility from model audit and benchmark data.

## Compliance

- `compliance/check_environment.py`: inspect local deployment and benchmarking dependencies.
- `compliance/check_compliance.py`: evaluate project metrics against challenge constraints.

## Reports

- `reports/summarize_multi_validation.py`: summarize multi-person `run_system` outputs.
- `reports/build_member_b_reports.py`: generate member B report documents.
- `reports/build_submission_ppt.py`: build the submission PPT draft.
- `reports/fix_ppt_fonts.py`: normalize PPT font and spacing.
