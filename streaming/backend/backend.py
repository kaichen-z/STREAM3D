from __future__ import annotations

import gc
from pathlib import Path

from streaming.data import StreamingExample

from .config import StreamingBackendConfig, build_streaming_backend_config
from .selected_chunk_plan import (
    _build_selected_chunk_plan_context,
    _prepare_warmup_result,
    _build_warmup_prefix_record,
    _build_selected_runtime,
    _run_stage2_disabled_chunk,
    _run_stage2_enabled_chunk,
    _save_selected_chunk_bundle,
)


class StreamingSam3DBackend:
    def __init__(self, config: StreamingBackendConfig) -> None:
        self.config = config

    def run_example(self, example: StreamingExample) -> Path | None:
        context = _build_selected_chunk_plan_context(
            self.config, example, example.execution_plan
        )

        for chunk_spec in context.execution_plan.warmup_plan:
            context.seen_chunk_specs.append(chunk_spec)
            warmup_result = _prepare_warmup_result(context, chunk_spec)
            context.warmup_prefix.append(
                _build_warmup_prefix_record(context, warmup_result)
            )

            if warmup_result.chunk_index not in context.reconstruction_indices:
                warmup_result.release()
                del warmup_result
                gc.collect()
                continue

            stage1_runtime = _build_selected_runtime(
                context,
                selected_views=warmup_result.selected_views,
                chunk_name=warmup_result.chunk_name,
                chunk_index=warmup_result.chunk_index,
            )
            if not context.stage2_selection_config.enabled:
                final_runtime, artifacts = _run_stage2_disabled_chunk(
                    context,
                    warmup_result=warmup_result,
                    stage1_runtime=stage1_runtime,
                )
            else:
                final_runtime, artifacts = _run_stage2_enabled_chunk(
                    context,
                    warmup_result=warmup_result,
                    stage1_runtime=stage1_runtime,
                )

            context.reconstructed_count += 1
            is_final_chunk = context.reconstructed_count == len(
                context.execution_plan.reconstruction_plan
            )
            _save_selected_chunk_bundle(
                context,
                warmup_result=warmup_result,
                stage1_runtime=stage1_runtime,
                final_runtime=final_runtime,
                artifacts=artifacts,
                is_final_chunk=is_final_chunk,
            )

            warmup_result.release()
            del stage1_runtime
            del final_runtime
            del artifacts
            del warmup_result
            gc.collect()

        return context.final_result_dir


__all__ = [
    "StreamingBackendConfig",
    "StreamingSam3DBackend",
    "build_streaming_backend_config",
    "run_selected_chunk_plan",
]
