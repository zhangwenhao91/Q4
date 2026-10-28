"""Explicit NPU metrics configuration; no silent downgrade to Level0."""


def extra_profile_options(profiler,pipe_metrics=False):
    if not pipe_metrics: return {}
    try:
        config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=profiler.AiCMetrics.PipeUtilization,
            export_type=profiler.ExportType.Text)
    except AttributeError as error:
        raise RuntimeError('Installed torch_npu lacks the requested Level1/PipeUtilization API') from error
    return {'experimental_config':config}
