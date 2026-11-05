"""Explicit NPU metrics configuration; no silent downgrade to Level0."""


def extra_profile_options(profiler,pipe_metrics=False,metric=None):
    metric = metric or ('PipeUtilization' if pipe_metrics else None)
    if metric is None: return {}
    try:
        config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=getattr(profiler.AiCMetrics, metric),
            export_type=profiler.ExportType.Text)
    except AttributeError as error:
        raise RuntimeError(f'Installed torch_npu lacks the requested Level1/{metric} API') from error
    return {'experimental_config':config}
