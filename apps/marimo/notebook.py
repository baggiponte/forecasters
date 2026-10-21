import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from forecasters_lineup.engine import (
        DemoSettings,
        default_model_specs,
        settings_from_env,
    )
    from forecasters_lineup_marimo.widget import LineupWidget

    return DemoSettings, LineupWidget, default_model_specs, mo, settings_from_env


@app.cell
def _(DemoSettings, default_model_specs, settings_from_env):
    from pathlib import Path

    settings = settings_from_env()
    if settings.cache_dir is None:
        local_cache = Path(".hf-cache")
        if local_cache.exists():
            settings = DemoSettings(cache_dir=local_cache, local_files_only=True)
    specs = default_model_specs(settings)
    return (specs,)


@app.cell
def _(specs):
    # Load each native checkpoint once in this kernel and reuse it for every request.
    models = {spec.key: spec.factory() for spec in specs}
    [(spec.key, type(models[spec.key]).__name__) for spec in specs]
    return (models,)


@app.cell
def _(LineupWidget, models, mo, specs):
    raw_lineup = LineupWidget(models, specs=specs)
    lineup = mo.ui.anywidget(raw_lineup)
    lineup  # noqa: B018 -- marimo renders the final expression


@app.cell
def _(mo):
    mo.md("""
    The widget calls the native adapters in this kernel and paints results from
    public AnyWidget custom messages (request/reply), with JSON-safe result
    snapshots retained on the widget for inspection. No external FastAPI server
    or forecast HTTP request is involved; the ordinary marimo websocket carries
    the notebook session. Draw a line, then try presets, model toggles and
    **Bands 80%**. Stale replies are ignored by the renderer.
    """)


if __name__ == "__main__":
    app.run()
