"""Streamlit widget for the Pipeline composition engine."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

# Module types available for pipeline steps.
# Excluded: DocQA / SemanticSearch (require pre-ingestion),
#           Extractor (requires a Pydantic output_schema that can't be
#           configured through a Streamlit form).
_AVAILABLE_MODULES = {
    "Summarizer": {
        "import": "aiforge.modules.summarize.module",
        "class": "Summarizer",
        "config_fields": {
            "style": {
                "type": "selectbox",
                "label": "Style",
                "options": ["concise", "detailed", "bullet_points", "executive"],
                "default": "concise",
            },
            "max_length": {
                "type": "slider",
                "label": "Max length (words)",
                "min": 50,
                "max": 2000,
                "default": 500,
            },
        },
    },
    "Classifier": {
        "import": "aiforge.modules.classify.module",
        "class": "Classifier",
        "config_fields": {
            "labels": {
                "type": "text_area",
                "label": "Labels (one per line)",
                "default": "positive\nnegative\nneutral",
                "help": "Enter each classification label on its own line.",
                "height": 80,
            },
            "multi_label": {
                "type": "checkbox",
                "label": "Multi-label",
                "default": False,
            },
        },
    },
    "Generator": {
        "import": "aiforge.modules.generate.module",
        "class": "Generator",
        "config_fields": {
            "temperature": {
                "type": "slider",
                "label": "Temperature",
                "min": 0.0,
                "max": 2.0,
                "default": 0.7,
                "step": 0.1,
            },
        },
    },
    "Moderator": {
        "import": "aiforge.modules.moderate.module",
        "class": "Moderator",
        "config_fields": {
            "checks": {
                "type": "multiselect",
                "label": "Checks",
                "options": ["pii", "toxicity", "prompt_injection"],
                "default": ["pii", "toxicity", "prompt_injection"],
            },
        },
    },
    "Translator": {
        "import": "aiforge.modules.translate.module",
        "class": "Translator",
        "config_fields": {
            "target_language": {
                "type": "selectbox",
                "label": "Target language",
                "options": [
                    "English",
                    "Spanish",
                    "French",
                    "German",
                    "Italian",
                    "Portuguese",
                    "Chinese",
                    "Japanese",
                    "Korean",
                    "Arabic",
                    "Hindi",
                    "Russian",
                    "Dutch",
                    "Swedish",
                    "Polish",
                ],
                "default": "Spanish",
            },
            "formality": {
                "type": "selectbox",
                "label": "Formality",
                "options": ["neutral", "formal", "informal"],
                "default": "neutral",
            },
        },
    },
    "Vision": {
        "import": "aiforge.modules.vision.module",
        "class": "Vision",
        "config_fields": {
            "task": {
                "type": "selectbox",
                "label": "Task",
                "options": ["describe", "qa", "extract", "classify"],
                "default": "describe",
            },
        },
    },
}

_SESSION_KEY = "pipeline_steps"
_MODULE_NAMES = list(_AVAILABLE_MODULES.keys())


def _init_state() -> None:
    """Initialise session state for the pipeline builder."""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = []


def _add_step() -> None:
    """Callback: append a new step with a stable unique ID."""
    idx = len(st.session_state[_SESSION_KEY])
    st.session_state[_SESSION_KEY].append(
        {"id": uuid.uuid4().hex[:8], "name": f"step_{idx + 1}", "module": "Summarizer"}
    )


def _remove_step(step_id: str) -> None:
    """Callback: remove a step by its stable ID."""
    st.session_state[_SESSION_KEY] = [
        s for s in st.session_state[_SESSION_KEY] if s["id"] != step_id
    ]


def _render_config_field(key_prefix: str, field_name: str, spec: dict[str, Any]) -> Any:
    """Render a single config field and return its value."""
    widget_key = f"{key_prefix}_{field_name}"
    ftype = spec["type"]
    if ftype == "selectbox":
        return st.selectbox(
            spec["label"],
            spec["options"],
            index=spec["options"].index(spec["default"]),
            key=widget_key,
        )
    if ftype == "slider":
        return st.slider(
            spec["label"],
            min_value=spec["min"],
            max_value=spec["max"],
            value=spec["default"],
            step=spec.get("step"),
            key=widget_key,
        )
    if ftype == "text_area":
        return st.text_area(
            spec["label"],
            value=spec["default"],
            help=spec.get("help", ""),
            height=spec.get("height", 80),
            key=widget_key,
        )
    if ftype == "checkbox":
        return st.checkbox(spec["label"], value=spec["default"], key=widget_key)
    if ftype == "multiselect":
        return st.multiselect(
            spec["label"],
            options=spec["options"],
            default=spec["default"],
            key=widget_key,
        )
    return spec.get("default")


def _build_module(module_type: str, config_values: dict[str, Any]) -> Any:
    """Instantiate a module from its type name and widget-collected config."""
    import importlib

    spec = _AVAILABLE_MODULES[module_type]
    mod = importlib.import_module(spec["import"])
    cls = getattr(mod, spec["class"])

    # Special handling for Classifier labels (text → list).
    kwargs = dict(config_values)
    if module_type == "Classifier" and "labels" in kwargs:
        raw = kwargs.pop("labels")
        kwargs["labels"] = [lb.strip() for lb in raw.splitlines() if lb.strip()]

    # Moderator: convert empty checks list to None (use defaults).
    if module_type == "Moderator" and "checks" in kwargs and not kwargs["checks"]:
        kwargs["checks"] = None

    return cls(**kwargs)


# ------------------------------------------------------------------
# Main render
# ------------------------------------------------------------------


def render() -> None:
    """Render the Pipeline builder playground widget."""
    _init_state()

    # ---- Step builder ----
    st.subheader("Pipeline Steps")
    st.caption(
        "Add modules to build a sequential pipeline. "
        "Each step receives the previous step's output as input."
    )

    steps = st.session_state[_SESSION_KEY]

    if not steps:
        st.info("No steps yet — click **Add Step** to start building a pipeline.")

    # Render each step in a container.  Use step["id"] (stable UUID) for
    # widget keys so removals don't cause stale session-state collisions.
    step_configs: list[tuple[str, str, dict[str, Any]]] = []
    for step in steps:
        sid = step["id"]
        with st.container(border=True):
            cols = st.columns([2, 3, 1])
            with cols[0]:
                name = st.text_input(
                    "Step name",
                    value=step["name"],
                    key=f"step_name_{sid}",
                )
            with cols[1]:
                module_type = st.selectbox(
                    "Module",
                    _MODULE_NAMES,
                    index=_MODULE_NAMES.index(step["module"]),
                    key=f"step_module_{sid}",
                )
            with cols[2]:
                st.write("")  # spacing
                st.button(
                    "✕",
                    key=f"remove_{sid}",
                    on_click=_remove_step,
                    args=(sid,),
                    help="Remove this step",
                )

            # Module-specific config fields.
            config_fields = _AVAILABLE_MODULES[module_type]["config_fields"]
            config_values: dict[str, Any] = {}
            if config_fields:
                with st.expander("Step configuration", expanded=False):
                    for field_name, field_spec in config_fields.items():
                        config_values[field_name] = _render_config_field(
                            f"cfg_{sid}", field_name, field_spec
                        )

            step_configs.append((name, module_type, config_values))

    st.button("➕ Add Step", on_click=_add_step)

    # ---- Input ----
    st.divider()
    st.subheader("Input")
    text = st.text_area(
        "Input text",
        height=200,
        placeholder="Enter the text to feed through the pipeline...",
    )

    # ---- Run ----
    can_run = bool(text.strip()) and len(step_configs) > 0
    if not st.button("▶ Run Pipeline", type="primary", disabled=not can_run):
        return

    # Validate step names are unique.
    names = [cfg[0] for cfg in step_configs]
    if len(names) != len(set(names)):
        st.error("Step names must be unique.")
        return

    with st.spinner("Building and running pipeline..."):
        try:
            from aiforge.pipeline.pipeline import Pipeline

            built_steps: list[tuple[str, Any]] = []
            for name, module_type, config_values in step_configs:
                module = _build_module(module_type, config_values)
                built_steps.append((name, module))

            pipe = Pipeline(built_steps)
            result = pipe.run(text)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline error: {exc}")
            return

    # ---- Results ----
    st.divider()

    if result.halted:
        st.warning(f"Pipeline halted at step **{result.halted_at}**.")

    # Aggregate metrics.
    st.subheader("Results")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric("Total Cost (USD)", f"${result.total_cost:.6f}")
    with mcol2:
        st.metric("Total Tokens", f"{result.total_tokens:,}")

    # Per-step results.
    for step_name, response in result.steps.items():
        with st.expander(f"**{step_name}**", expanded=True):
            # Show the main text output.
            try:
                output_text = response.to_text()
            except NotImplementedError:
                output_text = None

            if output_text:
                st.markdown(output_text)

            # Show full response details.
            with st.container():
                st.caption("Response details")
                st.json(response.model_dump())
