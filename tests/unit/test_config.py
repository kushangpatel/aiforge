"""Unit tests for AIForgeSettings (infra/config.py)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aiforge.infra.config import AIForgeSettings


class TestDefaults:
    def test_default_provider(self, test_settings: AIForgeSettings) -> None:
        assert test_settings.PROVIDER == "openai"

    def test_default_model(self, test_settings: AIForgeSettings) -> None:
        assert test_settings.MODEL == "gpt-4o"

    def test_default_embedding(self, test_settings: AIForgeSettings) -> None:
        assert test_settings.EMBEDDING_MODEL == "text-embedding-3-small"

    def test_default_temperature(self, test_settings: AIForgeSettings) -> None:
        assert test_settings.TEMPERATURE == 0.0


class TestValidation:
    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIForgeSettings(PROVIDER="unsupported_provider")

    def test_temperature_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIForgeSettings(TEMPERATURE=3.0, OPENAI_API_KEY="sk-test")

    def test_embedding_mismatch_raises(self) -> None:
        """voyage-3 is an Anthropic-ecosystem embedding, not valid for openai provider."""
        with pytest.raises(ValidationError):
            AIForgeSettings(
                PROVIDER="openai",
                EMBEDDING_MODEL="voyage-3",
                OPENAI_API_KEY="sk-test",
            )

    def test_valid_anthropic_embedding(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = AIForgeSettings(
                PROVIDER="anthropic",
                MODEL="claude-3-5-sonnet-20241022",
                EMBEDDING_MODEL="voyage-3",
                ANTHROPIC_API_KEY="test-key",
                VOYAGE_API_KEY="pa-test",
            )
        assert settings.EMBEDDING_MODEL == "voyage-3"


class TestEnvOverride:
    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIFORGE_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("AIFORGE_OPENAI_API_KEY", "sk-test")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = AIForgeSettings()
        assert settings.MODEL == "gpt-4o-mini"

    def test_constructor_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIFORGE_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("AIFORGE_OPENAI_API_KEY", "sk-test")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = AIForgeSettings(MODEL="gpt-4o")
        assert settings.MODEL == "gpt-4o"


class TestYAMLLoading:
    def test_from_yaml(self, tmp_path: Path) -> None:
        config = {
            "MODEL": "gpt-4o-mini",
            "TEMPERATURE": "0.5",
            "OPENAI_API_KEY": "sk-test-from-yaml",
        }
        yaml_file = tmp_path / "aiforge.yaml"
        yaml_file.write_text(yaml.dump(config))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = AIForgeSettings.from_yaml(yaml_file)

        assert settings.MODEL == "gpt-4o-mini"
        assert settings.TEMPERATURE == 0.5


class TestMissingKeyWarning:
    def test_missing_api_key_emits_warning(self) -> None:
        with pytest.warns(UserWarning, match="AIFORGE_OPENAI_API_KEY"):
            AIForgeSettings(PROVIDER="openai", OPENAI_API_KEY=None)


class TestModelFieldsSet:
    """Regression: model_fields_set must reflect when MODEL came from config vs. default.

    The model router in BaseModule uses ``"MODEL" in settings.model_fields_set``
    to decide whether to auto-select a model.  If the user configured a model
    (via YAML, env var, or constructor), the router must not override it.
    """

    def test_yaml_model_appears_in_fields_set(self, tmp_path: Path) -> None:
        """MODEL from YAML must appear in model_fields_set."""
        config = {"model": "gpt-4o", "openai_api_key": "sk-test"}
        yaml_file = tmp_path / "aiforge.yaml"
        yaml_file.write_text(yaml.dump(config))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = AIForgeSettings.from_yaml(yaml_file)

        assert "MODEL" in settings.model_fields_set

    def test_env_var_model_appears_in_fields_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MODEL from AIFORGE_MODEL env var must appear in model_fields_set."""
        monkeypatch.setenv("AIFORGE_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("AIFORGE_OPENAI_API_KEY", "sk-test")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = AIForgeSettings()

        assert "MODEL" in settings.model_fields_set

    def test_default_model_not_in_fields_set(self, tmp_path: Path) -> None:
        """When model is not set anywhere, MODEL must NOT be in model_fields_set."""
        config = {"openai_api_key": "sk-test"}
        yaml_file = tmp_path / "aiforge.yaml"
        yaml_file.write_text(yaml.dump(config))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = AIForgeSettings.from_yaml(yaml_file)

        assert "MODEL" not in settings.model_fields_set

    def test_constructor_kwarg_model_in_fields_set(self) -> None:
        """MODEL passed as constructor kwarg must appear in model_fields_set."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = AIForgeSettings(MODEL="gpt-4o", OPENAI_API_KEY="sk-test")
        assert "MODEL" in settings.model_fields_set
