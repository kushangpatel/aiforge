"""Unit tests for AIForgeSettings (infra/config.py)."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

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
        with pytest.raises(Exception):
            AIForgeSettings(PROVIDER="unsupported_provider")

    def test_temperature_out_of_range_raises(self) -> None:
        with pytest.raises(Exception):
            AIForgeSettings(TEMPERATURE=3.0, OPENAI_API_KEY="sk-test")

    def test_embedding_mismatch_raises(self) -> None:
        """voyage-3 is an Anthropic-ecosystem embedding, not valid for openai provider."""
        with pytest.raises(Exception):
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
