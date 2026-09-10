from config.settings import Settings


def test_defaults_are_safe_and_model_is_configurable():
    settings = Settings(_env_file=None)
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_model == "openai/gpt-4o-mini-2024-07-18"
    assert "OPENROUTER_API_KEY" in settings.missing_services()
