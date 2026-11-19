from pathlib import Path


def test_cli_validates_before_creating_run_artifacts():
    source = Path("rpent/cli/main.py").read_text(encoding="utf-8")

    assert "load_runtime_environment" not in source
    config_import = source.index("from rpent.utils.config import")
    cli_import = source.index("from rpent.cli.tui import")
    assert config_import < cli_import
    main = source.index("def main(")
    validation = source.index("validate_runtime_requirements(", main)
    output_initialization = source.index("init_output_dir(", main)
    assert validation < output_initialization






def test_example_leaves_shared_service_timeout_blank():
    values = dict(
        line.split("=", 1)
        for line in Path(".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    )

    assert values["RPENT_SHARED_SERVICE_TIMEOUT_S"] == ""


def test_project_uses_direct_dotenv_dependency_without_settings_framework():
    source = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"python-dotenv>=1.0"' in source
    assert "pydantic-settings" not in source
