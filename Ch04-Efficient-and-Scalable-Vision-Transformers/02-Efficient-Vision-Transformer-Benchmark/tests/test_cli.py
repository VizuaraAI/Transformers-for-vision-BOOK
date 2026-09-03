import json

from conftest import PROJECT_ROOT

from vision_bench.cli import build_parser, main


def test_parser_requires_a_subcommand() -> None:
    parser = build_parser()
    assert "show-config" in parser.format_help()


def test_show_config_command(capsys) -> None:
    result = main(
        [
            "show-config",
            "--preset",
            "quick",
            "--project-root",
            str(PROJECT_ROOT),
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "quick"
    assert len(payload["runs"]) == 6
