"""Light checks for the permanent Codex navigation files."""

from __future__ import annotations

from pathlib import Path
import unittest


class NavigationDocumentationTests(unittest.TestCase):
    def test_map_references_real_essential_paths(self) -> None:
        root = Path(__file__).parents[1]
        agents = root / "AGENTS.md"
        codex_map = root / "docs" / "CODEX_MAP.md"

        self.assertTrue(agents.is_file())
        self.assertTrue(codex_map.is_file())

        map_content = codex_map.read_text(encoding="utf-8")
        essential_paths = (
            "src/assistant_ia/__main__.py",
            "src/assistant_ia/application.py",
            "src/assistant_ia/runtime.py",
            "src/assistant_ia/core/assistant.py",
            "src/assistant_ia/intelligence/prompt.py",
            "src/assistant_ia/memory/memory_repository.py",
            "src/assistant_ia/interfaces/terminal.py",
            "src/assistant_ia/interfaces/gui.py",
        )

        for relative_path in essential_paths:
            self.assertTrue((root / relative_path).is_file())
            self.assertIn(relative_path, map_content)

    def test_agents_points_to_map_and_preserves_firmware(self) -> None:
        root = Path(__file__).parents[1]
        content = (root / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn("docs/CODEX_MAP.md", content)
        self.assertIn("avant de rechercher largement", content)
        self.assertIn(
            "firmware/fripouille_esp32/main/main.c",
            content,
        )


if __name__ == "__main__":
    unittest.main()
