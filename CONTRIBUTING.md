# Contributing to boxBot

Thank you for your interest in contributing! boxBot is designed to be extended
by the community — whether you're adding a new skill, building a display, or
improving the core.

## Ways to Contribute

### 1. Add a Skill
The easiest way to contribute. Skills are self-contained modules that give
boxBot new capabilities. See [docs/skills-development.md](docs/skills-development.md).

### 2. Add a Display
Create new screen layouts for the 7" display. See
[docs/display-development.md](docs/display-development.md).

### 3. Improve Core
Bug fixes, performance improvements, and new core features. Core changes
require discussion in an issue before a PR.

### 4. Documentation
Improve docs, add examples, fix typos — always welcome.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/boxBot.git
cd boxBot
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Follow the code style guidelines in [CLAUDE.md](CLAUDE.md)
3. Add tests for new functionality
4. Update documentation if needed
5. Open a PR with a clear description of the change

## Code Style

- Python 3.11+, type hints on public interfaces
- Keep modules small and single-purpose
- Hardware access through the HAL only
- Configuration via YAML, secrets via `.env`
- Run `ruff check` and `ruff format` before committing

## Reporting Issues

- Use GitHub Issues with the provided templates
- Include hardware details and OS version for hardware-related bugs
- Include logs from `logs/` if applicable

## Security

If you discover a security vulnerability, **do not open a public issue**.
Instead, contact the maintainers directly.
