# boxBot

An open-source Claude agent that lives in an elegant wooden box. Built on a
Raspberry Pi 5 with the [Claude Agent SDK](https://github.com/anthropics/claude-code-sdk),
boxBot sees, hears, remembers, and communicates — acting as an ambient
household assistant that recognizes the people around it.

## Features

- **Conversational AI** — natural voice interaction powered by Claude
- **Person recognition** — fused visual + audio identification of household
  members, all processed on-device
- **Persistent memory** — structured memory extraction and contextual recall
  across conversations
- **Skills framework** — modular, extensible capabilities (reminders, weather,
  home control, custom scripts)
- **Display system** — swappable screen layouts (calendar, weather, agent
  status, photo slideshow)
- **WhatsApp integration** — secure registration with code-based enrollment,
  admin-gated access, hard block on unknown numbers
- **Scheduled autonomy** — wake/sleep cycles with recurring and one-off tasks
- **Privacy by design** — no cloud processing for perception, minimal data
  leaves the box

## Hardware

| Component | Model |
|-----------|-------|
| Compute | Raspberry Pi 5 (8 GB) |
| AI Accelerator | Raspberry Pi AI HAT+ (13 TOPS) |
| Display | 7" HDMI IPS LCD, 1024x600 |
| Camera | Pi Camera Module 3 Wide NoIR |
| Microphone | ReSpeaker XMOS XVF3000 4-Mic Array |
| Speaker | Waveshare 8ohm 5W |
| Input Controller | Adafruit KB2040 (RP2040) |
| Power | Official Pi 27W PD (5.1V/5A) |

## Project Structure

```
boxBot/
├── src/boxbot/          # Core application
│   ├── core/            # Agent loop, config, scheduler
│   ├── tools/           # 9 always-loaded tools (script, speak, display, search, web, etc.)
│   ├── sdk/             # Immutable SDK for sandbox scripts (display builder, etc.)
│   ├── skills/          # Skills framework and built-in skills
│   ├── displays/        # Display framework and built-in displays
│   ├── hardware/        # Hardware abstraction layer
│   ├── perception/      # Person detection, ReID, speaker ID
│   ├── memory/          # Memory extraction, storage, retrieval
│   ├── communication/   # Voice I/O, WhatsApp, auth
│   └── photos/          # Photo management and slideshow
├── skills/              # User-installed skills (plugin directory)
├── displays/            # User-installed displays (plugin directory)
├── config/              # Configuration templates
├── docs/                # Documentation
├── tests/               # Test suite
└── scripts/             # Setup and utility scripts
```

## Quick Start

> **Note:** boxBot is under active development. Setup instructions will be
> added as components are implemented.

### Prerequisites

- Raspberry Pi 5 with Raspberry Pi OS Bookworm (64-bit)
- Python 3.11+
- Hailo runtime for AI HAT+
- Claude API key

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/boxBot.git
cd boxBot
pip install -e ".[dev]"
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your settings
```

## Extending boxBot

### Adding a Skill

Drop a skill module into `skills/` — see
[docs/skills-development.md](docs/skills-development.md) and
`skills/example_skill/` for the template.

### Adding a Display

Drop a display module into `displays/` — see
[docs/display-development.md](docs/display-development.md) and
`displays/example_display/` for the template.

## Architecture

See [CLAUDE.md](CLAUDE.md) for core design principles and architecture
boundaries. See [docs/architecture.md](docs/architecture.md) for detailed
technical documentation.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before submitting a pull request.

## License

MIT License — see [LICENSE](LICENSE) for details.
