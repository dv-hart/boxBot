# config/

Configuration templates for boxBot. These `.example.yaml` files are committed
to the repo. Your actual config files (`config.yaml`, `whatsapp.yaml`) are
gitignored and should never be committed.

## Setup

```bash
cp config.example.yaml config.yaml
cp whatsapp.example.yaml whatsapp.yaml
```

Edit each file with your settings. API keys and tokens go in `.env` at the
project root (also gitignored).

## Files

- `config.example.yaml` — main configuration: agent, schedule, display,
  camera, audio, perception, memory, photos, logging
- `whatsapp.example.yaml` — WhatsApp Business API credentials and user
  whitelist
