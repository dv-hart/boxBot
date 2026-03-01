# tests/

Test suite for boxBot. Tests are organized to mirror the source structure.

## Structure

```
tests/
├── unit/                    # Unit tests (no hardware, mocked dependencies)
│   ├── test_config.py
│   ├── test_scheduler.py
│   ├── test_memory_store.py
│   ├── test_memory_extraction.py
│   ├── test_memory_retrieval.py
│   ├── test_skill_registry.py
│   ├── test_display_manager.py
│   ├── test_auth.py
│   ├── test_photo_manager.py
│   └── test_fusion.py
├── integration/             # Integration tests (may need hardware or APIs)
│   ├── test_perception_pipeline.py
│   ├── test_voice_pipeline.py
│   └── test_agent_conversation.py
└── conftest.py              # Shared fixtures
```

## Running Tests

```bash
# All unit tests
pytest tests/unit/

# Specific module
pytest tests/unit/test_memory_store.py

# With coverage
pytest tests/ --cov=boxbot --cov-report=term-missing

# Integration tests (requires hardware or API keys)
pytest tests/integration/ --run-integration
```

## Conventions

- Unit tests mock all hardware and external APIs
- Integration tests are skipped by default (use `--run-integration` flag)
- Every skill and display should have at least one test
- Use `pytest-asyncio` for async tests
