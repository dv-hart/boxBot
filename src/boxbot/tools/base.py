"""Tool base class for the boxBot agent.

All tools inherit from Tool and implement the async execute() method.
The agent runtime uses name, description, and parameters to register
tools with the Claude Agent SDK.
"""

from __future__ import annotations

import abc
from typing import Any


class Tool(abc.ABC):
    """Base class for all boxBot tools.

    Subclasses must define:
        name: Unique identifier for the tool.
        description: Natural language description for the agent.
        parameters: JSON Schema dict defining input parameters.

    And implement:
        execute(**kwargs) -> str: Async method that performs the action.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Parameters matching the JSON Schema in self.parameters.

        Returns:
            A string result that becomes the tool response to the agent.
            For structured data, return JSON-serialized strings.
        """
        ...

    def to_schema(self) -> dict[str, Any]:
        """Return the tool definition as a dict for agent SDK registration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
