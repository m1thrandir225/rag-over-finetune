from dataclasses import dataclass
from typing import Optional

from langchain_mcp_adapters.client import MultiServerMCPClient


@dataclass
class MCPClient:

    servers: dict[str, dict]
    _client: Optional[MultiServerMCPClient] = None

    def __post_init__(self):
        if self.servers:
            self._client = MultiServerMCPClient(self.servers)

    async def get_tools(self) -> list:

        if not self._client:
            raise ValueError(
                "No MCP servers configured. Add servers to config.json under 'mcp_servers'."
            )

        return await self._client.get_tools()

    async def get_resources(self, server_name: str, uris: Optional[list[str]] = None):
        if not self._client:
            raise ValueError("No MCP servers configured")

        return await self._client.get_resources(server_name, uris=uris)

    async def get_prompt(
            self, server_name: str, prompt_name: str, arguments: Optional[dict] = None
    ):

        if not self._client:
            raise ValueError("No MCP servers configured")

        return await self._client.get_prompt(
            server_name, prompt_name, arguments=arguments
        )

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def server_names(self) -> list[str]:
        return list(self.servers.keys())