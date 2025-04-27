# agent/tools/loader.py
import json
import os
from typing import Optional
from .models import MCPConfig # Import the Pydantic model

# Store the loaded config globally to avoid repeated file reads/parsing
_mcp_config_cache: Optional[MCPConfig] = None

def get_mcp_config(mcp_json_path: str = "mcp.json") -> MCPConfig:
    """
    Loads and parses the mcp.json file into an MCPConfig object.
    Caches the result to avoid re-reading the file on subsequent calls.

    Args:
        mcp_json_path (str): The path to the mcp.json file relative to the project root.
                             Defaults to "mcp.json".

    Returns:
        MCPConfig: The parsed MCP configuration object.

    Raises:
        FileNotFoundError: If the mcp.json file is not found at the specified path.
        ValueError: If the JSON is invalid or doesn't match the MCPConfig schema.
    """
    global _mcp_config_cache
    if _mcp_config_cache is None:
        if not os.path.exists(mcp_json_path):
            raise FileNotFoundError(f"MCP configuration file not found at: {mcp_json_path}")

        try:
            with open(mcp_json_path, 'r') as f:
                mcp_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {mcp_json_path}: {e}")

        try:
            _mcp_config_cache = MCPConfig.model_validate(mcp_data)
            print(f"Successfully loaded and validated MCP config from {mcp_json_path}")
        except Exception as e: # Catch Pydantic validation errors
            raise ValueError(f"MCP configuration validation failed: {e}")

    return _mcp_config_cache

if __name__ == "__main__":
    # Example usage: Load the config and print server names
    try:
        config = get_mcp_config() # Assumes mcp.json is in the root directory
        print("\nLoaded MCP Servers:")
        for server_name in config.mcpServers:
            print(f"- {server_name}: {config.mcpServers[server_name].description}")

        # Test caching
        config_again = get_mcp_config()
        print(f"\nIs it the same config instance? {config is config_again}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading MCP config: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 