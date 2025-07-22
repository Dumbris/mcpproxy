"""FastMCP server implementation for Smart MCP Proxy."""

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mcp.types as types  # type: ignore[import-untyped]
from fastmcp import FastMCP  # type: ignore[import-untyped]
from fastmcp.client import Client  # type: ignore[import-untyped]
from fastmcp.tools.tool import Tool  # type: ignore[import-not-found]
from mcp.server.lowlevel.server import NotificationOptions  # type: ignore[import-untyped]

from ..indexer.facade import IndexerFacade
from ..logging import configure_logging, get_logger
from ..models.schemas import (
    EmbedderType,
    ToolMetadata,
    ToolRegistration,
)
from ..persistence.facade import PersistenceFacade
from .config import ConfigLoader

logger = get_logger()

# Patch the NotificationOptions constructor to change defaults
original_init = NotificationOptions.__init__


def patched_init(
    self, prompts_changed=False, resources_changed=False, tools_changed=True
):
    original_init(self, prompts_changed, resources_changed, tools_changed)


NotificationOptions.__init__ = patched_init


class SmartMCPProxyServer:
    """Smart MCP Proxy server using FastMCP."""

    def __init__(self, config_path: str = "mcp_config.json"):
        # Check environment variable first, then use provided path, then default
        self.config_path = os.getenv("MCPPROXY_CONFIG_PATH", config_path)
        self.config_loader = ConfigLoader(self.config_path)
        self.config = self.config_loader.load_config()

        # Transport configuration from environment variables
        self.transport = os.getenv("MCPPROXY_TRANSPORT", "stdio")
        self.host = os.getenv("MCPPROXY_HOST", "127.0.0.1")
        self.port = int(os.getenv("MCPPROXY_PORT", "8000"))

        # Tool pool limit configuration
        self.tools_limit = int(os.getenv("MCPPROXY_TOOLS_LIMIT", "15"))

        # Output truncation configuration
        truncate_len = os.getenv("MCPPROXY_TRUNCATE_OUTPUT_LEN")
        self.truncate_output_len = int(truncate_len) if truncate_len else None

        # External command execution after tools list changes
        self.list_changed_exec_cmd = os.getenv("MCPPROXY_LIST_CHANGED_EXEC")

        # Routing type configuration
        self.routing_type = os.getenv("MCPPROXY_ROUTING_TYPE", "CALL_TOOL").upper()
        if self.routing_type not in ["DYNAMIC", "CALL_TOOL"]:
            raise ValueError(f"Invalid MCPPROXY_ROUTING_TYPE: {self.routing_type}. Must be 'DYNAMIC' or 'CALL_TOOL'")

        # Tool name limit configuration
        tool_name_limit_env = os.getenv("MCPPROXY_TOOL_NAME_LIMIT")
        if tool_name_limit_env is not None:
            self.tool_name_limit = int(tool_name_limit_env)
        elif hasattr(self.config, "tool_name_limit") and self.config.tool_name_limit:
            self.tool_name_limit = self.config.tool_name_limit
        else:
            self.tool_name_limit = 60  # Default value

        # Will be initialized in lifespan
        self.persistence: PersistenceFacade | None = None
        self.indexer: IndexerFacade | None = None

        # Track upstream clients and proxy servers
        self.upstream_clients: dict[str, Client] = {}
        self.proxy_servers: dict[str, FastMCP] = {}
        self.registered_tools: dict[str, ToolRegistration] = {}
        self.current_tool_registrations: dict[str, Any] = {}

        # Store Tool objects from proxified servers in memory
        self.proxified_tools: dict[str, Tool] = {}  # tool_key -> Tool object

        # Track tool pool with metadata for eviction
        self.tool_pool_metadata: dict[
            str, dict[str, Any]
        ] = {}  # tool_name -> {timestamp, score, original_score}

        # Initialize FastMCP server with transport configuration
        if self.routing_type == "CALL_TOOL":
            instructions = """
            This server provides intelligent tool discovery and proxying for MCP servers.
            First, use 'retrieve_tools' to search and discover available tools from configured upstream servers.
            Then, use 'call_tool' with the tool name and arguments to execute the tool on the upstream server.
            Tools are not dynamically registered - use the call_tool interface instead.
            """
        else:  # DYNAMIC
            instructions = """
            This server provides intelligent tool discovery and proxying for MCP servers.
            Use 'retrieve_tools' to search and access tools from configured upstream servers.
            proxy tools are dynamically created and registered on the fly in accordance with the search results.
            Pass the original user query (if possible) to the 'retrieve_tools' tool to get the search results.
            """

        fastmcp_kwargs = {
            "name": "Smart MCP Proxy",
            "instructions": instructions,
            "lifespan": self._lifespan,
        }

        # Add host and port for non-stdio transports
        if self.transport != "stdio":
            fastmcp_kwargs["host"] = self.host
            fastmcp_kwargs["port"] = self.port

        self.mcp = FastMCP(**fastmcp_kwargs)
        self._setup_tools()

    def run(self) -> None:
        """Run the Smart MCP Proxy server with full initialization."""
        # Configure logging with debug level if requested
        log_level = os.getenv("MCPPROXY_LOG_LEVEL", "INFO")
        log_file = os.getenv("MCPPROXY_LOG_FILE")  # Optional file logging
        configure_logging(log_level, log_file)
        logger = get_logger()

        # Check for config file (already resolved in __init__)
        config_path = self.config_path

        if not Path(config_path).exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Creating sample configuration...")

            config_loader = ConfigLoader()
            config_loader.create_sample_config(config_path)

            logger.info(
                f"Please edit {config_path} and set required environment variables"
            )
            return

        try:
            logger.info(f"Starting Smart MCP Proxy on transport: {self.transport}")
            if self.transport != "stdio":
                logger.info(f"Server will be available at {self.host}:{self.port}")

            # Run the FastMCP app with configured transport
            if self.transport == "stdio":
                self.mcp.run()
            elif self.transport == "streamable-http":
                # For streamable-http transport, pass host and port
                self.mcp.run(
                    transport="streamable-http", host=self.host, port=self.port
                )
            elif self.transport == "sse":
                # For SSE transport (deprecated)
                self.mcp.run(transport="sse", host=self.host, port=self.port)
            else:
                # Fallback for any other transport
                self.mcp.run(transport=self.transport, host=self.host, port=self.port)

        except FileNotFoundError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error starting proxy: {e}")
            sys.exit(1)

    @asynccontextmanager
    async def _lifespan(self, app):
        """FastMCP lifespan context manager for resource management."""
        logger = get_logger()
        logger.info("Initializing Smart MCP Proxy resources...")

        try:
            await self._initialize_resources()
        except Exception as e:
            logger.error(f"Error during resource initialization: {type(e).__name__}: {e}")
            logger.error("Full initialization error details:", exc_info=True)
            raise  # Re-raise the exception

        try:
            yield  # Server is running
        finally:
            try:
                await self._cleanup_resources()
            except Exception as e:
                logger.error(f"Error during resource cleanup: {type(e).__name__}: {e}")
                logger.error("Full cleanup error details:", exc_info=True)

    async def _initialize_resources(self) -> None:
        """Core resource initialization logic."""
        logger = get_logger()

        try:
            # Check if we should reset data (useful when dimensions change)
            reset_data = os.getenv("MCPPROXY_RESET_DATA", "false").lower() == "true"

            # Determine vector dimension based on embedder type
            if self.config.embedder == EmbedderType.BM25:
                vector_dimension = 1  # BM25 uses placeholder vectors
            else:
                # For vector embedders, we'll set dimension after creating the embedder
                # Default to 384 for now, will be updated if needed
                vector_dimension = 384

            # Initialize persistence with appropriate dimension
            logger.debug(f"Initializing persistence with dimension: {vector_dimension}")
            self.persistence = PersistenceFacade(
                vector_dimension=vector_dimension, embedder_type=self.config.embedder
            )
            # Initialize PersistenceFacade asynchronously
            await self.persistence._ainit()

            # Reset data if requested
            if reset_data:
                logger.info("Resetting all data as requested...")
                await self.persistence.reset_all_data()

            # Initialize indexer
            logger.debug(f"Initializing indexer with embedder: {self.config.embedder}")
            self.indexer = IndexerFacade(
                self.persistence, self.config.embedder, self.config.hf_model
            )

            # Reset embedder data if requested (must be done after indexer creation)
            if reset_data:
                logger.info("Resetting embedder data...")
                await self.indexer.reset_embedder_data()

            # For non-BM25 embedders, update persistence with actual dimension
            if self.config.embedder != EmbedderType.BM25:
                actual_dimension = self.indexer.embedder.get_dimension()
                if actual_dimension != vector_dimension:
                    # Recreate persistence with correct dimension
                    logger.info(
                        f"Updating vector dimension from {vector_dimension} to {actual_dimension}"
                    )
                    await self.persistence.close()
                    self.persistence = PersistenceFacade(
                        vector_dimension=actual_dimension, embedder_type=self.config.embedder
                    )
                    # Update indexer to use new persistence
                    self.indexer.persistence = self.persistence

        except Exception as e:
            logger.error(f"Error during persistence/indexer initialization: {type(e).__name__}: {e}")
            logger.error("Full persistence/indexer error details:", exc_info=True)
            raise

        try:
            # Create upstream clients and proxy servers
            logger.debug("Creating upstream clients and proxy servers...")
            await self._create_upstream_clients_and_proxies()
            logger.debug(
                f"Created {len(self.proxy_servers)} proxy servers and {len(self.upstream_clients)} upstream clients"
            )
        except Exception as e:
            logger.error(f"Error during upstream client/proxy creation: {type(e).__name__}: {e}")
            logger.error("Full upstream client/proxy error details:", exc_info=True)
            raise

        try:
            # Discover and index tools from upstream servers
            logger.debug("Discovering and indexing tools...")
            await self.discover_and_index_tools()
        except Exception as e:
            logger.error(f"Error during tool discovery and indexing: {type(e).__name__}: {e}")
            logger.error("Full tool discovery error details:", exc_info=True)
            raise

        logger.info("Smart MCP Proxy resources initialized")

    async def _cleanup_resources(self) -> None:
        """Core resource cleanup logic."""
        logger = get_logger()
        logger.info("Shutting down Smart MCP Proxy resources...")

        # Close upstream clients
        for client in self.upstream_clients.values():
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        if self.persistence:
            await self.persistence.close()
        logger.info("Smart MCP Proxy resources cleaned up")

    def _setup_tools(self) -> None:
        """Setup core proxy tools."""

        @self.mcp.tool()
        async def retrieve_tools(query: str) -> str:
            """Search and retrieve tools based on query. Tools are dynamically created and registered on the fly in accordance with the search results.

            Args:
                query: Search query for finding relevant tools, pass the original user query (if possible) to the 'retrieve_tools' tool to get the search results.

            Returns:
                JSON string with discovered tools information
            """
            logger = get_logger()
            try:
                # Ensure indexer is initialized
                if not self.indexer:
                    return json.dumps({"error": "Indexer not initialized"})

                # Search for tools
                results = await self.indexer.search_tools(query, self.config.top_k)

                if not results:
                    return json.dumps(
                        {"message": "No relevant tools found", "tools": []}
                    )

                # For CALL_TOOL routing, just return tool information without registering
                if self.routing_type == "CALL_TOOL":
                    # Prepare tool information without registration
                    discovered_tools = []
                    for result in results:
                        tool_name = self._sanitize_tool_name(
                            result.tool.server_name, result.tool.name
                        )
                        
                        # Get input schema from params_json if available, or from proxified tools
                        input_schema = {}
                        if tool_name in self.proxified_tools:
                            proxified_tool = self.proxified_tools[tool_name]
                            if hasattr(proxified_tool, 'parameters'):
                                input_schema = proxified_tool.parameters
                        elif result.tool.params_json:
                            try:
                                input_schema = json.loads(result.tool.params_json)
                            except (json.JSONDecodeError, Exception):
                                input_schema = {}
                        
                        discovered_tools.append(
                            {
                                "name": tool_name,
                                "original_name": result.tool.name,
                                "server": result.tool.server_name,
                                "description": result.tool.description,
                                "score": result.score,
                                "input_schema": input_schema,
                            }
                        )

                    return json.dumps(
                        {
                            "message": f"Found {len(discovered_tools)} tools. Use 'call_tool' to execute them.",
                            "tools": discovered_tools,
                            "newly_registered": [],  # No registration in CALL_TOOL mode
                            "evicted_tools": [],  # No eviction in CALL_TOOL mode
                            "pool_size": 0,  # No pool in CALL_TOOL mode
                            "pool_limit": self.tools_limit,
                            "total_available_tools": len(self.proxified_tools),
                            "query": query,
                        }
                    )

                # For DYNAMIC routing, continue with existing registration logic
                # Prepare tools for registration
                tools_to_register = []
                for result in results:
                    tool_name = self._sanitize_tool_name(
                        result.tool.server_name, result.tool.name
                    )

                    # Skip if already registered
                    if tool_name in self.current_tool_registrations:
                        # Update timestamp for existing tool (freshen it)
                        if tool_name in self.tool_pool_metadata:
                            self.tool_pool_metadata[tool_name]["timestamp"] = (
                                time.time()
                            )
                            self.tool_pool_metadata[tool_name]["score"] = max(
                                self.tool_pool_metadata[tool_name]["score"],
                                result.score,
                            )
                        continue

                    # Get the actual Tool object from memory using sanitized key
                    if tool_name in self.proxified_tools:
                        actual_tool = self.proxified_tools[tool_name]
                        tools_to_register.append((tool_name, actual_tool, result.score))
                    else:
                        logger.warning(
                            f"Tool {tool_name} not found in proxified_tools memory"
                        )

                # Enforce pool limit before registering new tools
                evicted_tools = []
                if tools_to_register:
                    new_tools_info = [
                        (name, score) for name, _, score in tools_to_register
                    ]
                    evicted_tools = await self._enforce_tool_pool_limit(new_tools_info)

                # Register new tools using actual Tool objects
                newly_registered = []
                for tool_name, actual_tool, score in tools_to_register:
                    # Find the original server name from results
                    original_server_name = None
                    for result in results:
                        if (
                            self._sanitize_tool_name(
                                result.tool.server_name, result.tool.name
                            )
                            == tool_name
                        ):
                            original_server_name = result.tool.server_name
                            break
                    await self._register_proxy_tool(
                        actual_tool, tool_name, score, original_server_name
                    )
                    newly_registered.append(tool_name)

                # Prepare tool information
                registered_tools = []
                for result in results:
                    tool_name = self._sanitize_tool_name(
                        result.tool.server_name, result.tool.name
                    )
                    registered_tools.append(
                        {
                            "name": tool_name,
                            "original_name": result.tool.name,
                            "server": result.tool.server_name,
                            "description": result.tool.description,
                            "score": result.score,
                            "newly_registered": tool_name in newly_registered,
                        }
                    )

                message = f"Found {len(registered_tools)} tools, registered {len(newly_registered)} new tools"
                if evicted_tools:
                    message += f", evicted {len(evicted_tools)} tools to stay within limit ({self.tools_limit})"

                # Notify connected clients that the available tools list has changed so they can refresh
                try:
                    if newly_registered or evicted_tools:
                        # Standard notification (proper MCP way)
                        await self.mcp._mcp_server.request_context.session.send_notification(
                            types.ToolListChangedNotification(
                                method="notifications/tools/list_changed"
                            ),
                            related_request_id=self.mcp._mcp_server.request_context.request_id,
                        )
                        logger.debug(
                            f"Sent tools/list_changed notification {self.mcp._mcp_server.request_context.request_id}"
                        )

                        # Execute external command to trigger client refresh (workaround for clients 
                        # that don't properly handle tools/list_changed notifications)
                        await self._execute_list_changed_command()

                except Exception as notify_err:
                    logger.warning(
                        f"Failed to emit tools/list_changed notification: {notify_err}"
                    )

                return json.dumps(
                    {
                        "message": message,
                        "tools": registered_tools,
                        "newly_registered": newly_registered,
                        "evicted_tools": evicted_tools,
                        "pool_size": len(self.current_tool_registrations),
                        "pool_limit": self.tools_limit,
                        "total_available_tools": len(self.proxified_tools),
                        "query": query,
                    }
                )

            except Exception as e:
                return json.dumps({"error": str(e)})

        # Add call_tool function for CALL_TOOL routing type
        if self.routing_type == "CALL_TOOL":
            @self.mcp.tool()
            async def call_tool(name: str, args: dict[str, Any]) -> str:
                """Execute a tool on the upstream server using the call_tool interface.

                Args:
                    name: Name of the tool to execute (use names from retrieve_tools response)
                    args: Arguments to pass to the tool

                Returns:
                    Tool execution result
                """
                logger = get_logger()
                try:
                    # Parse server name and tool name from the sanitized name
                    # Format is typically: servername_toolname
                    if "_" not in name:
                        return json.dumps({"error": f"Invalid tool name format: {name}. Expected format: servername_toolname"})
                    
                    # Find the original tool in our indexed tools
                    if not self.indexer:
                        return json.dumps({"error": "Indexer not initialized"})
                    
                    # Search for the tool by name to get the original details
                    # This is a bit of a hack - we search for the tool name to find the original
                    all_tools = await self.persistence.get_all_tools() if self.persistence else []
                    
                    matching_tool = None
                    for tool_metadata in all_tools:
                        sanitized_name = self._sanitize_tool_name(tool_metadata.server_name, tool_metadata.name)
                        if sanitized_name == name:
                            matching_tool = tool_metadata
                            break
                    
                    if not matching_tool:
                        return json.dumps({"error": f"Tool '{name}' not found. Use retrieve_tools first to discover available tools."})
                    
                    # Get the proxy server for this tool's server
                    server_name = matching_tool.server_name
                    proxy_server = self.proxy_servers.get(server_name)
                    if not proxy_server:
                        return json.dumps({"error": f"Server '{server_name}' not available"})
                    
                    # Execute the tool on the upstream server
                    original_tool_name = matching_tool.name
                    logger.debug(f"Executing tool '{original_tool_name}' on server '{server_name}' with args: {args}")
                    
                    result = await proxy_server._mcp_call_tool(original_tool_name, args)
                    
                    # Process the result
                    output = ""
                    if result and len(result) > 0:
                        content = result[0]
                        if hasattr(content, "text"):
                            output = content.text
                        elif isinstance(content, dict) and "text" in content:
                            output = content["text"]
                        else:
                            output = str(result)
                    else:
                        output = str(result)

                    # Apply output truncation if configured
                    output = self._truncate_output(output)
                    
                    return output

                except Exception as e:
                    logger.error(f"Error executing tool '{name}': {e}")
                    return json.dumps({"error": f"Error executing tool '{name}': {str(e)}"})

    def _sanitize_tool_name(self, server_name: str, tool_name: str) -> str:
        """Sanitize tool name to comply with Google Gemini API requirements.

        Google Gemini API Requirements (more strict than general MCP):
        - Must start with letter or underscore
        - Only lowercase letters (a-z), numbers (0-9), underscores (_)
        - Maximum length configurable via MCPPROXY_TOOL_NAME_LIMIT (default 60)
        - No dots or dashes (unlike general MCP spec)

        Args:
            server_name: Name of the server
            tool_name: Original tool name

        Returns:
            Sanitized tool name that complies with Google Gemini API
        """
        import re

        # First sanitize individual parts - be more aggressive for Google API
        # Convert to lowercase and replace anything that isn't alphanumeric with underscore
        server_clean = re.sub(r"[^a-z0-9_]", "_", server_name.lower())
        tool_clean = re.sub(r"[^a-z0-9_]", "_", tool_name.lower())

        # Remove consecutive underscores
        server_clean = re.sub(r"_+", "_", server_clean)
        tool_clean = re.sub(r"_+", "_", tool_clean)

        # Remove leading/trailing underscores from parts
        server_clean = server_clean.strip("_")
        tool_clean = tool_clean.strip("_")

        # If parts are empty after cleaning, use defaults
        if not server_clean:
            server_clean = "server"
        if not tool_clean:
            tool_clean = "tool"

        # Combine server and tool name
        combined = f"{server_clean}_{tool_clean}"

        # Ensure starts with letter or underscore (Google requirement)
        if not re.match(r"^[a-z_]", combined):
            combined = f"tool_{combined}"

        # Truncate to configured limit if needed
        max_length = self.tool_name_limit
        if len(combined) > max_length:
            # Try to keep server prefix if possible
            if "_" in combined:
                parts = combined.split("_", 1)
                server_part = parts[0]
                tool_part = parts[1]

                # Reserve space for server part + underscore
                available_space = max_length - len(server_part) - 1
                if available_space > 3:  # Keep at least 3 chars of tool name
                    truncated = f"{server_part}_{tool_part[:available_space]}"
                else:
                    # Not enough space for meaningful server prefix
                    truncated = combined[:max_length]
            else:
                truncated = combined[:max_length]

            # Ensure doesn't end with underscore
            truncated = truncated.rstrip("_")

            # If we stripped all chars, add fallback
            if not truncated:
                truncated = "tool"

            combined = truncated

        # Final validation - ensure it still starts correctly and is valid
        if not re.match(r"^[a-z_]", combined):
            combined = f"tool_{combined}"
            # Re-truncate if needed
            if len(combined) > max_length:
                combined = combined[:max_length].rstrip("_")

        # Final fallback if somehow we ended up empty
        if not combined:
            combined = "tool"

        # Debug logging for validation
        logger = get_logger()
        if (
            len(combined) > max_length
            or not re.match(r"^[a-z_][a-z0-9_]*$", combined)
            or combined.endswith("_")
        ):
            logger.warning(
                f"Tool name may still be invalid: '{server_name}' + '{tool_name}' -> '{combined}' (len={len(combined)})"
            )

        return combined

    def _truncate_output(self, output: str) -> str:
        """Truncate output if it exceeds the configured length limit.

        Args:
            output: The original output string

        Returns:
            Truncated output with placeholder if needed, or original if within limit
        """
        if not self.truncate_output_len or len(output) <= self.truncate_output_len:
            return output

        # Calculate how many chars to show at start and end
        last_chars = 50
        first_chars = (
            self.truncate_output_len
            - last_chars
            - len(" <truncated by smart mcp proxy> ")
        )

        if first_chars <= 0:
            # If truncate length is too small, just show beginning
            return (
                output[: self.truncate_output_len] + " <truncated by smart mcp proxy>"
            )

        truncated = (
            output[:first_chars]
            + " <truncated by smart mcp proxy> "
            + output[-last_chars:]
        )

        return truncated

    async def _execute_list_changed_command(self) -> None:
        """Execute external command after tools list changes to trigger client refresh.
        
        This is a workaround for MCP clients that don't properly handle tools/list_changed notifications.
        """
        if not self.list_changed_exec_cmd:
            return
            
        logger = get_logger()
        try:
            # Execute command in background without blocking
            logger.debug(f"Executing list changed command: {self.list_changed_exec_cmd}")
            
            # Run command asynchronously
            process = await asyncio.create_subprocess_shell(
                self.list_changed_exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=60.0  # 60 second timeout
            )
            
            if process.returncode == 0:
                logger.debug("List changed command executed successfully")
            else:
                logger.warning(
                    f"List changed command failed with code {process.returncode}: "
                    f"stderr={stderr.decode().strip()}"
                )
                
        except asyncio.TimeoutError:
            logger.warning("List changed command timed out after 60 seconds")
        except Exception as e:
            logger.warning(f"Error executing list changed command: {e}")

    def _calculate_tool_weight(self, score: float, added_timestamp: float) -> float:
        """Calculate weighted score for tool eviction based on score and freshness.

        Args:
            score: Original search score (0.0 to 1.0)
            added_timestamp: Timestamp when tool was added to pool

        Returns:
            Weighted score (higher is better, less likely to be evicted)
        """
        current_time = time.time()
        age_seconds = current_time - added_timestamp

        # Normalize age (0 = fresh, 1 = old)
        # Tools older than 30 minutes get maximum age penalty
        max_age_seconds = 30 * 60  # 30 minutes
        age_normalized = min(1.0, age_seconds / max_age_seconds)

        # Weighted formula: 70% score, 30% freshness
        score_weight = 0.7
        freshness_weight = 0.3
        freshness_score = 1.0 - age_normalized

        weighted_score = (score * score_weight) + (freshness_score * freshness_weight)
        return weighted_score

    async def _enforce_tool_pool_limit(
        self, new_tools: list[tuple[str, float]]
    ) -> list[str]:
        """Enforce tool pool limit by evicting lowest-scoring tools.

        Args:
            new_tools: List of (tool_name, score) for tools to be added

        Returns:
            List of tool names that were evicted
        """
        current_pool_size = len(self.current_tool_registrations)
        new_tools_count = len(new_tools)
        total_after_addition = current_pool_size + new_tools_count

        if total_after_addition <= self.tools_limit:
            return []  # No eviction needed

        # Calculate how many tools to evict
        tools_to_evict_count = total_after_addition - self.tools_limit

        # Calculate weighted scores for all existing tools
        tool_weights = []
        for tool_name, metadata in self.tool_pool_metadata.items():
            if tool_name in self.current_tool_registrations:
                weight = self._calculate_tool_weight(
                    metadata["score"], metadata["timestamp"]
                )
                tool_weights.append((tool_name, weight))

        # Sort by weight (ascending - lowest weights first for eviction)
        tool_weights.sort(key=lambda x: x[1])

        # Evict the lowest scoring tools
        evicted_tools = []
        for i in range(min(tools_to_evict_count, len(tool_weights))):
            tool_name = tool_weights[i][0]
            await self._evict_tool(tool_name)
            evicted_tools.append(tool_name)

        return evicted_tools

    async def _evict_tool(self, tool_name: str) -> None:
        """Remove a tool from the active pool.

        Args:
            tool_name: Name of tool to evict
        """
        # Remove from FastMCP server
        if hasattr(self.mcp, "remove_tool"):
            self.mcp.remove_tool(tool_name)

        # Clean up tracking
        self.current_tool_registrations.pop(tool_name, None)
        self.registered_tools.pop(tool_name, None)
        self.tool_pool_metadata.pop(tool_name, None)

        logger = get_logger()
        logger.debug(f"Evicted tool from pool: {tool_name}")

    async def _send_unsolicited_tools_list(self) -> None:
        """EXPERIMENTAL: Send unsolicited tools list for testing purposes.

        This violates MCP protocol but can be useful for testing.
        """
        logger = get_logger()
        try:
            # Get current tools from FastMCP
            current_tools = await self.mcp._mcp_list_tools()

            # Create ListToolsResult
            tools_result = types.ListToolsResult(tools=current_tools)

            try:
                await self.mcp._mcp_server.request_context.session._send_response(
                    request_id=999,  # self.mcp._mcp_server.request_context.request_id,
                    response=tools_result,
                )
                logger.debug(
                    f"Sent experimental unsolicited tools list as response: {len(current_tools)} tools"
                )

            except Exception as e2:
                logger.debug(f"Method 2 (dummy response) failed: {e2}")

        except Exception as e:
            logger.warning(f"Error in experimental unsolicited tools list: {e}")

    async def _register_proxy_tool(
        self,
        tool_metadata: Any,
        tool_name: str,
        score: float = 0.0,
        server_name: str | None = None,
    ) -> None:
        """Register a single tool as a proxy that calls the upstream server transparently using Tool.from_tool."""
        from fastmcp.tools.tool import Tool

        # tool_metadata should be a Tool object from proxified servers
        if not isinstance(tool_metadata, Tool):
            logger = get_logger()
            logger.error(f"Expected Tool object, got {type(tool_metadata)}")
            return

        original_tool: Tool = tool_metadata
        # Use provided server_name or extract from tool_name (format: servername_toolname)
        if server_name is None:
            if "_" in tool_name:
                server_name = tool_name.split("_", 1)[0]
            else:
                server_name = "unknown"
        original_tool_name = original_tool.name

        # Define a transform_fn that forwards the call to the upstream server
        async def transform_fn(**kwargs):
            proxy_server = self.proxy_servers.get(server_name)
            if not proxy_server:
                return f"Error: Server {server_name} not available"
            # Forward the call to the upstream tool with original parameters
            result = await proxy_server._mcp_call_tool(original_tool_name, kwargs)
            output = ""
            if result and len(result) > 0:
                content = result[0]
                if hasattr(content, "text"):
                    output = content.text
                elif isinstance(content, dict) and "text" in content:
                    output = content["text"]
                else:
                    output = str(result)
            else:
                output = str(result)

            # Apply output truncation if configured
            return self._truncate_output(output)

        # Create a proxified tool using Tool.from_tool
        proxified_tool = Tool.from_tool(
            tool=original_tool,
            transform_fn=transform_fn,
            name=tool_name,  # Use unique name (e.g., servername_toolname)
        )
        self.mcp.add_tool(proxified_tool)

        # Track this registration
        self.current_tool_registrations[tool_name] = proxified_tool
        self.tool_pool_metadata[tool_name] = {
            "timestamp": time.time(),
            "score": score,
            "original_score": score,
        }

        # Track in registered_tools for metadata
        input_schema = (
            original_tool.parameters if hasattr(original_tool, "parameters") else {}
        )
        self.registered_tools[tool_name] = ToolRegistration(
            name=tool_name,
            description=original_tool.description or "",
            input_schema=input_schema,
            server_name=server_name,
        )

        logger = get_logger()
        logger.debug(
            f"Registered proxy tool (from_tool): {tool_name} (original: {original_tool_name}, score: {score:.3f})"
        )

    async def _create_upstream_clients_and_proxies(self) -> None:
        """Create upstream clients and proxy servers for all configured servers."""
        logger = get_logger()
        logger.info(
            f"Creating upstream clients for {len(self.config.mcp_servers)} servers..."
        )

        for server_name, server_config in self.config.mcp_servers.items():
            try:
                logger.debug(
                    f"Creating client for {server_name}: url={server_config.url}, command={getattr(server_config, 'command', None)}"
                )

                if server_config.url:
                    # Create client for URL-based server
                    logger.debug(
                        f"Creating URL-based client for {server_name} at {server_config.url}"
                    )
                    try:
                        client = Client(server_config.url)
                        logger.debug(f"URL client created for {server_name}")
                    except Exception as client_error:
                        logger.error(f"Failed to create URL client for {server_name}: {type(client_error).__name__}: {client_error}")
                        logger.error(f"URL client error details for {server_name}:", exc_info=True)
                        continue
                    
                    # Create proxy server using FastMCP.as_proxy
                    try:
                        proxy_server = FastMCP.as_proxy(
                            client, name=f"{server_name}_proxy"
                        )
                        logger.debug(f"URL proxy server created for {server_name}")
                    except Exception as proxy_error:
                        logger.error(f"Failed to create URL proxy server for {server_name}: {type(proxy_error).__name__}: {proxy_error}")
                        logger.error(f"URL proxy error details for {server_name}:", exc_info=True)
                        continue
                        
                elif server_config.command:
                    # Create client for command-based server
                    # Ensure env is always a dict, never None
                    env_value = getattr(server_config, "env", {})
                    if env_value is None:
                        env_value = {}
                    
                    config_dict = {
                        "mcpServers": {
                            server_name: {
                                "command": server_config.command,
                                "args": getattr(server_config, "args", []),
                                "env": env_value,
                            }
                        }
                    }
                    logger.debug(
                        f"Creating command-based client for {server_name}: {config_dict}"
                    )
                    try:
                        client = Client(config_dict)
                        logger.debug(f"Command client created for {server_name}")
                    except Exception as client_error:
                        logger.error(f"Failed to create command client for {server_name}: {type(client_error).__name__}: {client_error}")
                        logger.error(f"Command client error details for {server_name}:", exc_info=True)
                        continue
                    
                    # Create proxy server using FastMCP.as_proxy
                    try:
                        proxy_server = FastMCP.as_proxy(
                            client, name=f"{server_name}_proxy"
                        )
                        logger.debug(f"Command proxy server created for {server_name}")
                    except Exception as proxy_error:
                        logger.error(f"Failed to create command proxy server for {server_name}: {type(proxy_error).__name__}: {proxy_error}")
                        logger.error(f"Command proxy error details for {server_name}:", exc_info=True)
                        continue
                else:
                    logger.warning(
                        f"Skipping {server_name}: no URL or command specified"
                    )
                    continue

                self.upstream_clients[server_name] = client
                self.proxy_servers[server_name] = proxy_server
                logger.info(
                    f"Created upstream client and proxy server for {server_name}"
                )

                # Test if proxy server is properly initialized
                logger.debug(
                    f"Proxy server for {server_name}: type={type(proxy_server)}, has_get_tools={hasattr(proxy_server, 'get_tools')}"
                )

            except Exception as e:
                logger.error(
                    f"Error creating upstream client/proxy for {server_name}: {type(e).__name__}: {e}"
                )
                logger.error(
                    f"Full exception details for {server_name}:", exc_info=True
                )
                logger.debug(f"Server config: {server_config}")
                # Continue with other servers even if one fails

    async def discover_and_index_tools(self) -> None:
        """Discover tools from all configured servers and index them."""
        logger = get_logger()
        logger.info(f"Starting tool discovery for {len(self.proxy_servers)} servers...")

        try:
            # First, get all current servers from configuration
            current_servers = set(self.proxy_servers.keys())

            # Clean up tools from servers that no longer exist in config
            logger.debug("Cleaning up stale servers...")
            await self._cleanup_stale_servers(current_servers)
        except Exception as e:
            logger.error(f"Error during stale server cleanup: {type(e).__name__}: {e}")
            logger.error("Full stale server cleanup error details:", exc_info=True)
            raise

        # Discover and index tools from current servers
        current_tools = {}  # server_name -> set of tool names
        for server_name, proxy_server in self.proxy_servers.items():
            try:
                logger.debug(f"Starting tool discovery for server: {server_name}")
                tools = await self._discover_server_tools(server_name, proxy_server)
                current_tools[server_name] = set(tools.keys()) if tools else set()
                logger.debug(f"Completed tool discovery for server: {server_name}")
            except Exception as e:
                logger.error(
                    f"Error discovering tools from {server_name}: {type(e).__name__}: {e}"
                )
                logger.error(f"Exception details for {server_name}:", exc_info=True)
                current_tools[server_name] = set()  # Mark as having no tools

        try:
            # Clean up tools that no longer exist on their servers
            logger.debug("Cleaning up stale tools...")
            await self._cleanup_stale_tools(current_tools)
        except Exception as e:
            logger.error(f"Error during stale tool cleanup: {type(e).__name__}: {e}")
            logger.error("Full stale tool cleanup error details:", exc_info=True)
            raise

    async def _cleanup_stale_servers(self, current_servers: set[str]) -> None:
        """Remove tools from servers that no longer exist in configuration."""
        logger = get_logger()

        if not self.persistence:
            return

        # Get all tools from persistence
        all_tools = await self.persistence.get_all_tools()

        # Find servers in database that are not in current config
        db_servers = {tool.server_name for tool in all_tools}
        stale_servers = db_servers - current_servers

        if stale_servers:
            logger.info(
                f"Removing tools from {len(stale_servers)} stale servers: {stale_servers}"
            )
            for server_name in stale_servers:
                await self.persistence.delete_tools_by_server(server_name)
                logger.debug(f"Removed all tools from stale server: {server_name}")

    async def _cleanup_stale_tools(self, current_tools: dict[str, set[str]]) -> None:
        """Remove tools that no longer exist on their servers."""
        logger = get_logger()

        if not self.persistence:
            return

        removed_count = 0
        for server_name, tool_names in current_tools.items():
            # Get tools from database for this server
            db_tools = await self.persistence.get_tools_by_server(server_name)

            # Find tools in database that no longer exist on the server
            db_tool_names = {tool.name for tool in db_tools}
            stale_tool_names = db_tool_names - tool_names

            if stale_tool_names:
                logger.debug(
                    f"Server {server_name}: removing {len(stale_tool_names)} stale tools: {stale_tool_names}"
                )

                # Remove stale tools one by one (no bulk delete by name method)
                for tool in db_tools:
                    if tool.name in stale_tool_names:
                        # Remove from database (this will also handle vector store cleanup if needed)
                        await self._remove_tool_from_persistence(tool)
                        removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} stale tools from database")

    async def _remove_tool_from_persistence(self, tool: ToolMetadata) -> None:
        """Remove a specific tool from persistence layer."""
        # For now, we'll implement this by deleting all tools for the server and re-adding the valid ones
        # This is a simplified approach - a more efficient implementation would delete individual tools
        logger = get_logger()
        logger.debug(
            f"Would remove tool {tool.name} from {tool.server_name} (simplified cleanup)"
        )
        # TODO: Implement individual tool deletion in persistence layer if needed

    async def _discover_server_tools(
        self, server_name: str, proxy_server: FastMCP
    ) -> dict[str, str]:
        """Discover tools from a specific server using its proxy."""
        logger = get_logger()

        try:
            # Ensure indexer is available
            if not self.indexer:
                logger.warning(
                    f"Indexer not available, skipping discovery for {server_name}"
                )
                return {}

            logger.debug(f"Getting tools from proxy server for {server_name}...")

            # Check if proxy server is ready
            if not hasattr(proxy_server, "get_tools"):
                logger.error(
                    f"Proxy server for {server_name} does not have get_tools method"
                )
                return {}

            # Get tools from the proxy server - these are actual Tool objects
            try:
                logger.debug(f"Calling get_tools() on {server_name} proxy server...")
                tools = await proxy_server.get_tools()
                logger.debug(
                    f"get_tools() returned: {type(tools)} with {len(tools) if tools else 0} items"
                )
            except Exception as get_tools_error:
                logger.error(
                    f"Error calling get_tools on {server_name}: {type(get_tools_error).__name__}: {get_tools_error}"
                )
                raise

            if not tools:
                logger.warning(f"No tools returned from {server_name}")
                return {}

            # Store Tool objects in memory and index them
            indexed_count = 0
            tool_names = {}
            for tool_name, tool_obj in tools.items():
                try:
                    # Store Tool object in memory with sanitized key
                    sanitized_key = self._sanitize_tool_name(server_name, tool_name)
                    self.proxified_tools[sanitized_key] = tool_obj

                    logger.debug(f"Indexing tool: {tool_name} from {server_name}")
                    await self.indexer.index_tool_from_object(tool_obj, server_name)
                    indexed_count += 1
                    tool_names[tool_name] = sanitized_key
                except Exception as tool_error:
                    error_msg = str(tool_error)
                    if (
                        "assert d == self.d" in error_msg
                        or "dimension" in error_msg.lower()
                    ):
                        logger.error(
                            f"Vector dimension mismatch for tool {tool_name}. This usually means:"
                        )
                        logger.error(
                            "1. Existing FAISS index has different dimension than current embedder"
                        )
                        logger.error(
                            "2. Solution: Set MCPPROXY_RESET_DATA=true environment variable to reset data"
                        )
                        logger.error("3. Or delete these data files manually:")
                        logger.error("   - tools.faiss (FAISS vector index)")
                        logger.error("   - proxy.db (SQLite database)")

                        # Add BM25-specific info if using BM25
                        if (
                            self.config.embedder == EmbedderType.BM25
                            and self.indexer
                            and hasattr(self.indexer.embedder, "index_dir")
                        ):
                            bm25_dir = self.indexer.embedder.index_dir
                            logger.error(f"   - {bm25_dir}/ (BM25 index directory)")
                        else:
                            logger.error(
                                "   - BM25 index directory (if using BM25: usually a temp dir with bm25s_index/)"
                            )
                        logger.error(
                            "4. For BM25: the index dir is typically a temp directory starting with 'bm25s_'"
                        )
                    logger.error(
                        f"Error indexing tool {tool_name} from {server_name}: {type(tool_error).__name__}: {tool_error}"
                    )
                    logger.error(f"Full stack trace for {tool_name}:", exc_info=True)
                    logger.debug(f"Tool object details: {tool_obj}")

            logger.info(
                f"Successfully indexed {indexed_count}/{len(tools)} tools from {server_name}"
            )
            return tool_names

        except Exception as e:
            logger.error(
                f"Error discovering tools from {server_name}: {type(e).__name__}: {e}"
            )
            logger.error("Full exception details:", exc_info=True)
            return {}

    # Legacy methods for manual initialization (fallback)
    async def initialize_resources(self) -> None:
        """Manual initialization - fallback if lifespan doesn't work."""
        await self._initialize_resources()

    async def cleanup_resources(self) -> None:
        """Manual cleanup - fallback if lifespan doesn't work."""
        await self._cleanup_resources()

    def get_app(self):
        """Get the FastMCP application."""
        return self.mcp
