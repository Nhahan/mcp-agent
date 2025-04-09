# MCP Agent Development Guide

## Project Overview

This project implements an AI agent that can utilize [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/spec) servers to extend its capabilities. The AI agent is designed to be:

1. **Web-based**: Runs in a web container
2. **Lightweight**: Uses the ONNX runtime with a quantized model
3. **Extensible**: Can leverage any MCP server defined in `mcp.json`
4. **Language-agnostic**: Primarily operates in English internally, but can respond in the user's language

## Core Components

- **InferenceService**: Handles model loading, inference, and the ReAct pattern implementation
- **MCPService**: Manages MCP server connections and tool execution
- **API Layer**: FastAPI endpoints that handle requests and responses

## Key Features

### Language Handling and Translation Flow

1. **Input Processing**:
   - Detect the language of user input 
   - If Korean (ko), translate to English before sending to the AI model
   - For other languages, pass through as-is

2. **Internal Processing**:
   - AI operates entirely in English, regardless of input language
   - All prompts, system instructions, and tool descriptions are in English
   - ReAct pattern reasoning is performed in English

3. **Output Processing**:
   - If original input was Korean, translate the AI's response back to Korean
   - For other languages, return the AI's English response as-is

This approach ensures consistent reasoning quality while providing a natural user experience in different languages.

### MCP Server Integration

- The AI agent uses the MCP protocol to interact with external tools
- MCP servers are defined in `mcp.json` and loaded dynamically
- No hardcoded MCP servers - all tools must be discovered at runtime
- Tool schemas are cached to improve performance and reduce latency

### ReAct Pattern Implementation

The AI agent uses the ReAct (Reasoning + Acting) pattern:

1. **Thought**: The AI analyzes the query, available tools, and previous observations
2. **Action**: The AI either calls a tool or provides a final answer
3. **Observation**: Results from tool execution are provided back to the AI
4. **Repeat**: The process continues until a final answer is reached

### Logging System

- All interactions are logged to a centralized `meta.json` file per session
- Logs include user queries, AI responses, tool calls, and observations
- Logs are stored in `logs/api_logs/{session_id}/meta.json`

## Development Rules

1. **No Hardcoding of MCP Servers**:
   - The system must detect and use MCP servers defined in `mcp.json`
   - The user can add or remove MCP servers at any time

2. **English-Only Internal Processing**:
   - All system prompts and internal reasoning must be in English
   - Translation must happen at the API boundary, not within the core logic

3. **Simplified Tool Descriptions**:
   - Initially provide only a list of available tools with brief descriptions
   - Provide detailed tool information only when a specific tool is selected

4. **Centralized Logging**:
   - All logs for a session must be stored in a single `meta.json` file
   - No file fragmentation or redundant logs

5. **Proper Error Handling**:
   - All errors must be properly captured and formatted for the user
   - Failed tool executions should not crash the system

6. **No Test Mocking**:
   - Tests must run against the actual system, not mocked components
   - Tests must not be skipped and should reflect real production scenarios

## Model Requirements

- The project uses a quantized ONNX model: `gemma-3-1b-it-ONNX/model_q4f16.onnx`
- The model is ~1GB in size and should be downloaded at runtime, not hosted on GitHub
- The model should be loaded into memory only when needed

## Code Structure and Standards

- Code should be well-structured and maintainable
- Follow proper error handling and logging practices
- Avoid duplicate code and ensure good separation of concerns
- All user-facing messages should adapt to the user's language
- Internal system messages should be in English

## How to Test

- Run tests to ensure all components work correctly
- Check logs to verify proper flow of information
- Test with different languages to ensure translation works correctly
- Test with various MCP servers to ensure tool discovery and execution works

## Important Notes

- The agent must be able to run in a web container
- The agent must be OS-agnostic
- The agent must dynamically adapt to available MCP servers
- Always follow the ReAct pattern for reasoning and tool use 