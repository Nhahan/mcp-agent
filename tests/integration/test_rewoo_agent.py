# tests/integration/test_rewoo_agent.py
import pytest
import asyncio
import json
import re
import logging
import os
import warnings

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import agent components
from agent.graph import build_rewoo_graph
from agent.state import ReWOOState

# Import LLM Loader
# Assume core.llm_loader exists and is configured correctly
SKIP_INTEGRATION_TESTS = False
try:
    from core.llm_loader import load_llm as core_load_llm
    print("DEBUG: Successfully imported core.llm_loader.load_llm") # DEBUG 로그 추가
except ImportError:
    print("DEBUG: ImportError occurred during core.llm_loader import.") # DEBUG 로그 추가
    core_load_llm = None
    SKIP_INTEGRATION_TESTS = True
except Exception as e:
    # ADDED: Log the specific exception caught
    print(f"DEBUG: Caught unexpected Exception during core.llm_loader import: {type(e).__name__}: {e}")
    # logger is not defined yet, so using print
    # logger.error(f"Error importing core.llm_loader: {e}", exc_info=True) # Keep original intention
    core_load_llm = None
    SKIP_INTEGRATION_TESTS = True

logger = logging.getLogger(__name__)

# Mark tests as async
pytestmark = pytest.mark.anyio

@pytest.mark.skipif(SKIP_INTEGRATION_TESTS, reason="Skipping integration test due to LLM loader import failure.")
@pytest.mark.asyncio
async def test_rewoo_agent_echo_flow():
    """Integration test for the ReWOO agent with an echo flow."""

    # Use the real LLM
    llm = core_load_llm()

    # Load connections from mcp.json
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_file_path = os.path.join(project_root, 'mcp.json')
    if not os.path.exists(config_file_path):
         pytest.fail(f"MCP config file not found at: {config_file_path}. Ensure it exists in the project root.")

    try:
        with open(config_file_path, 'r') as f:
            mcp_config = json.load(f)
            # Ensure the config format includes transport type if missing
            connections_config = mcp_config.get('servers', {})
            for server_name, config in connections_config.items():
                if 'transport' not in config:
                     # Assuming stdio if not specified, based on previous context
                     connections_config[server_name]['transport'] = 'stdio'
                     logger.warning(f"Transport type not found for server '{server_name}' in mcp.json, defaulting to 'stdio'.")
    except json.JSONDecodeError:
        pytest.fail(f"Error decoding JSON from mcp.json at: {config_file_path}")
    except Exception as e:
        pytest.fail(f"Error reading or processing mcp.json: {e}")


    logger.info(f"Initializing MCP client with connections loaded from {config_file_path}")
    mcp_client = MultiServerMCPClient(connections=connections_config)


    async with mcp_client:  # Manage client lifecycle
        # Ensure tools are loaded
        logger.info("Waiting for MCP server(s) to start...")
        # Increase wait time further for npx execution
        await asyncio.sleep(15)
        # Remove await again, as get_tools() is synchronous according to docs
        logger.info("Calling mcp_client.get_tools() (synchronously)")
        tools = mcp_client.get_tools() # Remove await again
        logger.info(f"Tools received from MCP client: {tools}") # Log the received tools
        if not tools:
            warnings.warn("MCP Client returned no tools. Check server status and mcp.json.")
            pytest.skip("Skipping test because no tools were returned by MCP client.")

        logger.info('Building ReWOO graph with real LLM and mock MCP client...')
        # Build the graph using the real LLM and the mock client
        app, tools_str = await build_rewoo_graph(llm=llm, mcp_client=mcp_client)

        if tools_str.startswith("Error:"):
            pytest.fail(f"Failed to get tool descriptions from mock MCP server: {tools_str}")
        logger.info(f'Tools available from mock server: {tools_str}')

        # 3. Prepare initial state
        test_query = "What is the current time in YYYY-MM-DD HH:mm:ss format?"
        initial_state = ReWOOState(
            original_query=test_query,
            plan=[],
            current_step_index=0,
            tool_name=None,
            tool_input=None,
            evidence=[],
            final_answer=None,
            error_message=None,
            max_retries=1, # Use a reasonable retry limit for tests
            workflow_status='planning',
            tools_str=tools_str,
        )

        # 4. Run the graph
        logger.info(f'Running graph with query: \'{test_query}\'')
        final_state = None
        run_config = RunnableConfig(configurable={"llm": llm, "tools_str": tools_str})
        run_config["recursion_limit"] = 15 # Set recursion limit

        try:
            async for event in app.astream(initial_state, config=run_config):
                for key, value in event.items():
                    logger.info(f'--- Completed Node \'{key}\' --- Status: \'{value.get("workflow_status")}\' --- Step Index: {value.get("current_step_index")}')
                    # Log evidence updates
                    if 'evidence' in value:
                        new_evidence = value['evidence']
                        if len(new_evidence) > len(final_state.get('evidence', []) if final_state else []):
                             logger.info(f'    New Evidence: {new_evidence[-1]}')
                    final_state = value # Keep track of the latest state
            logger.info('Graph execution finished.')
        except Exception as e:
            logger.error(f'Graph execution failed: {e}', exc_info=True)
            pytest.fail(f"Graph execution threw an unexpected exception: {e}")

        # 5. Assert final state
        logger.info('\n--- Asserting Final State ---')
        # Log values before assertion
        logger.info(f"Asserting final_state is not None. Value: {final_state}")
        assert final_state is not None, "Graph did not produce a final state."

        final_status = final_state.get('workflow_status')
        final_error = final_state.get('error_message')
        final_answer_val = final_state.get('final_answer')
        final_evidence = final_state.get('evidence')

        logger.info(f'Final Status: {final_status}')
        logger.info(f'Final Answer: {final_answer_val}')
        logger.info(f'Error Message: {final_error}')
        logger.info(f'Evidence: {final_evidence}')

        logger.info(f"Asserting status == 'finished'. Value: {final_status}")
        assert final_status == "finished", \
               f"Expected status 'finished', got '{final_status}'"
        logger.info(f"Asserting error_message is None. Value: {final_error}")
        assert final_error is None, \
               f"Expected no error message, but got '{final_error}'"
        logger.info(f"Asserting final_answer is not None. Value: {final_answer_val}")
        assert final_answer_val is not None, "Expected a final answer, but got None."

        # Check if the final answer contains a timestamp in the expected format
        time_pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        final_answer_str = final_answer_val if final_answer_val else ""
        logger.info(f"Asserting time pattern in final answer '{final_answer_str}'")
        assert time_pattern.search(final_answer_str), \
               f"Expected a timestamp in YYYY-MM-DD HH:mm:ss format, but got: {final_answer_str}"
        logger.info(f"Successfully found time pattern in final answer: {final_answer_str}")

        # Check if evidence contains a timestamp (might be slightly different format in evidence)
        evidence_str_list = [str(ev) for ev in final_evidence if final_evidence]
        logger.info(f"Asserting time pattern in evidence {evidence_str_list}")
        assert any(time_pattern.search(ev_str) for ev_str in evidence_str_list), \
               f"Expected a timestamp in evidence, but got: {final_evidence}"
        logger.info("Successfully found time pattern in evidence.")

    logger.info('\n--- Integration Test: test_rewoo_agent_echo_flow PASSED ---') 