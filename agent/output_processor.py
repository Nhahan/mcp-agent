from typing import Any, Dict, Union, Tuple

class OutputProcessingError(Exception):
    """Custom exception for errors during output processing."""
    pass

class OutputProcessor:
    """Handles processing and normalization of tool execution results."""

    def process(self, raw_output: Any) -> Dict[str, Any]:
        """
        Processes the raw output from a tool execution into a standardized format.
        This base implementation provides basic handling and normalization.
        Subclasses can override this for tool-specific processing.

        Args:
            raw_output: The raw output received from the tool execution 
                        (e.g., from MultiServerMCPClient.execute_tool).
                        This could be a string, dict, list, error, etc.

        Returns:
            A dictionary containing the processed output, typically including:
            - 'status': 'success' or 'error'
            - 'data': The processed data (e.g., parsed JSON, cleaned text)
            - 'raw': The original raw output
            - 'error_message': Description of the error if status is 'error'
        """
        print(f"\n--- Processing Tool Output --- \nRaw Type: {type(raw_output)}\nRaw Output: {str(raw_output)[:500]}..." ) # Log input

        processed_result: Dict[str, Any] = {
            "status": "success",
            "data": None,
            "raw": raw_output,
            "error_message": None
        }

        try:
            # Basic normalization attempt:
            if isinstance(raw_output, str):
                # Try to parse if it looks like JSON, otherwise keep as string
                try:
                    # Simple check for JSON start/end chars
                    if (raw_output.strip().startswith('{') and raw_output.strip().endswith('}')) or \
                       (raw_output.strip().startswith('[') and raw_output.strip().endswith(']')):
                        parsed_data = json.loads(raw_output)
                        processed_result["data"] = parsed_data
                        print("Processed output as JSON.")
                    else:
                        processed_result["data"] = raw_output # Keep as string
                        print("Processed output as String.")
                except json.JSONDecodeError:
                    # Not valid JSON, treat as plain string
                    processed_result["data"] = raw_output
                    print("Processed output as String (JSON parse failed).")
            elif isinstance(raw_output, (dict, list, int, float, bool)):
                 # Already a usable Python type
                 processed_result["data"] = raw_output
                 print(f"Processed output as {type(raw_output).__name__}.")
            elif raw_output is None:
                 processed_result["data"] = None
                 print("Processed output as None.")
            elif isinstance(raw_output, Exception):
                 # Handle cases where the execution itself returned an Exception
                 processed_result["status"] = "error"
                 processed_result["error_message"] = f"Tool execution resulted in an error: {type(raw_output).__name__}: {str(raw_output)}"
                 processed_result["data"] = None
                 print(f"Error in raw output: {processed_result['error_message']}")
            else:
                 # Unknown type, convert to string
                 print(f"Warning: Unknown raw output type '{type(raw_output).__name__}'. Converting to string.")
                 processed_result["data"] = str(raw_output)

        except Exception as e:
            # Catch errors during the processing itself
            import traceback
            traceback.print_exc()
            error_msg = f"Error processing tool output: {e}"
            print(error_msg)
            processed_result = {
                "status": "error",
                "data": None,
                "raw": raw_output, # Keep original raw output for debugging
                "error_message": error_msg
            }

        print(f"Processing Result: Status='{processed_result['status']}', Data Type={type(processed_result['data']).__name__}")
        return processed_result

# Example Usage
if __name__ == "__main__":
    import json # Ensure json is imported for the example
    processor = OutputProcessor()

    print("--- Testing Output Processor ---")

    test_cases = [
        ("Simple string output", "This is a plain text result."),
        ("JSON string output", '{"key": "value", "number": 123}'),
        ("Malformed JSON string", '{"key": "value", "number": 123'),
        ("Python dictionary output", {"a": 1, "b": [1, 2]}),
        ("Python list output", ["item1", {"sub_item": True}]),
        ("Integer output", 42),
        ("None output", None),
        ("Exception output", ValueError("Tool failed internally")),
        ("Custom object output", object()) # Unknown type
    ]

    for name, test_input in test_cases:
        print(f"\n--- Test Case: {name} ---")
        result = processor.process(test_input)
        print(f"Processed Result: {result}")

        # Basic assertions
        assert 'status' in result
        assert 'raw' in result
        assert result['raw'] == test_input
        if name == "Exception output":
            assert result['status'] == 'error'
            assert result['error_message'] is not None
            assert "ValueError" in result['error_message']
        elif name == "Malformed JSON string":
             assert result['status'] == 'success' # Processing itself succeeds
             assert result['data'] == test_input # Treats as plain string
        elif name == "Custom object output":
             assert result['status'] == 'success'
             assert isinstance(result['data'], str) # Converted to string
        else:
            assert result['status'] == 'success'
            if name == "JSON string output":
                assert isinstance(result['data'], dict)
                assert result['data'] == {"key": "value", "number": 123}
            elif name == "Python dictionary output":
                 assert result['data'] == test_input

    print("\nOutput processor examples finished.") 