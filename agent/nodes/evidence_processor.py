# agent/nodes/evidence_processor.py
from typing import Dict, Any
import logging
import json
from langgraph.graph import END

from ..state import ReWOOState

logger = logging.getLogger(__name__)


async def evidence_processor_node(state: ReWOOState) -> Dict[str, Any]:
    """
    Processes the latest piece of raw evidence.
    If it's a dictionary or list, converts it to a pretty-printed JSON string.
    Otherwise, converts to a plain string.
    Overwrites the original evidence with the processed string version.
    Routes back to tool_selector.
    """
    logger.info("--- Starting Evidence Processor Node ---")
    logger.debug(f"Full evidence from state at start of evidence_processor: {state.get('evidence')}")
    current_step_index_from_state = state.get("current_step_index", 0)
    evidence_dict = state.get("evidence", {}).copy()
    
    all_parsed_steps = state.get("all_parsed_steps")
    tool_invocation_inputs = state.get("current_tool_invocation_inputs") 
    input_key_as_evidence = state.get("use_input_as_evidence")

    processed_step_config_index = current_step_index_from_state - 1 

    if processed_step_config_index < 0 or not all_parsed_steps or processed_step_config_index >= len(all_parsed_steps):
         logger.warning(f"Evidence processor called with invalid processed_step_config_index: {processed_step_config_index}. Current step index from state: {current_step_index_from_state}. all_parsed_steps length: {len(all_parsed_steps) if all_parsed_steps else 'None'}")
         return {
             "evidence": evidence_dict,
             "workflow_status": "failed", 
             "error_message": "Invalid step index for evidence processing.",
             "next_node": END,
             "current_step_index": current_step_index_from_state,
             "use_input_as_evidence": None, # Clear consumed state
             "current_tool_invocation_inputs": None # Clear consumed state
        }

    current_step_details_from_plan = all_parsed_steps[processed_step_config_index]
    # 여기서 가져오는 evidence_variable은 "E1", "E2" 등 #이 없는 형태여야 합니다.
    evidence_var_key_from_plan = current_step_details_from_plan.get("tool_call", {}).get("evidence_variable") 

    if not evidence_var_key_from_plan:
        plan_text = current_step_details_from_plan.get("plan", "N/A") # plan이 없을 경우 대비
        logger.warning(f"No evidence_variable defined in plan for step index {processed_step_config_index} ('{plan_text}'). This might be a thought-only step or a plan error.")
        # 생각 전용 스텝이거나, 도구 호출이 없는 스텝은 현재 evidence를 수정하지 않고 다음으로 넘어갑니다.
        # tool_selector가 다음 스텝을 결정할 것입니다.
        return {
            "evidence": evidence_dict, # 변경 없음
            "workflow_status": "routing_complete",
            "next_node": "tool_selector", 
            "current_step_index": current_step_index_from_state,
            "use_input_as_evidence": None, # Clear consumed state
            "current_tool_invocation_inputs": None # Clear consumed state
        }
    
    # 기본적으로는 도구의 실행 결과 (tool_executor가 evidence_dict에 저장한 값)를 raw_evidence로 간주
    raw_evidence_source = evidence_dict.get(evidence_var_key_from_plan)

    # input_key_as_evidence가 설정되어 있고, tool_invocation_inputs가 있다면 입력값을 증거로 사용
    if input_key_as_evidence and tool_invocation_inputs:
        logger.info(f"Using input parameter '{input_key_as_evidence}' as evidence instead of tool output for {evidence_var_key_from_plan}.")
        raw_evidence_source = tool_invocation_inputs.get(input_key_as_evidence)
        if raw_evidence_source is None:
            logger.warning(f"Specified input key '{input_key_as_evidence}' not found in tool_invocation_inputs for {evidence_var_key_from_plan}. Falling back to tool output if available, or None.")
            raw_evidence_source = evidence_dict.get(evidence_var_key_from_plan) # 폴백 시도 (이미 위에서 한 번 가져왔지만 명시적)
    
    # processed_evidence_str 초기화 (오류 또는 내용으로 채워짐)
    processed_evidence_str = f"Error: Evidence for {evidence_var_key_from_plan} (source: {'input' if input_key_as_evidence and tool_invocation_inputs else 'output'}) could not be determined or was None."
    if raw_evidence_source is None:
        logger.warning(f"Raw evidence source for {evidence_var_key_from_plan} is None.")

    if raw_evidence_source is not None:
        logger.info(f"Processing raw evidence for step {processed_step_config_index + 1} ('{evidence_var_key_from_plan}'), type: {type(raw_evidence_source)}")
        if isinstance(raw_evidence_source, str):
            try:
                parsed_json = json.loads(raw_evidence_source)
                if isinstance(parsed_json, (dict, list)):
                    processed_evidence_str = json.dumps(parsed_json, indent=2)
                    logger.info("Parsed string evidence as JSON and formatted it.")
                else:
                    processed_evidence_str = raw_evidence_source
                    logger.info("Parsed string evidence, but it wasn't dict/list. Using raw string.")
            except json.JSONDecodeError:
                processed_evidence_str = raw_evidence_source
                logger.info("String evidence is not JSON. Using raw string.")
        elif isinstance(raw_evidence_source, (dict, list)):
            try:
                processed_evidence_str = json.dumps(raw_evidence_source, indent=2)
                logger.info("Converted dict/list evidence to formatted JSON string.")
            except TypeError as e:
                logger.warning(f"Could not JSON serialize dict/list evidence for {evidence_var_key_from_plan}, falling back to str(): {e}")
                processed_evidence_str = str(raw_evidence_source)
        else:
            processed_evidence_str = str(raw_evidence_source)
            logger.info(f"Converted evidence of type {type(raw_evidence_source)} to string.")
            
    evidence_dict[evidence_var_key_from_plan] = processed_evidence_str
    logger.info(f"Stored processed evidence for {evidence_var_key_from_plan}: {processed_evidence_str[:200]}...")
    logger.debug(f"Evidence dictionary after processing: {evidence_dict}")
    
    return {
        "evidence": evidence_dict,
        "workflow_status": "routing_complete",
        "next_node": "tool_selector", 
        "current_step_index": current_step_index_from_state, 
        "use_input_as_evidence": None, # Clear consumed state
        "current_tool_invocation_inputs": None # Clear consumed state
    } 