"""
SequentialThinking 내장형 구현

AI가 복잡한 문제를 해결할 때 단계적으로 사고할 수 있도록 지원합니다.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union, TypedDict
from datetime import datetime
from pathlib import Path

from app.services.react_utils import save_step_log

logger = logging.getLogger(__name__)

class ThoughtData(TypedDict, total=False):
    """단계별 사고 데이터 클래스"""
    thought: str
    thoughtNumber: int
    totalThoughts: int
    isRevision: Optional[bool]
    revisesThought: Optional[int]
    branchFromThought: Optional[int]
    branchId: Optional[str]
    needsMoreThoughts: Optional[bool]
    nextThoughtNeeded: bool

class ThoughtResponse(TypedDict):
    """사고 처리 응답 클래스"""
    thoughtNumber: int
    totalThoughts: int
    nextThoughtNeeded: bool
    branches: List[str]
    thoughtHistoryLength: int

class ErrorResponse(TypedDict):
    """오류 응답 클래스"""
    error: str
    status: str

class SequentialThinking:
    """
    순차적 사고 처리 클래스
    
    복잡한 문제를 단계별로 분석하고 해결하기 위한 내장형 구현입니다.
    """
    
    def __init__(self):
        """SequentialThinking 초기화"""
        self.thought_history: List[ThoughtData] = []
        self.branches: Dict[str, List[ThoughtData]] = {}
        self.session_id: Optional[str] = None
        self.log_dir: Optional[Path] = None
        logger.info("SequentialThinking initialized")
    
    def set_session(self, session_id: str, log_dir: Path) -> None:
        """
        새 세션을 설정합니다.
        
        Args:
            session_id: 세션 ID
            log_dir: 로그 디렉토리 경로
        """
        self.session_id = session_id
        self.log_dir = log_dir
        self.reset()
        logger.info(f"SequentialThinking session set: {session_id}")
    
    def validate_thought_data(self, input_data: Any) -> ThoughtData:
        """
        입력된 사고 데이터의 유효성을 검증합니다.
        
        Args:
            input_data: 검증할 사고 데이터
            
        Returns:
            ThoughtData: 검증된 사고 데이터
            
        Raises:
            ValueError: 유효하지 않은 데이터가 입력된 경우
        """
        if not isinstance(input_data, dict):
            raise ValueError("Invalid input: must be a dictionary")
        
        data = input_data
        
        if "thought" not in data or not isinstance(data["thought"], str):
            raise ValueError("Invalid thought: must be a string")
        
        if "thoughtNumber" not in data or not isinstance(data["thoughtNumber"], int):
            raise ValueError("Invalid thoughtNumber: must be an integer")
        
        if "totalThoughts" not in data or not isinstance(data["totalThoughts"], int):
            raise ValueError("Invalid totalThoughts: must be an integer")
        
        if "nextThoughtNeeded" not in data or not isinstance(data["nextThoughtNeeded"], bool):
            raise ValueError("Invalid nextThoughtNeeded: must be a boolean")
        
        result: ThoughtData = {
            "thought": data["thought"],
            "thoughtNumber": data["thoughtNumber"],
            "totalThoughts": data["totalThoughts"],
            "nextThoughtNeeded": data["nextThoughtNeeded"],
        }
        
        # 선택적 필드 처리
        if "isRevision" in data and isinstance(data["isRevision"], bool):
            result["isRevision"] = data["isRevision"]
        
        if "revisesThought" in data and isinstance(data["revisesThought"], int):
            result["revisesThought"] = data["revisesThought"]
        
        if "branchFromThought" in data and isinstance(data["branchFromThought"], int):
            result["branchFromThought"] = data["branchFromThought"]
        
        if "branchId" in data and isinstance(data["branchId"], str):
            result["branchId"] = data["branchId"]
        
        if "needsMoreThoughts" in data and isinstance(data["needsMoreThoughts"], bool):
            result["needsMoreThoughts"] = data["needsMoreThoughts"]
        
        return result
    
    def format_thought(self, thought_data: ThoughtData) -> str:
        """
        사고 데이터를 포맷팅하여 문자열로 반환합니다.
        
        Args:
            thought_data: 포맷팅할 사고 데이터
            
        Returns:
            str: 포맷팅된 사고 문자열
        """
        thought_number = thought_data["thoughtNumber"]
        total_thoughts = thought_data["totalThoughts"]
        thought = thought_data["thought"]
        
        prefix = ""
        context = ""
        
        if thought_data.get("isRevision"):
            prefix = "🔄 Revision"
            context = f" (revising thought {thought_data.get('revisesThought')})"
        elif thought_data.get("branchFromThought"):
            prefix = "🌿 Branch"
            context = f" (from thought {thought_data.get('branchFromThought')}, ID: {thought_data.get('branchId')})"
        else:
            prefix = "💭 Thought"
            context = ""
        
        header = f"{prefix} {thought_number}/{total_thoughts}{context}"
        border = "─" * max(len(header), len(thought) + 2)
        
        return f"""
┌{border}┐
│ {header} │
├{border}┤
│ {thought.ljust(len(border) - 2)} │
└{border}┘"""
    
    def process_thought(self, input_data: Any) -> Dict[str, Any]:
        """
        사고 데이터를 처리하고 결과를 반환합니다.
        
        Args:
            input_data: 처리할 사고 데이터
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            validated_input = self.validate_thought_data(input_data)
            
            # 사고 번호가 총 사고 수보다 크면 총 사고 수를 조정
            if validated_input["thoughtNumber"] > validated_input["totalThoughts"]:
                validated_input["totalThoughts"] = validated_input["thoughtNumber"]
            
            # 사고 기록에 추가
            self.thought_history.append(validated_input)
            
            # 분기 처리
            branch_from = validated_input.get("branchFromThought")
            branch_id = validated_input.get("branchId")
            
            if branch_from and branch_id:
                if branch_id not in self.branches:
                    self.branches[branch_id] = []
                self.branches[branch_id].append(validated_input)
            
            # 포맷팅된 사고 로깅
            formatted_thought = self.format_thought(validated_input)
            logger.info(f"Processing thought #{validated_input['thoughtNumber']}:\n{formatted_thought}")
            
            # 응답 생성
            response: ThoughtResponse = {
                "thoughtNumber": validated_input["thoughtNumber"],
                "totalThoughts": validated_input["totalThoughts"],
                "nextThoughtNeeded": validated_input["nextThoughtNeeded"],
                "branches": list(self.branches.keys()),
                "thoughtHistoryLength": len(self.thought_history)
            }
            
            # 결과 생성
            result = {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response, indent=2)
                }]
            }
            
            # 로그 저장
            # if self.log_dir and self.session_id:
            #     log_data = {
            #         "timestamp": datetime.now().isoformat(),
            #         "session_id": self.session_id,
            #         "thought_number": validated_input["thoughtNumber"],
            #         "total_thoughts": validated_input["totalThoughts"],
            #         "thought": validated_input["thought"],
            #         "formatted_thought": formatted_thought,
            #         "is_revision": validated_input.get("isRevision", False),
            #         "revises_thought": validated_input.get("revisesThought"),
            #         "branch_from": branch_from,
            #         "branch_id": branch_id,
            #         "next_thought_needed": validated_input["nextThoughtNeeded"],
            #         "response": response
            #     }
                
            #     try:
            #         save_step_log(
            #             self.log_dir, 
            #             validated_input["thoughtNumber"],
            #             f"sequential_thinking", 
            #             log_data
            #         )
            #         logger.debug(f"Saved SequentialThinking log for thought #{validated_input['thoughtNumber']}")
            #     except Exception as e:
            #         logger.error(f"Failed to save SequentialThinking log: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing thought: {e}")
            error_response: ErrorResponse = {
                "error": str(e),
                "status": "failed"
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(error_response, indent=2)
                }],
                "isError": True
            }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        SequentialThinking 도구 정의를 반환합니다.
        
        Returns:
            Dict[str, Any]: 도구 정의
        """
        return {
            "name": "sequentialthinking",
            "description": """A detailed tool for dynamic and reflective problem-solving through thoughts.
This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
Each thought can build on, question, or revise previous insights as understanding deepens.

When to use this tool:
- Breaking down complex problems into steps
- Planning and design with room for revision
- Analysis that might need course correction
- Problems where the full scope might not be clear initially
- Problems that require a multi-step solution
- Tasks that need to maintain context over multiple steps
- Situations where irrelevant information needs to be filtered out

Key features:
- You can adjust total_thoughts up or down as you progress
- You can question or revise previous thoughts
- You can add more thoughts if needed, even at the "end"
- You can express uncertainty and explore alternative approaches
- Not every thought needs to build linearly - you can branch or backtrack
- Generates a solution hypothesis
- Verifies the hypothesis based on the Chain of Thought steps
- Repeats the process until satisfied
- Provides a correct answer

Parameters explained:
- thought: Your current thinking step, which can include:
* Regular analytical steps
* Revisions of previous thoughts
* Questions about previous decisions
* Realizations about needing more analysis
* Changes in approach
* Hypothesis generation
* Hypothesis verification
- next_thought_needed: True if you need more thinking, even if at what seemed like the end
- thought_number: Current number in sequence (can go beyond initial total if needed)
- total_thoughts: Current estimate of thoughts needed (can be adjusted up/down)
- is_revision: A boolean indicating if this thought revises previous thinking
- revises_thought: If is_revision is true, which thought number is being reconsidered
- branch_from_thought: If branching, which thought number is the branching point
- branch_id: Identifier for the current branch (if any)
- needs_more_thoughts: If reaching end but realizing more thoughts needed

You should:
1. Start with an initial estimate of needed thoughts, but be ready to adjust
2. Feel free to question or revise previous thoughts
3. Don't hesitate to add more thoughts if needed, even at the "end"
4. Express uncertainty when present
5. Mark thoughts that revise previous thinking or branch into new paths
6. Ignore information that is irrelevant to the current step
7. Generate a solution hypothesis when appropriate
8. Verify the hypothesis based on the Chain of Thought steps
9. Repeat the process until satisfied with the solution
10. Provide a single, ideally correct answer as the final output
11. Only set next_thought_needed to false when truly done and a satisfactory answer is reached""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your current thinking step"
                    },
                    "nextThoughtNeeded": {
                        "type": "boolean",
                        "description": "Whether another thought step is needed"
                    },
                    "thoughtNumber": {
                        "type": "integer",
                        "description": "Current thought number",
                        "minimum": 1
                    },
                    "totalThoughts": {
                        "type": "integer",
                        "description": "Estimated total thoughts needed",
                        "minimum": 1
                    },
                    "isRevision": {
                        "type": "boolean",
                        "description": "Whether this revises previous thinking"
                    },
                    "revisesThought": {
                        "type": "integer",
                        "description": "Which thought is being reconsidered",
                        "minimum": 1
                    },
                    "branchFromThought": {
                        "type": "integer",
                        "description": "Branching point thought number",
                        "minimum": 1
                    },
                    "branchId": {
                        "type": "string",
                        "description": "Branch identifier"
                    },
                    "needsMoreThoughts": {
                        "type": "boolean",
                        "description": "If more thoughts are needed"
                    }
                },
                "required": ["thought", "nextThoughtNeeded", "thoughtNumber", "totalThoughts"]
            }
        }

    def reset(self) -> None:
        """
        사고 기록과 분기를 초기화합니다.
        """
        self.thought_history = []
        self.branches = {}
        logger.info("SequentialThinking reset")

# 싱글톤 인스턴스
_sequential_thinking = None

def get_sequential_thinking() -> SequentialThinking:
    """
    SequentialThinking 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        SequentialThinking: 싱글톤 인스턴스
    """
    global _sequential_thinking
    if _sequential_thinking is None:
        _sequential_thinking = SequentialThinking()
    return _sequential_thinking 


## Below code is the original code for the SequentialThinking MCP Server tool. Above code is the implementation of the original code in Python for the SequentialThinking MCP Server tool to be used in this project.
## To know what is MCP Server, please refer to the following markdown file. `root/GUIDE.md``
## it is not used in the current implementation.
## it is kept here for future reference.

# import { Server } from "@modelcontextprotocol/sdk/server/index.js";
# import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
# import {
#   CallToolRequestSchema,
#   ListToolsRequestSchema,
#   Tool,
# } from "@modelcontextprotocol/sdk/types.js";
# // Fixed chalk import for ESM
# import chalk from 'chalk';

# interface ThoughtData {
#   thought: string;
#   thoughtNumber: number;
#   totalThoughts: number;
#   isRevision?: boolean;
#   revisesThought?: number;
#   branchFromThought?: number;
#   branchId?: string;
#   needsMoreThoughts?: boolean;
#   nextThoughtNeeded: boolean;
# }

# class SequentialThinkingServer {
#   private thoughtHistory: ThoughtData[] = [];
#   private branches: Record<string, ThoughtData[]> = {};

#   private validateThoughtData(input: unknown): ThoughtData {
#     const data = input as Record<string, unknown>;

#     if (!data.thought || typeof data.thought !== 'string') {
#       throw new Error('Invalid thought: must be a string');
#     }
#     if (!data.thoughtNumber || typeof data.thoughtNumber !== 'number') {
#       throw new Error('Invalid thoughtNumber: must be a number');
#     }
#     if (!data.totalThoughts || typeof data.totalThoughts !== 'number') {
#       throw new Error('Invalid totalThoughts: must be a number');
#     }
#     if (typeof data.nextThoughtNeeded !== 'boolean') {
#       throw new Error('Invalid nextThoughtNeeded: must be a boolean');
#     }

#     return {
#       thought: data.thought,
#       thoughtNumber: data.thoughtNumber,
#       totalThoughts: data.totalThoughts,
#       nextThoughtNeeded: data.nextThoughtNeeded,
#       isRevision: data.isRevision as boolean | undefined,
#       revisesThought: data.revisesThought as number | undefined,
#       branchFromThought: data.branchFromThought as number | undefined,
#       branchId: data.branchId as string | undefined,
#       needsMoreThoughts: data.needsMoreThoughts as boolean | undefined,
#     };
#   }

#   private formatThought(thoughtData: ThoughtData): string {
#     const { thoughtNumber, totalThoughts, thought, isRevision, revisesThought, branchFromThought, branchId } = thoughtData;

#     let prefix = '';
#     let context = '';

#     if (isRevision) {
#       prefix = chalk.yellow('🔄 Revision');
#       context = ` (revising thought ${revisesThought})`;
#     } else if (branchFromThought) {
#       prefix = chalk.green('🌿 Branch');
#       context = ` (from thought ${branchFromThought}, ID: ${branchId})`;
#     } else {
#       prefix = chalk.blue('💭 Thought');
#       context = '';
#     }

#     const header = `${prefix} ${thoughtNumber}/${totalThoughts}${context}`;
#     const border = '─'.repeat(Math.max(header.length, thought.length) + 4);

#     return `
# ┌${border}┐
# │ ${header} │
# ├${border}┤
# │ ${thought.padEnd(border.length - 2)} │
# └${border}┘`;
#   }

#   public processThought(input: unknown): { content: Array<{ type: string; text: string }>; isError?: boolean } {
#     try {
#       const validatedInput = this.validateThoughtData(input);

#       if (validatedInput.thoughtNumber > validatedInput.totalThoughts) {
#         validatedInput.totalThoughts = validatedInput.thoughtNumber;
#       }

#       this.thoughtHistory.push(validatedInput);

#       if (validatedInput.branchFromThought && validatedInput.branchId) {
#         if (!this.branches[validatedInput.branchId]) {
#           this.branches[validatedInput.branchId] = [];
#         }
#         this.branches[validatedInput.branchId].push(validatedInput);
#       }

#       const formattedThought = this.formatThought(validatedInput);
#       console.error(formattedThought);

#       return {
#         content: [{
#           type: "text",
#           text: JSON.stringify({
#             thoughtNumber: validatedInput.thoughtNumber,
#             totalThoughts: validatedInput.totalThoughts,
#             nextThoughtNeeded: validatedInput.nextThoughtNeeded,
#             branches: Object.keys(this.branches),
#             thoughtHistoryLength: this.thoughtHistory.length
#           }, null, 2)
#         }]
#       };
#     } catch (error) {
#       return {
#         content: [{
#           type: "text",
#           text: JSON.stringify({
#             error: error instanceof Error ? error.message : String(error),
#             status: 'failed'
#           }, null, 2)
#         }],
#         isError: true
#       };
#     }
#   }
# }

# const SEQUENTIAL_THINKING_TOOL: Tool = {
#   name: "sequentialthinking",
#   description: `A detailed tool for dynamic and reflective problem-solving through thoughts.
# This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
# Each thought can build on, question, or revise previous insights as understanding deepens.

# When to use this tool:
# - Breaking down complex problems into steps
# - Planning and design with room for revision
# - Analysis that might need course correction
# - Problems where the full scope might not be clear initially
# - Problems that require a multi-step solution
# - Tasks that need to maintain context over multiple steps
# - Situations where irrelevant information needs to be filtered out

# Key features:
# - You can adjust total_thoughts up or down as you progress
# - You can question or revise previous thoughts
# - You can add more thoughts even after reaching what seemed like the end
# - You can express uncertainty and explore alternative approaches
# - Not every thought needs to build linearly - you can branch or backtrack
# - Generates a solution hypothesis
# - Verifies the hypothesis based on the Chain of Thought steps
# - Repeats the process until satisfied
# - Provides a correct answer

# Parameters explained:
# - thought: Your current thinking step, which can include:
# * Regular analytical steps
# * Revisions of previous thoughts
# * Questions about previous decisions
# * Realizations about needing more analysis
# * Changes in approach
# * Hypothesis generation
# * Hypothesis verification
# - next_thought_needed: True if you need more thinking, even if at what seemed like the end
# - thought_number: Current number in sequence (can go beyond initial total if needed)
# - total_thoughts: Current estimate of thoughts needed (can be adjusted up/down)
# - is_revision: A boolean indicating if this thought revises previous thinking
# - revises_thought: If is_revision is true, which thought number is being reconsidered
# - branch_from_thought: If branching, which thought number is the branching point
# - branch_id: Identifier for the current branch (if any)
# - needs_more_thoughts: If reaching end but realizing more thoughts needed

# You should:
# 1. Start with an initial estimate of needed thoughts, but be ready to adjust
# 2. Feel free to question or revise previous thoughts
# 3. Don't hesitate to add more thoughts if needed, even at the "end"
# 4. Express uncertainty when present
# 5. Mark thoughts that revise previous thinking or branch into new paths
# 6. Ignore information that is irrelevant to the current step
# 7. Generate a solution hypothesis when appropriate
# 8. Verify the hypothesis based on the Chain of Thought steps
# 9. Repeat the process until satisfied with the solution
# 10. Provide a single, ideally correct answer as the final output
# 11. Only set next_thought_needed to false when truly done and a satisfactory answer is reached`,
#   inputSchema: {
#     type: "object",
#     properties: {
#       thought: {
#         type: "string",
#         description: "Your current thinking step"
#       },
#       nextThoughtNeeded: {
#         type: "boolean",
#         description: "Whether another thought step is needed"
#       },
#       thoughtNumber: {
#         type: "integer",
#         description: "Current thought number",
#         minimum: 1
#       },
#       totalThoughts: {
#         type: "integer",
#         description: "Estimated total thoughts needed",
#         minimum: 1
#       },
#       isRevision: {
#         type: "boolean",
#         description: "Whether this revises previous thinking"
#       },
#       revisesThought: {
#         type: "integer",
#         description: "Which thought is being reconsidered",
#         minimum: 1
#       },
#       branchFromThought: {
#         type: "integer",
#         description: "Branching point thought number",
#         minimum: 1
#       },
#       branchId: {
#         type: "string",
#         description: "Branch identifier"
#       },
#       needsMoreThoughts: {
#         type: "boolean",
#         description: "If more thoughts are needed"
#       }
#     },
#     required: ["thought", "nextThoughtNeeded", "thoughtNumber", "totalThoughts"]
#   }
# };

# const server = new Server(
#   {
#     name: "sequential-thinking-server",
#     version: "0.2.0",
#   },
#   {
#     capabilities: {
#       tools: {},
#     },
#   }
# );

# const thinkingServer = new SequentialThinkingServer();

# server.setRequestHandler(ListToolsRequestSchema, async () => ({
#   tools: [SEQUENTIAL_THINKING_TOOL],
# }));

# server.setRequestHandler(CallToolRequestSchema, async (request) => {
#   if (request.params.name === "sequentialthinking") {
#     return thinkingServer.processThought(request.params.arguments);
#   }

#   return {
#     content: [{
#       type: "text",
#       text: `Unknown tool: ${request.params.name}`
#     }],
#     isError: true
#   };
# });

# async function runServer() {
#   const transport = new StdioServerTransport();
#   await server.connect(transport);
#   console.error("Sequential Thinking MCP Server running on stdio");
# }

# runServer().catch((error) => {
#   console.error("Fatal error running server:", error);
#   process.exit(1);
# });
