# ReWOO (Reasoning Without Observation) Pattern and LangGraph Implementation Summary

## ReWOO Pattern Core Concepts

*   **Flow:** Plan -> Execute -> Solve.
    1.  **Plan:** Generate a multi-step plan upfront based on the task and available tools. Use variables (e.g., `#E1`, `#E2`) to represent intermediate results (evidence) needed for subsequent steps.
    2.  **Execute:** Sequentially execute each step of the plan using the designated tools. Substitute the `#E` variables with the actual results from previous steps.
    3.  **Solve:** Once all steps are executed and evidence is collected, synthesize the final answer using the original task, the plan, and all collected evidence.
*   **Components:**
    *   **Planner (LLM):** Creates the initial plan, defining thoughts, tool calls (with variable placeholders), and expected outcomes for each step.
    *   **Executor (Worker):** Executes tools according to the plan, handling variable substitution.
    *   **Solver (LLM):** Generates the final answer based on the complete context (task, plan, evidence).
*   **Advantages:**
    *   Reduces LLM calls compared to ReAct, potentially saving tokens and time.
    *   Planner can be trained/fine-tuned without needing actual tool execution results during training (in theory).
*   **Limitations:**
    *   Planner effectiveness depends on initial context; may require few-shot examples or fine-tuning.
    *   Execution is sequential, limiting parallelism.

## LangGraph Implementation Notes

*   **State Management:** Use `StateGraph` and define a state schema (e.g., using `TypedDict`) to hold information like the task, plan (as string and parsed steps), current step index, execution results/evidence, and final answer.
*   **Nodes:**
    *   `Planner Node`: Takes the initial task, calls an LLM with a specific ReWOO planning prompt, parses the output plan (e.g., using regex or expecting structured output like JSON), and updates the state with the plan details (`plan_string`, `steps`).
    *   `Executor Node`: Determines the current step, performs variable substitution (`#E1`, `#E2`, etc.) in the tool input using previously stored results, executes the tool (often via another LLM call or a direct tool invocation like a search API), and stores the output in the state (e.g., a `results` dictionary mapped by `#E` variables).
    *   `Solver Node`: Takes the final state (including the full plan and all results), formats them into a prompt for the Solver LLM, and generates the final answer, updating the state.
*   **Edges & Routing:** Define edges to control the flow: `START` -> `plan` -> `tool` -> (conditional loop back to `tool` or proceed to `solve`) -> `solve` -> `END`. Conditional logic checks if all plan steps have been executed.
*   **Variable Substitution:** Implement logic within the Executor node to replace `#E<n>` placeholders in tool arguments with the corresponding results stored in the state.
*   **Parsing:** Robust parsing of the Planner LLM's output is crucial. This might involve regex for specific formats (like `Plan: ... #E1 = Tool[Input]`) or expecting structured output (like JSON). Errors in parsing are a common failure point.
