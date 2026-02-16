**Advanced AutoBot Architecture — Detailed Overview**

This document explains the content, flow, failure modes, inputs/outputs, and design rationale of `autobot/architecture.py` in this repository. It also documents recent hardening changes and recommended next steps for testing and production use.

**Purpose:**
- Describe the multi-agent, blackboard-driven orchestrator implemented in `architecture.py`.
- Explain how data flows between agents, what each agent does, expected inputs/outputs, and fallback behaviors.
- Provide diagrams and JSON examples for model outputs used by the pipeline.

**File:** [autobot/architecture.py](autobot/architecture.py)

**High-level summary**
- Single-file orchestrator that runs a sequence of agents to decompose a user query, gather evidence, reason, and synthesize a final answer.
- Uses a blackboard (`AgentContext.blackboard`) and `AgentContext` as the shared state for the run.
- Agents interact by posting messages with `ctx.post(sender, recipient, content)` and by mutating the `ctx` fields.
- Model calls are abstracted by `HFModelPool` and tool calls by `ToolBox`.

**Primary components**
- `Settings`: runtime configuration (model names, timeouts, memory file).
- `AgentContext`: the shared run-time context. Holds query, plan, observations, messages, blackboard artifacts, memory recall, warnings/errors, etc.
- `HFModelPool`: adapter to call Hugging Face Inference client (or return mocks when unavailable).
- `ToolBox`: helper to run web searches and scrapers.
- `MemoryStore`: persistent episodic/semantic/graph memory (file-backed JSON).
- `BaseAgent` and concrete agents:
  - `IntentParsingAgent`, `MetaControllerAgent`, `PlanningAgent`, `GoalDecompositionAgent`, `TreeOfThoughtsAgent`, `ToolUseAgent`, `ReActAgent`, `BlackboardAgent`, `PEVAgent`, `EnsembleAgent`, `GraphAgent`, `MentalLoopAgent`, `DryRunAgent`, `RLHFAgent`, `ReflexiveMetacognitiveAgent`, and supporting coordinator agents.
- `AdvancedAutoBotArchitecture`: orchestrator that wires agents and runs staged/parallel groups.

**Data flow and lifecycle (simplified):**
1. Run started with `AdvancedAutoBotArchitecture.run(query)` creating `AgentContext(query=...)`.
2. `IntentParsingAgent` parses the query (via model) into structured intent JSON.
3. `MetaControllerAgent` sets a route and selects which capabilities and stages to run.
4. `PlanningAgent` produces a numbered plan; `GoalDecompositionAgent` and `TreeOfThoughtsAgent` can add candidate plans.
5. `MultiAgentCoordinator` assigns agents to subtasks.
6. `ToolUseAgent` and `ReActAgent` gather evidence (web search, scraping, tool calls), update `ctx.observations` and `ctx.sources`.
7. `BlackboardAgent`, `PEVAgent`, and `DryRunAgent` verify evidence and simulate risk.
8. `EnsembleAgent`, `RLHFAgent`, and `ReflexiveMetacognitiveAgent` refine and critique the final answer.
9. `EpisodicSemanticAgent` writes memory; orchestration returns `AgentContext` with `final_answer` and artifacts.

**Mermaid flowchart**

```mermaid
flowchart TD
  User[User Query]
  Intent[IntentParsingAgent]\n(parse -> JSON)
  Meta[MetaControllerAgent]\n(select route)
  Plan[PlanningAgent]\n(plan steps)
  ParallelGroup{Parallel Analysis?}
  Tool[ToolUseAgent]
  ReAct[ReActAgent]
  Blackboard[BlackboardAgent]
  PEV[PEVAgent]
  Ensemble[EnsembleAgent]
  Synth[SynthesisEngine]\n(synthesize final answer)
  Memory[EpisodicSemanticAgent]\n(save)
  User --> Intent --> Meta --> Plan --> ParallelGroup
  ParallelGroup -->|yes| Tool & ReAct & Ensemble & Graph
  ParallelGroup -->|no| Tool --> ReAct
  ReAct --> Blackboard --> PEV --> Synth
  Ensemble --> Synth
  Synth --> Reflexive[ReflexiveMetacognitiveAgent] --> Memory --> User
```

**Agent responsibilities and outputs (concise)**
- IntentParsingAgent
  - Input: `ctx.query` and memory context
  - Output: structured `ctx.intent_analysis` and `ctx.task_probabilities`
  - Expected model output: strict JSON containing keys such as `primary_task`, `secondary_tasks`, `task_probabilities`, `intent_tags`, `detected_capabilities`, `entities`, `urgency`, `ambiguity_score`, `requires_clarification`, `missing_information`, `clarification_questions`, `reasoning_summary`.

- MetaControllerAgent
  - Input: `intent_analysis`
  - Output: `ctx.route`, booleans indicating which stages to run (tool_use, tree_of_thoughts, ensemble, etc.)

- PlanningAgent
  - Input: `ctx.query`, `ctx.route`, memory summaries
  - Output: `ctx.plan` (string) and `ctx.plan_steps` (list)

- ToolUseAgent & ReActAgent
  - Input: plan, query
  - Output: `ctx.observations`, `ctx.sources`, `ctx.blackboard.artifacts` containing evidence

- PEVAgent
  - Input: observations & plan
  - Output: verification status `ctx.pev_verified` and coverage metrics

- EnsembleAgent, RLHFAgent, ReflexiveMetacognitiveAgent
  - Input: draft answer & supporting artifacts
  - Output: refined `ctx.final_answer`, `ctx.confidence`, and memory facts

**Model output expectations and validation**
- Model responses are often noisy. This repository contains helpers to robustly parse common shapes:
  - `_strip_code_fence(text)` removes ```json fences.
  - `_balanced_json_slices(text)` extracts balanced `{...}` or `[...]` substrings.
  - `_first_json(text)` tries multiple fallbacks (direct, code fence, balanced slices) to produce a JSON payload.
  - `ModelSchemaAdapter.extract_text(response)` normalizes varied HF response shapes (dicts with `generated_text`, `choices`, streaming deltas, Pydantic models, etc.) into a single string.
  - Schema validators (`ModelSchemaAdapter.validate_intent_payload(...)`, etc.) exist to verify required fields and surface issues as lists of strings.

**Recent hardening changes applied**
- `AgentContext.post()` is now thread-safe: a `threading.Lock` (field `ctx.lock`) protects mutations to `messages`, `trace`, and `blackboard['messages']`.
  - Why: previously concurrent writes from parallel stages could corrupt lists or interleave messages.
- `_run_parallel_group()` now executes each parallel stage on a `deepcopy(ctx)` and merges safe artifacts back under `ctx.lock`.
  - Why: agents mutate `ctx` concurrently. Running on isolated snapshots prevents race conditions and allows controlled merging of artifacts (messages, warnings, errors, observations, artifacts).
  - Merge behavior: messages, warnings, errors, observations, fallback_events, executed/skipped stages, and blackboard artifacts/status/messages are merged. Plan is set from a stage if main `ctx.plan` is empty.

**Fallback behaviors and error handling**
- If JSON parsing fails for critical agents (Intent.Parser, ToT, GoalDecomposition), fallback heuristics are used:
  - `_fallback_intent_inference(query, memory_recall)` provides a safe intent structure derived from token similarity and default heuristics.
  - For ToT/branches/subtasks, empty or malformed outputs revert to conservative defaults (single-step plan, simple subtask list).
- Model client failures (`HFModelPool._client` or `.generate`) fall back to `_mock()` which returns safe placeholder responses. These are logged and added to `ctx.fallback_events`.
- Memory persistence errors in `MemoryStore.save()` are recorded as warnings and do not crash the orchestrator; subsequent attempts or retries should be scheduled.
- When a stage raises an exception within `_run_stage()`, `_record_error()` records the failure and `_replan_after_error()` tries to generate a recovery plan using the planning model. If replan fails, orchestrator emits a fallback final answer via `_fallback_answer_from_context()`.

**Concurrent execution notes**
- Parallel analysis is only enabled when the execution profile deems the task `complexity == 'hard'` and several gates are true.
- Because agents can still perform I/O (web requests) and model calls, thread pools are bounded to the number of parallel stages to avoid resource exhaustion.
- The merging strategy after parallel stage completion is intentionally conservative: only append lists and shallow-merge blackboard artifacts/status. Avoid merging complex nested structures blindly.

**Inputs & Outputs (examples)**

- Example user input (CLI):

  ```text
  --query "Find and summarize known solutions to handle intermittent API 429 errors when scraping a public endpoint."
  ```

- Example expected `IntentParsingAgent` JSON output (ideal):

  ```json
  {
    "primary_task": "debugging",
    "secondary_tasks": ["data_gathering","analysis"],
    "task_probabilities": {"debugging":0.7,"analysis":0.2,"data_gathering":0.1},
    "intent_tags": ["debugging","data_gathering"],
    "detected_capabilities": ["Automated Debugging","Deep Web Research"],
    "entities": [{"type":"service","value":"public endpoint"}],
    "urgency":"medium",
    "ambiguity_score":0.12,
    "requires_clarification":false,
    "missing_information":[],
    "clarification_questions":[],
    "reasoning_summary":"User asks for strategies to mitigate intermittent 429 rate-limit errors while scraping."
  }
  ```

- Example final answer artifact (simplified):

  ```json
  {
    "final_answer": "Common approaches: (1) exponential backoff + jitter, (2) respect Retry-After header, (3) use rotating proxies/credentials, (4) rate limit client-side and cache responses. Example code snippet: ...",
    "confidence": 0.78,
    "sources": [{"title":"Example blog","url":"https://example.com/rate-limits"}],
    "artifacts": {"plan":"1. Gather sources\n2. Recommend approaches\n3. Provide code snippets"}
  }
  ```

**Why the design choices were made**
- Blackboard + AgentContext: promotes decoupling of responsibilities, easy inspection of run-time state, and extensibility when adding agents.
- Per-agent small responsibility: simpler reasoning about failures and targeted fallbacks.
- Deepcopy-per-stage for parallel execution: practical trade-off to avoid hard-to-debug races while keeping merge logic explicit and auditable.
- Multiple JSON extraction heuristics: models frequently return text wrappers, fences, or partial JSON — multi-step extraction increases robustness.

**Limitations & risks**
- Deepcopying `AgentContext` can be expensive for very large contexts; consider snapshotting only required fields for heavy runs.
- Merging results naively may cause duplicate messages or repeated artifacts; deduplication logic may be needed for high-throughput runs.
- Current merge strategy uses list `extend()` semantics and dict `update()` for shallow merges — complex nested merges can overwrite data unintentionally.
- HF client assumptions: different response schemas from HF or other providers may require additional adapters. `ModelSchemaAdapter` handles many shapes but not all.
- Memory persistence is file-backed and not transactional. For production, consider using a small database or append-only write-ahead log.

**Recommended next steps**
1. Run static syntax checks (e.g., `python -m pyflakes` or `ruff`) and unit tests focused on parsing helpers (`_first_json`, `_strip_code_fence`, `_balanced_json_slices`).
2. Add unit tests for `ModelSchemaAdapter.extract_text()` across typical HF response shapes (dict with `generated_text`, streaming `delta` lists, lists of choices).
3. Add more conservative merge/dedupe logic after parallel stages (e.g., dedupe messages by content hash, dedupe sources by URL).
4. Improve `MemoryStore.save()` with retry/backoff and non-blocking semantics (log and continue on failure).
5. Consider replacing deep `deepcopy(ctx)` with a `ContextSnapshot` that contains only the fields agents will mutate, reducing memory and CPU overhead.
6. Add CI to run a smoke test of a dry-run (no HF token) to validate orchestration doesn't crash with mocks.

**Where the recent edits live**
- Thread-safety changes were applied to `AgentContext.post()` and `_run_parallel_group()` in [autobot/architecture.py](autobot/architecture.py).

**Production hardening checklist (short)**
- [ ] Add a transactional memory backend (SQLite/LMDB).
- [ ] Add structured logging and log levels forwarded to a file or telemetry system.
- [ ] Add rate-limiting and retries for external web requests in `ToolBox`.
- [ ] Add robust schema validation and fallback tests for each agent.
- [ ] Add optional distributed execution layer (e.g., task queue) if scaling beyond single-machine concurrency.

---

If you want, I can now:
- Run a static syntax check on `autobot/architecture.py` and report any errors.
- Add unit tests for the JSON parsing helpers and `ModelSchemaAdapter.extract_text()`.
- Implement memory save retry/backoff in `MemoryStore.save()`.

Tell me which you'd like next.

### Example interaction scenarios — inputs and expected outputs

Below are representative inputs and how the architecture typically processes them (which agents are involved) and the expected output shape.

- **Greeting / Casual**
  - Input examples: `hi`, `hello`, `hello my name is Alice, what is yours?`
  - Processing:
    - `IntentParsingAgent` classifies as conversational/general_qa; low complexity.
    - `MetaControllerAgent` selects a simple route (no web/tool use).
    - `PlanningAgent` may produce a tiny plan like `['Respond politely']`.
    - `ReflexiveMetacognitiveAgent` / `EnsembleAgent` may add tone/clarity polishing.
  - Expected output (final):
    ```json
    {
      "final_answer": "Hi Alice — I'm an assistant here to help. What would you like to talk about?",
      "confidence": 0.62,
      "artifacts": {"plan":"Respond politely"}
    }
    ```

- **Capabilities / Discovery**
  - Input example: `Tell me what are your capabilities.`
  - Processing:
    - `IntentParsingAgent` sets `primary_task` to `general_qa` and tags capabilities.
    - `PlanningAgent` creates a short plan to list capabilities and examples.
    - `EnsembleAgent` or `RLHFAgent` may refine phrasing.
  - Expected output:
    ```json
    {
      "final_answer": "I can help with code, debugging, research, summaries, and workflow automation. For example: (1) summarize articles, (2) generate code snippets, (3) help debug errors.",
      "confidence": 0.71,
      "artifacts": {"capabilities":["code_generation","debugging","data_gathering","workflow_automation"]}
    }
    ```

- **Definitions / Comparison / Code Example**
  - Input example: `Explain data and machine learning, compare them in a table, and show a short Python example.`
  - Processing:
    - `IntentParsingAgent` marks analysis + code_generation as intents.
    - `MetaControllerAgent` enables `tree_of_thoughts` or `ensemble` if complexity is non-trivial.
    - `PlanningAgent` decomposes into subtasks: define terms, produce comparison table, generate code snippet.
    - `ToolUseAgent` not required unless external examples are requested.
  - Expected output (truncated):
    ```json
    {
      "final_answer": "Definitions:\n- Data: ...\n- Machine Learning: ...\n\nComparison table: [ ['Aspect','Data','ML'], ['Purpose','raw observations','learn models from data'], ... ]\n\nExample:\n```python\nimport numpy as np\nfrom sklearn.linear_model import LinearRegression\n...\n```",
      "confidence": 0.74,
      "artifacts": {"plan":["Define terms","Create comparison table","Provide code sample"]}
    }
    ```

- **Emotion / Empathy**
  - Input examples: `I love you`, `I am sad` (emotional statements)
  - Processing:
    - `IntentParsingAgent` detects conversational/emotional tone (special-case tag).
    - `MetaControllerAgent` routes to low-complexity, high-safety response (no web/tool use).
    - `ReflexiveMetacognitiveAgent` or `RLHFAgent` contributes an empathetic, safe reply; `PEV` not required.
    - If message indicates distress or self-harm, architecture should fall back to safety templates and resource suggestions (this repo includes no crisis-line lookup by default; integrate tools for local resources if required).
  - Expected outputs:
    - Input: `I love you` → Output: polite, boundary-aware reply.
      ```json
      {"final_answer":"Thank you — I'm here to help and support you. How can I assist you today?","confidence":0.58}
      ```
    - Input: `I am sad` → Output: empathetic reply + optional help suggestions.
      ```json
      {"final_answer":"I'm sorry you're feeling sad. Would you like to talk about what's on your mind? If you are in crisis, please contact local emergency services or a crisis hotline.","confidence":0.67}
      ```

Notes on safety and personality:
- The architecture separates factual assistance from emotional support: factual tasks use planning, evidence collection, and verification (PEV), while emotional queries prioritize safe, empathetic, and non-clinical responses.
- For any input indicating self-harm or immediate danger, integrate an external safety policy and local resource lookup rather than relying solely on model text.

---

If you want, I can expand these examples into unit tests (input → expected ctx state and final_answer assertions) or add canned safety templates for emotion responses. Which would you like next?
