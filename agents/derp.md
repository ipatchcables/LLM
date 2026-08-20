1. Deterministic control flow over model-owned flow, where possible
The more decision-making you push into the LLM's own reasoning loop, the more you're at the mercy of hallucinated tool calls, skipped steps, and non-reproducible runs. A pattern like lemi4's — model proposes, but a deterministic gateway/state machine actually decides what's allowed — tends to age much better than a "let the model figure out the whole loop" design. You've already landed on this with the WALL gateway and closed action vocabulary, and it's the right instinct.

2. A narrow, well-typed action/tool vocabulary
Orchestrators fail most often not because the model reasons badly, but because it has too many ways to express intent and the parser/dispatcher has to guess. Closed vocabularies (enum-like actions, structured args) beat open-ended "call any tool with any JSON" schemes for reliability and auditability — especially for security tooling where you need replayability.

3. Explicit state, not implicit conversation history
Raw chat history as "state" balloons context and buries the signal. A structured state object (current phase, findings so far, coverage map, budget remaining) that gets handed to each subagent/turn is both more token-efficient and easier to reason about failure from. This lines up with your 32K-context / structured-state recommendation for payload-gen agents.

4. Hard boundaries between planning and execution
Planner/executor separation isn't just a context-window optimization — it's a safety and debuggability boundary. The planner never sees raw HTTP traffic; the executor never makes strategic decisions. When something goes wrong, you know which layer to look at.

5. A completion/coverage contract
Orchestrators without a defined "done" condition tend to either loop forever or quit early having covered 20% of the surface. Your completion contract in lemi4 (coverage guarantees) is doing real work here — it's the difference between "the agent stopped" and "the agent finished."

6. Deny-by-default gateways for anything destructive
Especially relevant for security testing agents: any action with side effects (state-changing requests, destructive IDOR tests) needs to pass through a gate that defaults to deny, not one that defaults to allow-unless-flagged. Chokepoint architecture over scattered checks — one place to audit, one place to change policy.

7. Subagent delegation with narrow scope, not general-purpose subagents
spawn_subtask-style delegation works best when each specialist has a tightly scoped mandate (e.g., "confirm this SQLi finding" not "explore this app"). General-purpose subagents reintroduce all the control-flow problems you solved at the top level, just one layer down.

8. Observability/replay
For anything you'll need to put in a client report, being able to reconstruct why the orchestrator made a decision (which state, which action, which gateway rule fired) matters as much as the decision itself.

If you want, I can look at this against lemi4's actual architecture specifically — e.g., where the model-owned control flow and the deterministic WALL gateway might be fighting each other, or how the completion contract interacts with subagent delegation boundaries.
