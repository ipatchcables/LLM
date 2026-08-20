1. Context isolation matches your own context-sizing findings
You'd already converged on planner/executor separation being the most impactful context reduction strategy, with 32K as the sweet spot. A monolithic "do everything" agent has to hold XSS payload knowledge, SQLi syntax variants, SSRF bypass techniques, and IDOR heuristics all in context simultaneously, plus the raw HTTP traffic from testing all of them. A dedicated XSS subagent only needs XSS-relevant context — sink types, encoding contexts, CSP bypass patterns — which is a much smaller, denser, more relevant window. This isn't just cost savings; a smaller, more focused context tends to produce better payload selection because the model isn't diluting attention across unrelated vulnerability classes.

2. Vulnerability classes have genuinely different decision trees
XSS testing is fundamentally about output context (HTML body vs attribute vs JS string vs URL) driving payload shape. SQLi is about injection point + DB backend + error visibility driving technique (error-based vs blind vs time-based). These aren't just "different payloads," they're different reasoning strategies — different signals to look for, different confirmation methods, different false-positive patterns. Cramming both into one agent's system prompt means either bloating it with conditional logic ("if testing XSS, do X; if testing SQLi, do Y") or accepting worse decisions from a generalist trying to do both reasoning styles at once.

3. It maps cleanly onto your gateway/policy model
A per-vuln-class subagent gives you a natural place to scope the WALL gateway's allowed-action set tightly — the SQLi subagent's action vocabulary doesn't need to include DOM manipulation payloads, and vice versa. Tighter allowed-action sets per subagent is a smaller attack surface for the gateway to reason about and a smaller blast radius if a subagent misbehaves (prompt injection from a malicious response, say) — it can only misuse the tools it was scoped, which is the same "capability-not-credential discipline at the tool layer" principle you'd already landed on for the LangGraph crawler.

4. Specialization is independently testable and independently improvable
This is the practical payoff: you can eval, tune, and version the XSS subagent against a golden set of known-XSS-vulnerable test apps without touching the SQLi subagent at all. Given you're already doing replay-equivalence testing (the lemi SQLi detection work), a monolithic agent makes it much harder to isolate why a regression happened — was it the SQLi logic or did an unrelated prompt change hurt XSS detection as a side effect?

Where you don't want full fragmentation:

Don't split by vuln class if the vuln classes overlap heavily in mechanism. SSRF and IDOR-via-URL-manipulation share a lot of "does this parameter control server-side resource resolution" reasoning — splitting those into fully separate subagents can cause redundant discovery work (both probing the same parameter independently) or missed compound findings (an SSRF that's exploitable because of an IDOR). Group by shared reasoning pattern, not strictly by CWE/OWASP category.
Don't spawn a subagent per class unconditionally — that's wasted concurrency budget and cost if the target surface obviously has no injection points at all (e.g., a purely static site). The planner should decide which specialist subagents are worth spawning based on a cheap initial recon pass (parameter discovery, tech-stack fingerprinting), not fan out all specialists against everything by default.
Share a common finding schema and coverage tracker across all subagents, even though they're specialized — this is exactly the deterministic state layer from the LangGraph discussion. The specialization should be in the reasoning (planner routing + per-class subagent prompts/tools), not duplicated in the bookkeeping (coverage, completion contract, report formatting), or you end up reimplementing that logic N times with N chances to diverge.

Concretely, the shape I'd build:

Planner (generalist, deterministic-ish routing)
  ├── recon pass → identifies candidate injection points, tech stack, input surfaces
  ├── routes to relevant specialists based on recon, not blind fan-out
  │
  ├── XSS subagent (specialized: reflected/stored/DOM sink analysis)
  ├── SQLi subagent (specialized: error/blind/time-based by DB backend)
  ├── SSRF/IDOR subagent (grouped: shared "server-side resource resolution" reasoning)
  ├── Deserialization subagent (specialized: language/framework-specific gadget chains)
  │
  └── shared: coverage tracker, completion contract, WALL gateway (scoped per-subagent
      allowed-actions), finding schema
