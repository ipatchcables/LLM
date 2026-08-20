Good question — this splits into two extraction problems that converge on one schema, and the trick is doing DOM-based extraction for both rather than treating static HTML parsing and SPA extraction as separate code paths.

Why not parse raw HTML with BeautifulSoup/lxml

For traditional server-rendered forms this works fine, but it breaks down for SPAs where forms are constructed by JS after load (React controlled inputs, Vue v-model, dynamically injected <select> options, shadow DOM web components). If your crawler has a "static HTML mode" and a "SPA mode" as separate extractors, you get drift — two code paths to maintain, two sets of bugs, and edge cases (a Django app with a sprinkle of Alpine.js) that don't cleanly belong to either.

Better approach: always extract from the rendered DOM via Playwright/CDP, never from raw response bodies. Since you're already using Playwright as executor in lemi4/deepagent3, this is just "run the same extraction routine after networkidle or a stability heuristic, regardless of whether the page was server-rendered or client-rendered." Static HTML becomes a degenerate case of SPA extraction — it's a DOM state that happened not to need JS to reach.

Core extraction routine (via page.evaluate)

Walk the DOM (including shadow roots, and same-origin iframes recursively) and pull:

form elements: action, method, enctype, name/id
All descendant inputs whether or not they're inside a <form> tag — SPAs frequently build "forms" as plain <div> wrappers with a JS-bound submit handler on a button, no <form> element at all
For each field: tag, type, name, id, associated <label> (via for, wrapping, or aria-label/aria-labelledby), placeholder, required, pattern, maxlength, current value, and for select/radio/checkbox groups, the full option set
Framework fingerprints where useful: React sets __reactProps$* / fiber keys on the DOM node, Vue attaches __vue__/_vnode — detecting these tells you the field is controlled by JS state rather than being a dumb input, which matters for how you'll interact with it later (native fill() vs dispatching input events so React's synthetic event system picks it up)
Client-side validation hints: pattern, required, plus a lightweight static scan of associated <script>/bundle for validation library signatures if you want to get fancy (probably overkill for v1)
Handling the "form" that has no submit button

Common in SPA CRUD UIs — fields plus an onClick handler on a <button type="button"> that fires an XHR/fetch. Two options:

Structural heuristic: cluster inputs that share a common DOM ancestor within N levels, with a trailing button-like element (role="button", <button>, or clickable div with submit-ish text) as the completion action.
Network correlation (more reliable): attach the extraction pass to your existing network observer. When a button is clicked and it triggers a state-changing XHR/fetch/GraphQL mutation shortly after, retroactively tag the input cluster you interacted with as the "form" and record the actual submission endpoint, method, and payload shape from the intercepted request — this is strictly more accurate than the action/method attributes because SPAs often don't set them meaningfully anyway.


Trigger timing for SPA extraction

Since forms can be conditionally rendered (multi-step wizards, modals, tab panels), a single extraction pass on page load isn't enough:
Re-run extraction after every navigation-equivalent event: route change (listen for History API pushState/replaceState + popstate), DOM mutation bursts (MutationObserver debounced ~300-500ms), and after any of your agent's own interactions (click, focus) settle
This ties naturally into your existing coverage/state model in lemi4 — treat "new form discovered" as a coverage-relevant state transition, same as a new URL

Unified output schema

Whatever the source, normalize to one structure so downstream (your WALL gateway, IDOR testing lanes, payload injection) doesn't care how the form was found:

json
{
  "form_id": "stable-hash-of-dom-path-or-endpoint",
  "discovery_method": "static_dom | network_correlated",
  "submit_target": {"url": "...", "method": "POST", "content_type": "application/json"},
  "fields": [
    {"name": "...", "type": "...", "required": true, "constraints": {...}, "label": "..."}
  ],
  "page_context": {"url": "...", "route": "..."}
}

discovery_method matters for your completion contract — if a form was only discoverable via network correlation, you know static crawling alone would've missed it, which is useful signal for coverage confidence scoring.

One gotcha worth flagging

Re-running full-DOM extraction on every mutation is expensive at scale. Debounce it, and scope the MutationObserver to meaningful subtrees rather than the whole document.body if you can identify an app root (#root, #app) — cuts a lot of noise from unrelated UI churn (toasts, animations) triggering unnecessary re-extraction passes.

Re-run extraction after every navigation-equivalent event: route change (listen for History API pushState/replaceState + popstate), DOM mutation bursts (MutationObserver debounced ~300-500ms), and after any of your agent's own interactions (click, focus) settle
This ties naturally into your existing coverage/state model in lemi4 — treat "new form discovered" as a coverage-relevant state transition, same as a new URL
