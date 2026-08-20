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
