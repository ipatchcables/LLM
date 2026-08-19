| Field | Value |
|-------|-------|
| **Current URL** | {url} |
| **Step number** | {step_number} |
| **Available credential tokens** | {available_tokens} |
| **Screenshot provided** | {screenshot_provided} |

## Steps completed so far
{history_summary}

## VISIBLE DOM elements (use these selectors)
```json
{visible_summary}
```

## HIDDEN DOM elements (DO NOT act on these)
```json
{hidden_summary}
```

## OCCLUDED elements (visible in DOM but currently blocked by an overlay — DO NOT act on these)
```json
{occluded_summary}
```

## Instructions

1. If a screenshot is attached, look at it first. Identify what the page is visually asking for, then match those fields to the **VISIBLE** DOM selectors above. The screenshot overrides the DOM — if a field isn't visible on screen, do not include it in `fields`, even if it exists in the VISIBLE list.
2. If no screenshot is attached (`screenshot_provided=false`), rely on the VISIBLE DOM list alone and say so in `notes`.
3. Treat any VISIBLE-list field as suspect — and exclude it from `fields` — if it looks like a honeypot: zero width/height, off-screen coordinates, opacity near 0, or a name/label that duplicates another field's purpose. Flag excluded fields briefly in `notes`.
4. **Never** select a field or `submit_selector` that appears in the OCCLUDED list, even if it also appears in VISIBLE — occlusion takes precedence. If the field you need is occluded, set `page_type="blocked_by_overlay"` instead of attempting the click.
5. Check `history_summary` before acting:
   - Don't repeat a submit action already attempted at this exact step.
   - If the same `page_type` has now been produced 3+ times in a row, set `page_type="error"` and `done=true` instead of retrying again.
   - If `page_type="blocked_by_overlay"` was already returned at this step once before, this turn should assume the dismiss action ran — re-check the (now updated) OCCLUDED list rather than immediately returning `blocked_by_overlay` again.
6. If multiple visible, non-occluded fields could plausibly match the same purpose (e.g. two password inputs), prefer the one whose `autocomplete` or `name` attribute matches the standard value (`current-password`, `new-password`, `username`, `one-time-code`) over a positional guess. Note the tie-break in `notes`.

## Expected response

Return **ONLY** a raw JSON object — no markdown code fences, no preamble, no trailing commentary, nothing before or after the `{{` and `}}`:

```json
{{
  "done": true | false,
  "page_type": "email_entry" | "password_entry" | "otp_entry" | "combined_login" | "mfa_challenge" | "captcha" | "blocked_by_overlay" | "logged_in" | "error" | "unknown",
  "confidence": "high" | "low",
  "fields": [
    {{"selector": "<CSS selector from VISIBLE list, not in OCCLUDED>", "value": "<literal or {{{{token_name}}}}>"}}
  ],
  "submit_selector": "<CSS selector or null>",
  "dismiss_selector": "<CSS selector for overlay/backdrop to dismiss, or null>",
  "notes": "<one-line explanation of what you see, any exclusions/tie-breaks, and the occluder class if blocked>"
}}
```

## Rules

- Use credential tokens for any secret value: the literal string is `{{{{token_name}}}}` — two curly braces, the token name, two curly braces — e.g. `{{{{username}}}}`, `{{{{password}}}}`, `{{{{otp_code}}}}`. **Never** embed a literal password or OTP value.
- Set `done=true` and `page_type="logged_in"` when the user is authenticated.
- Set `done=true` and `page_type="error"` if you see a login failure message, or if the same `page_type` has repeated 3+ times per rule 5.
- Set `page_type="blocked_by_overlay"` (with `done=false`) whenever the field or submit control you need is in the OCCLUDED list. Populate `dismiss_selector` with the best candidate to clear it — prefer a `.cdk-overlay-backdrop` element if one is present in OCCLUDED/HIDDEN, otherwise leave `dismiss_selector=null` and note in `notes` that an Escape-key dismiss should be tried first. Leave `fields` and `submit_selector` null on this response — do not attempt the click through the overlay.
- Set `page_type="mfa_challenge"` for push notifications, authenticator app prompts, or "check your phone/device" screens — do not classify these as `otp_entry` unless there's a visible code input field.
- Set `page_type="captcha"` for reCAPTCHA/hCaptcha or similar challenge widgets. Leave `fields` empty and `submit_selector=null` — these require external handling.
- Use the most specific CSS selector possible (prefer `#id`, then a unique attribute selector, then the shallowest reliable path — avoid deep nth-child chains).
- If the page is unrecognized and you cannot act, set `page_type="unknown"`, `done=true`, `confidence="low"`, and explain why in `notes`.
- Set `confidence="low"` any time you excluded a suspected honeypot, broke a tie between ambiguous fields, returned `blocked_by_overlay`, or acted without a screenshot.
