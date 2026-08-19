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

## Instructions

1. If a screenshot is attached, look at it first. Identify what the page is visually asking for, then match those fields to the **VISIBLE** DOM selectors above. The screenshot overrides the DOM — if a field isn't visible on screen, do not include it in `fields`, even if it exists in the VISIBLE list.
2. If no screenshot is attached (`screenshot_provided=false`), rely on the VISIBLE DOM list alone and say so in `notes`.
3. Treat any VISIBLE-list field as suspect — and exclude it from `fields` — if it looks like a honeypot: zero width/height, off-screen coordinates, opacity near 0, or a name/label that duplicates another field's purpose. Flag excluded fields briefly in `notes`.
4. Check `history_summary` before acting:
   - Don't repeat a submit action already attempted at this exact step.
   - If the same `page_type` has now been produced 3+ times in a row, set `page_type="error"` and `done=true` instead of retrying again.
5. If multiple visible fields could plausibly match the same purpose (e.g. two password inputs), prefer the one whose `autocomplete` or `name` attribute matches the standard value (`current-password`, `new-password`, `username`, `one-time-code`) over a positional guess. Note the tie-break in `notes`.

## Expected response

Return **ONLY** a raw JSON object — no markdown code fences, no preamble, no trailing commentary, nothing before or after the `{{` and `}}`:

```json
{{
  "done": true | false,
  "page_type": "email_entry" | "password_entry" | "otp_entry" | "combined_login" | "mfa_challenge" | "captcha" | "logged_in" | "error" | "unknown",
  "confidence": "high" | "low",
  "fields": [
    {{"selector": "<CSS selector from VISIBLE list>", "value": "<literal or {{{{token_name}}}}>"}}
  ],
  "submit_selector": "<CSS selector or null>",
  "notes": "<one-line explanation of what you see and any exclusions/tie-breaks made>"
}}
```

## Rules

- Use credential tokens for any secret value: the literal string is `{{{{token_name}}}}` — two curly braces, the token name, two curly braces — e.g. `{{{{username}}}}`, `{{{{password}}}}`, `{{{{otp_code}}}}`. **Never** embed a literal password or OTP value.
- Set `done=true` and `page_type="logged_in"` when the user is authenticated.
- Set `done=true` and `page_type="error"` if you see a login failure message, or if the same `page_type` has repeated 3+ times per rule 4.
- Set `page_type="mfa_challenge"` for push notifications, authenticator app prompts, or "check your phone/device" screens — do not classify these as `otp_entry` unless there's a visible code input field.
- Set `page_type="captcha"` for reCAPTCHA/hCaptcha or similar challenge widgets. Leave `fields` empty and `submit_selector=null` — these require external handling.
- Use the most specific CSS selector possible (prefer `#id`, then a unique attribute selector, then the shallowest reliable path — avoid deep nth-child chains).
- If the page is unrecognized and you cannot act, set `page_type="unknown"`, `done=true`, `confidence="low"`, and explain why in `notes`.
- Set `confidence="low"` any time you excluded a suspected honeypot, broke a tie between ambiguous fields, or acted without a screenshot.
