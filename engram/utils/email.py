from __future__ import annotations

import os

from engram.utils.logger import get_logger

logger = get_logger(__name__)


def send_api_key_email(to_email: str, api_key: str) -> bool:
    """Send an API key to a developer via email.

    Security properties:
    - API key sent over HTTPS to Resend — encrypted in transit
    - Key never logged — not even first characters
    - Resend API key read from environment — never hardcoded
    - Returns True/False — never raises to expose internals
    - Email address never logged in full — only domain

    Parameters
    ----------
    to_email: Recipient email address. Already validated upstream.
    api_key:  The raw API key to send. Never logged or stored after this.

    Returns
    -------
    True if email sent successfully. False otherwise.
    """
    resend_api_key = os.environ.get("RESEND_API_KEY")
    if not resend_api_key:
        logger.error("RESEND_API_KEY not set — cannot send email")
        return False

    try:
        import resend
        resend.api_key = resend_api_key

        resend.Emails.send({
            "from":    "Engram <onboarding@resend.dev>",
            "to":      [to_email],
            "subject": "Your Engram API Key",
            "html":    _build_email_html(api_key),
        })

        logger.info("api key email sent", extra={
            "domain": to_email.split("@")[1]
        })
        return True

    except Exception as e:
        logger.error("email send failed", extra={"error": type(e).__name__, "detail": str(e)})
        return False


def _build_email_html(api_key: str) -> str:
    """Build the API key email HTML."""
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0a0a;
    color: #e0e0e0;
    padding: 40px 20px;
    margin: 0;
  }}
  .container {{
    max-width: 560px;
    margin: 0 auto;
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 12px;
    padding: 40px;
  }}
  .logo {{
    font-size: 24px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 24px;
  }}
  .logo span {{ color: #4ade80; }}
  .key-box {{
    background: #0a0a0a;
    border: 1px solid #4ade80;
    border-radius: 8px;
    padding: 16px 20px;
    font-family: monospace;
    font-size: 14px;
    color: #4ade80;
    word-break: break-all;
    margin: 24px 0;
  }}
  .warning {{
    font-size: 13px;
    color: #888;
    margin-top: 16px;
  }}
  .code {{
    background: #0a0a0a;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: monospace;
    font-size: 13px;
    color: #e0e0e0;
    margin: 16px 0;
  }}
  a {{ color: #4ade80; }}
</style>
</head>
<body>
<div class="container">
  <div class="logo">engram<span>.</span></div>

  <p>Your Engram API key is ready.</p>

  <div class="key-box">{api_key}</div>

  <p class="warning">
    ⚠️ Save this key somewhere safe. It cannot be recovered if lost.
    If you lose it, request a new one at the dashboard.
  </p>

  <p style="margin-top: 24px; color: #888; font-size: 14px;">
    Quick start:
  </p>

  <div class="code">
curl -X POST https://web-production-07b0a4.up.railway.app/v1/ingest \\<br>
&nbsp;&nbsp;-H "X-API-Key: {api_key[:20]}..." \\<br>
&nbsp;&nbsp;-d '{{"user_id": "me", "content": "I prefer Python"}}'
  </div>

  <p style="margin-top: 24px; font-size: 14px; color: #888;">
    <a href="https://web-production-07b0a4.up.railway.app/docs">API Docs</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/deekshith080/engram-public">GitHub</a>
    &nbsp;·&nbsp;
    <a href="mailto:cdeekshith1@gmail.com">Support</a>
  </p>

  <p style="margin-top: 32px; font-size: 12px; color: #444;">
    Memory belongs to the user. Engram is the guardian, not the owner.
  </p>
</div>
</body>
</html>
"""