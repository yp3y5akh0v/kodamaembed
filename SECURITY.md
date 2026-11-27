# Security Policy

## Reporting a Vulnerability

Do not report security vulnerabilities through public GitHub issues. Use one
of these instead:

- [GitHub Private Vulnerability Reporting](https://github.com/yp3y5akh0v/kodamaembed/security/advisories/new)
  (preferred)
- Email: yuriy.peysakhov@gmail.com

Include:

- a description of the vulnerability
- steps to reproduce it
- the affected versions
- its potential impact

## Response

Reports are acknowledged within 5 business days, with updates while the issue
is investigated and fixed.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

Upgrade to the latest release before reporting an issue in an older version.

## Scope

- The HTTP server has no authentication and listens on 127.0.0.1 by default.
  Exposing it with `--host` is the operator's choice, behind an authenticating
  proxy.
- Models download at a pinned commit and are checked against SHA-256 before
  anything parses them; a way around that check is in scope.
- Memory-safety bugs reachable from model files, tokenizer files or request
  bodies are in scope.

## Responsible Disclosure

- Report vulnerabilities privately, through the channels above.
- Give enough detail to reproduce the issue.
- Do not disclose the vulnerability publicly until a fix is available.
- Do not exploit the vulnerability beyond what is needed to verify it.
