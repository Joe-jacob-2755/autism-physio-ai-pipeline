# Rule: Security Baseline

## Scope
All code handling file I/O, user input, serialisation, and network connections.

## Rules

### R6.1 — No eval, exec, or pickle on untrusted input
Never use `eval()`, `exec()`, or `pickle.loads()` on data from external sources. For model serialisation, prefer `joblib` with trusted paths only, and document the trust boundary.

### R6.2 — Validate file paths
User-provided paths MUST be validated:
- Resolve to absolute path and verify it's within expected directories
- No path traversal (`../` sequences)
- Check file exists before processing (explicit FileNotFoundError, not silent fallback)

### R6.3 — No secrets in code
No API keys, credentials, passwords, or tokens in any tracked file. Use environment variables or `.env` files (which MUST be in `.gitignore`).

### R6.4 — Sanitise CSV input
When reading external CSVs, validate:
- Column names match expected schema
- Data types are correct (no strings where floats expected)
- Row count is within reasonable bounds
- No formula injection characters (`=`, `+`, `-`, `@`) in string fields that may be exported

### R6.5 — Participant data minimisation
Output files should contain the minimum data necessary. Demographic details should use coded values (severity=2, not narrative descriptions that could identify individuals). Output paths should not contain identifiable information.

### R6.6 — Network connections require authentication context
Any code opening network connections (Mode 2.3 live ingestion) MUST document:
- What authenticates the connection
- What data is transmitted
- Whether encryption is used
- What happens if a malicious service is on the expected port

## Rationale
This pipeline handles physiological data from children — a vulnerable population under GDPR Article 9, UK DPA 2018, and potentially HIPAA. Security failures don't just expose data; they can harm research participants who cannot advocate for themselves.
