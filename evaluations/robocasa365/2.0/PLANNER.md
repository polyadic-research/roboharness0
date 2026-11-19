# Private planner API

The maintainer supplies an API endpoint and key outside the repository. Consumers configure `OPENAI_BASE_URL` and `OPENAI_API_KEY` in their process environment or an explicitly selected, untracked `.env` file. The source package contains client configuration and public interface requirements; API service implementation, deployment and private access arrangements stay external.

The selected 2.0 source uses the historical client identifier `codexgateway:gpt-6-astra`. This identifies the frozen client protocol and does not require users to deploy a gateway. The supplied API must support the client's OpenAI-compatible Chat Completions interface with images and tools, the `X-Codex-Session-ID` header, session continuation and cleanup at `DELETE codex/sessions/<session-id>`. A generic endpoint without those semantics is not established as compatible by an offline configuration check.

## Public settings

| Setting | Value | Scope |
| --- | --- | --- |
| Planner backend | `api` | Harness CLI |
| Historical client/model selector | `codexgateway:gpt-6-astra` | Unchanged client protocol |
| Model | `gpt-6-astra` | Supplied API |
| Reasoning effort | `xhigh` | Supplied API; the frozen client does not transmit this setting |
| Output cap | `--max-tokens 0` | Omit the client output-token cap |
| Nested planning timeout | `--plan-generation-timeout-s 180` | Each initial planning/replanning request |
| Outer Planner timeout | Unset (`null`) | Entire solve has no added CLI timeout |
| HTTP request timeout | 330 seconds | Frozen API client |

The machine-readable [configuration](planner.json) retains the historical source identity and controls. Its `gateway` key is the existing configuration schema name. It is not a service deployment specification. The ten selected historical 600-second nested-planning overrides remain in this configuration and [historical parameters](historical-parameters.json); they are not automatically applied to a future run. Do not use `--planner-timeout-s 330` as a replacement for the HTTP timeout.

## Configure private access

Copy `.env.example` to an untracked private file and fill the API variables there, or export them in the evaluation process. Actual URLs, keys and private connection files stay outside Git. Public inspection does not read credentials or send a request:

```bash
python tools/configure_robocasa_planner.py
```

An optional local check reads only the explicit private file and process variables; it prints presence and URL-shape results without private values:

```bash
/path/to/runtime/coordinator/bin/python tools/configure_robocasa_planner.py --check-env --env-file .env
```

Process variables take precedence over file values. Only the two API variables are loaded, without interpolation or an implicit dotenv search. An offline check does not contact the API.

The evaluation entry point accepts `--api-settings-confirmed` to acknowledge the supplied API settings before an explicitly requested execution. The earlier `--gateway-settings-confirmed` spelling remains an alias for existing scripts. The source package does not require API availability/quota receipts or service deployment files to be stored in the repository.

## Historical configuration evidence

The recorded 2.0 evaluations used OpenAI Codex SDK via CodexGateway, SDK version `0.159.0`, with `gpt-6-astra`, server-side `xhigh`, provider `openai` and provider/model fallback disabled. The sanitized [evidence inventory](planner-evidence.json) preserves the original source/record hashes and allowlisted settings. This is configuration provenance, not an independent attestation of the upstream model's identity. The source-binding record whose name contains `actual03` binds service source bytes only; excluded followup03 results remain excluded.
