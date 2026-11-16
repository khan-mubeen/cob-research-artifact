# CoB frontend

A React/TypeScript interface for the earlier CoB results dashboard, saved bidding replays and the optional interactive demo. The paper's v5 figures and tables are in [reports/v5/evaluation](../../reports/v5/evaluation/README.md); the dashboard is not a replacement for them.

## Setup

Use a Node.js version supported by the locked Vite release (Node 22.12+ or a compatible newer release). Node 24 was used for the audit checks.

From this folder:

```sh
npm ci
npm run dev
```

Open the address printed by Vite. The dashboard and saved replays use the checked-in files under `public/data/`; no model or backend is required for those views.

In PowerShell, use `npm.cmd` if the shell's execution policy blocks `npm.ps1`.

## Checks

```sh
npm run build
npm run lint
```

The build writes ignored output to `dist/`. `npm run preview` serves that build locally.

## Live demo

For live bidding, start the [API backend](../../src/api/README.md) separately. Its default address is `http://127.0.0.1:8000`; the frontend's live controls connect to that local service.

The full model-backed demo needs local GGUF files. Rule-only bidding does not. The backend uses earlier fine-tuning runs, not the v5 adapters evaluated in the paper.

The `make ui_data` target regenerates the older dashboard data from historical reports. It is not needed for setup and should not be run just to view the included dashboard.
