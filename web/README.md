# Sentinel web UI

React + TypeScript + Vite front end for the Sentinel agent API
(`sentinel/api/app.py`).
A single-page orchestrator shell: session sidebar, agent chat with a
confirmation rail, and an operations panel (agent activity, model
leaderboard, alerts).

See the root `README.md` for what Sentinel is and how to run the backend.

## Run it

```bash
npm install
npm run dev
```

Point it at a running API with `VITE_API_BASE` (see `.env.example`,
defaults to `http://localhost:8000`).

## Scripts

- `npm run dev` - Vite dev server
- `npm run build` - typecheck (`tsc -b`) + production build
- `npm run lint` - oxlint
- `npx vitest run` - unit/component tests
