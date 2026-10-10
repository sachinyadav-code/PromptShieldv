# PromptShield Containerization Guide (Phase 6 Stage 6.5)

This guide documents the containerization architecture, security model, and execution procedures for running PromptShield via Docker and Docker Compose.

---

## 1. Architecture Overview

PromptShield is packaged into two decoupled, containerized services:

1. **Backend Gateway (`backend/Dockerfile`)**:
   - Base image: `python:3.14-slim` (matches verified host runtime and ABI-compatible binary wheels for NumPy, SciPy, and scikit-learn).
   - Serves the FastAPI ASGI application via `uvicorn` on port `8000`.
   - Runs under a dedicated, unprivileged non-root user (`promptshield`, UID/GID created during build).
   - Includes deterministic detection rules, ML prompt injection model pipeline (`backend/app/detection/models/prompt_injection_v1.joblib`), and RAG corpus dataset (`datasets/rag_corpus_v1.jsonl`).
   - Stores persistent audit events in SQLite WAL mode on a named persistent volume mounted at `/data`.
   - Healthcheck: verified against public unauthenticated endpoint `GET /health`.

2. **Frontend Security Dashboard (`frontend/Dockerfile`)**:
   - Multi-stage build:
     - **Build Stage**: `node:22-alpine` installs dependencies reproducibly via `npm ci` and compiles TypeScript + Vite static assets (`npm run build`).
     - **Runtime Stage**: `nginxinc/nginx-unprivileged:alpine-slim` serves static HTML/JS/CSS assets on unprivileged port `8080`.
   - Zero Node.js runtime, compilers, or `node_modules` remain in the production image layer.
   - Configures the backend API URL build argument `VITE_API_BASE_URL`.

---

## 2. Browser Networking vs. Build-Time Static Assets

### Critical Browser Architecture Constraint
The React frontend executes **in the user's web browser**, not inside the internal Docker bridge network.
- Browser requests to `http://backend:8000` will fail because the client machine's browser cannot resolve internal Docker container hostnames.
- The browser must reach the backend through a host-accessible origin (e.g., `http://127.0.0.1:8000` or `http://localhost:8000`).
- Correspondingly, `PROMPTSHIELD_ALLOWED_ORIGINS` on the backend must permit requests from the frontend origin (default in Compose: `http://localhost:8080,http://127.0.0.1:8080`).

### Build-Time Embedding Notice (Vite)
Because Vite compiles single-page application (SPA) bundles at build time, `VITE_API_BASE_URL` is **embedded into static JavaScript files during `docker compose build`**.
- Changing `VITE_API_BASE_URL` or changing `BACKEND_PORT` at container runtime does **not** alter an already-built static bundle.
- If you override `BACKEND_PORT` or change the backend origin, you **must rebuild the frontend container** with `docker compose build frontend` or `docker compose up --build`.

---

## 3. Running with Docker Compose

### Prerequisites
- Docker Engine with Compose V2.
- A local `.env` file (copied from `.env.example`).

### Step 1: Configure Secrets
Copy the template and set high-entropy API keys:
```bash
cp .env.example .env
```
Edit `.env` to define:
```env
PROMPTSHIELD_API_KEYS=your-high-entropy-api-key-here
```
*Note: If `PROMPTSHIELD_API_KEYS` is left empty when authentication is enabled, all requests to sensitive gateway endpoints fail closed with HTTP 401 Unauthorized.*

### Step 2: Build and Run
```bash
docker compose up --build -d
```

### Step 3: Verify Status
```bash
docker compose ps
docker compose logs -f
```

- Access Dashboard UI: `http://localhost:8080`
- Access Backend Health: `http://localhost:8000/health`
- Access API Documentation: `http://localhost:8000/docs`

---

## 4. Port Override Procedure

If default host ports (`8000` for backend, `8080` for frontend) conflict with existing host services, you must synchronize three related variables and rebuild the frontend container:

### Example: Running Backend on Port 9000 and Frontend on Port 3000
Define the following in your `.env` file:
```env
# 1. Map backend container port 8000 to host port 9000
BACKEND_PORT=9000

# 2. Map frontend container port 8080 to host port 3000
FRONTEND_PORT=3000

# 3. Instruct Vite to target the overridden host backend port
VITE_API_BASE_URL=http://127.0.0.1:9000

# 4. Instruct backend CORS middleware to accept the overridden frontend origin
PROMPTSHIELD_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Mandatory Rebuild Step:
Rebuild and launch with:
```bash
docker compose up --build -d
```
*(Simply restarting without `--build` will cause the browser to continue targeting the previously baked API port).*

---

## 5. Production Hardening Checklist

When deploying to production container orchestrators (Kubernetes, AWS ECS, Google Cloud Run):
- **Secrets Management**: Inject `PROMPTSHIELD_API_KEYS` via orchestrator secret stores (e.g., Kubernetes Secrets, HashiCorp Vault), never plaintext environment files.
- **Audit Volume**: Mount `/data` to durable persistent block storage with backup snapshots.
- **HTTPS & Ingress**: Terminate TLS at the reverse proxy / ingress controller and update `PROMPTSHIELD_ALLOWED_ORIGINS` to the exact production HTTPS domain.
- **Security Options**: Maintain `no-new-privileges: true` and drop all Linux capabilities (`cap_drop: [ALL]`).