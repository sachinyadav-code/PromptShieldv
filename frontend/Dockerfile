# syntax=docker/dockerfile:1
# PromptShield Enterprise LLM Security Gateway - Frontend Dockerfile
# Multi-stage build: Node.js build stage -> Nginx unprivileged static web server

# ------------------------------------------------------------------------------
# Stage 1: Build static web bundle
# ------------------------------------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Copy dependency specifications for deterministic npm ci installation
COPY frontend/package.json frontend/package-lock.json /app/

RUN npm ci

# Copy frontend source code and configuration files
COPY frontend /app/

# Build argument for backend API URL consumed by Vite during static asset compilation
# Default is empty string so client falls back to http://127.0.0.1:8000
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Compile TypeScript and build production bundle into /app/dist
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Serve static bundle via Nginx unprivileged
# ------------------------------------------------------------------------------
FROM nginxinc/nginx-unprivileged:alpine-slim AS runtime

# Copy compiled static assets from build stage to nginx document root
COPY --from=build --chown=nginx:nginx /app/dist /usr/share/nginx/html

# Expose unprivileged web server port (default for nginx-unprivileged is 8080)
EXPOSE 8080

# Healthcheck verified against nginx server
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:8080/ || exit 1
CMD ["nginx", "-g", "daemon off;"]