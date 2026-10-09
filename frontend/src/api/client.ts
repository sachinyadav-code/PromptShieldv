import type {
  AuditQueryParams,
  AuditQueryResponse,
  HealthResponse,
  InspectRequest,
  InspectResponse,
  ProxyResponse,
  RAGInspectResponse,
  RAGProxyResponse,
  RAGRequest,
  RuleInfo,
} from '../types/api';

export class ApiError extends Error {
  status: number;
  retryAfter?: number;
  detail?: string;

  constructor(status: number, message: string, detail?: string, retryAfter?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
  }
}

export function getDefaultApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (typeof envUrl === 'string' && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  return 'http://127.0.0.1:8000';
}

export class PromptShieldApiClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl: string, apiKey: string = '') {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.apiKey = apiKey.trim();
  }

  setApiKey(key: string) {
    this.apiKey = key.trim();
  }

  setBaseUrl(url: string) {
    this.baseUrl = url.trim().replace(/\/+$/, '');
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  hasApiKey(): boolean {
    return this.apiKey.length > 0;
  }

  private getHeaders(extraHeaders: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = {
      ...extraHeaders,
    };
    if (this.apiKey) {
      // PromptShield supports Authorization: Bearer <key>
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    const headers = this.getHeaders(
      (options.headers as Record<string, string>) || {}
    );

    let response: Response;
    try {
      response = await fetch(url, {
        ...options,
        headers,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Network request failed';
      throw new ApiError(0, `Connection error to ${this.baseUrl}: ${msg}`);
    }

    const retryAfterHeader = response.headers.get('Retry-After');
    const retryAfter = retryAfterHeader ? parseInt(retryAfterHeader, 10) : undefined;

    if (!response.ok) {
      let detail = '';
      try {
        const errJson = await response.json();
        if (typeof errJson === 'object' && errJson !== null) {
          if (typeof errJson.detail === 'string') {
            detail = errJson.detail;
          } else if (Array.isArray(errJson.detail)) {
            detail = errJson.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join(', ');
          } else {
            detail = JSON.stringify(errJson);
          }
        }
      } catch {
        // Non-JSON response
        detail = await response.text();
      }

      let errorMsg = `HTTP ${response.status}`;
      if (response.status === 401) {
        errorMsg = 'Authentication required or invalid API key (401).';
      } else if (response.status === 403) {
        errorMsg = 'Forbidden (403).';
      } else if (response.status === 429) {
        errorMsg = `Rate limit exceeded (429).${retryAfter ? ` Retry in ${retryAfter}s.` : ''}`;
      } else if (response.status === 503) {
        errorMsg = `Gateway capacity reached (503).${retryAfter ? ` Retry in ${retryAfter}s.` : ''}`;
      } else if (response.status === 422) {
        errorMsg = `Validation error (422): ${detail || 'Invalid payload.'}`;
      } else if (detail) {
        errorMsg = `${errorMsg}: ${detail}`;
      }

      throw new ApiError(response.status, errorMsg, detail, retryAfter);
    }

    try {
      return (await response.json()) as T;
    } catch {
      throw new ApiError(response.status, 'Invalid JSON returned from server.');
    }
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async getRules(): Promise<RuleInfo[]> {
    return this.request<RuleInfo[]>('/api/v1/rules');
  }

  async inspectPrompt(payload: InspectRequest): Promise<InspectResponse> {
    return this.request<InspectResponse>('/api/v1/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  async proxyPrompt(payload: InspectRequest): Promise<ProxyResponse> {
    return this.request<ProxyResponse>('/api/v1/proxy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  async inspectRAG(payload: RAGRequest): Promise<RAGInspectResponse> {
    return this.request<RAGInspectResponse>('/api/v1/rag/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  async proxyRAG(payload: RAGRequest): Promise<RAGProxyResponse> {
    return this.request<RAGProxyResponse>('/api/v1/rag/proxy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  async getAuditEvents(params: AuditQueryParams = {}): Promise<AuditQueryResponse> {
    const query = new URLSearchParams();
    if (params.limit !== undefined) query.set('limit', String(params.limit));
    if (params.offset !== undefined) query.set('offset', String(params.offset));
    if (params.endpoint) query.set('endpoint', params.endpoint);
    if (params.decision) query.set('decision', params.decision);
    if (params.event_type) query.set('event_type', params.event_type);
    if (params.start_time) query.set('start_time', params.start_time);
    if (params.end_time) query.set('end_time', params.end_time);

    const queryString = query.toString();
    const endpoint = queryString ? `/api/v1/audit/events?${queryString}` : '/api/v1/audit/events';
    return this.request<AuditQueryResponse>(endpoint);
  }
}
