/**
 * TypeScript definitions mirroring PromptShield backend Pydantic models.
 */

export interface HealthResponse {
  status: string;
  service: string;
  phase: number;
}

export interface RuleInfo {
  rule_id: string;
  name: string;
  description: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  action: 'ALLOW' | 'FLAG' | 'BLOCK';
}

export interface MLAssessment {
  available: boolean;
  score: number | null;
  threshold: number;
  predicted_label: string | null;
  would_block: boolean;
  model_version: string | null;
  error_code: string | null;
}

export interface InspectRequest {
  prompt: string;
  request_id?: string | null;
}

export interface InspectResponse {
  decision: 'ALLOW' | 'BLOCK' | 'REVIEW';
  risk_score: number;
  matched_rules: string[];
  explanation: string;
  request_id: string;
  client_request_id?: string | null;
  ml_assessment?: MLAssessment | null;
  decision_source?: string | null;
}

export interface MockDownstreamResponse {
  simulated: boolean;
  provider: string;
  message: string;
  received_prompt_length: number;
}

export interface ProxyResponse {
  decision: 'ALLOW' | 'BLOCK' | 'REVIEW';
  request_id: string;
  client_request_id?: string | null;
  matched_rules: string[];
  explanation: string;
  downstream_response?: MockDownstreamResponse | null;
  ml_assessment?: MLAssessment | null;
  decision_source?: string | null;
}

export interface RAGRequest {
  query: string;
  request_id?: string | null;
  top_k?: number;
}

export interface RetrievedDocumentSummary {
  doc_id: string;
  title: string;
  source: string;
  relevance_score: number;
  matched_tokens: string[];
}

export interface DocumentInspectionSummary {
  doc_id?: string | null;
  assessment: 'ALLOW' | 'BLOCK' | 'REVIEW';
  risk_score: number;
  matched_rules: string[];
  is_complete?: boolean;
  explanation: string;
}

export interface RAGFindingSummary {
  scope: 'direct_query' | 'document' | 'chunk' | 'combined_context';
  assessment: 'ALLOW' | 'BLOCK' | 'REVIEW';
  rule_id: string;
  doc_id?: string | null;
  chunk_index?: number | null;
  char_start?: number | null;
  char_end?: number | null;
  risk_score: number;
  explanation: string;
}

export interface RAGInspectResponse {
  decision: 'ALLOW' | 'BLOCK' | 'REVIEW';
  decision_source: string;
  risk_score: number;
  request_id: string;
  client_request_id?: string | null;
  is_complete: boolean;
  forwarded: boolean;
  direct_query_assessment: 'ALLOW' | 'BLOCK' | 'REVIEW';
  combined_context_assessment: 'ALLOW' | 'BLOCK' | 'REVIEW';
  matched_rules: string[];
  explanation: string;
  retrieved_documents: RetrievedDocumentSummary[];
  document_assessments: DocumentInspectionSummary[];
  findings: RAGFindingSummary[];
  ml_assessment?: MLAssessment | null;
}

export interface RAGProxyResponse extends RAGInspectResponse {
  downstream_response?: MockDownstreamResponse | null;
}

export interface AuditEvent {
  event_id: string;
  request_id: string;
  client_request_id?: string | null;
  timestamp: string;
  endpoint: string;
  event_type: string;
  decision?: 'ALLOW' | 'BLOCK' | 'REVIEW' | string | null;
  decision_source?: string | null;
  matched_rule_ids: string[];
  heuristic_risk_score?: number | null;
  forwarded?: boolean | null;
  prompt_length?: number | null;
  prompt_sha256?: string | null;
  query_length?: number | null;
  query_sha256?: string | null;
  retrieved_doc_ids: string[];
  num_retrieved?: number | null;
  is_complete?: boolean | null;
  error_code?: string | null;
  outcome?: string | null;
  finding_count?: number | null;
  output_status?: string | null;
  output_findings_count?: number | null;
  output_matched_rule_ids: string[];
  output_error_code?: string | null;
  tool_name?: string | null;
  argument_count?: number | null;
  argument_keys?: string[];
  argument_sha256?: string | null;
}

export interface ToolCallRequest {
  tool_name: string;
  arguments: Record<string, unknown>;
  call_id?: string | null;
}

export interface ToolValidationResponse {
  decision: 'ALLOW' | 'BLOCK';
  error_code?: string | null;
  explanation: string;
  tool_name: string;
  category?: string | null;
  validated_arguments?: Record<string, unknown> | null;
  requires_confirmation: boolean;
}

export interface ToolExecutionResponse {
  success: boolean;
  decision: 'ALLOW' | 'BLOCK';
  tool_name: string;
  call_id?: string | null;
  output?: unknown | null;
  error_code?: string | null;
  error_message?: string | null;
  output_status?: string | null;
  output_matched_rule_ids?: string[];
}

export interface AuditQueryResponse {
  events: AuditEvent[];
  total_count: number;
  limit: number;
  offset: number;
}

export interface AuditQueryParams {
  limit?: number;
  offset?: number;
  endpoint?: string;
  decision?: string;
  event_type?: string;
  start_time?: string;
  end_time?: string;
}
