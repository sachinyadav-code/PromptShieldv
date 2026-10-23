import React, { useState } from 'react';
import { PromptShieldApiClient, ApiError } from '../api/client';
import type { ToolExecutionResponse, ToolValidationResponse } from '../types/api';

type ToolEndpointMode = 'execute' | 'validate';

interface ToolsViewProps {
  apiClient: PromptShieldApiClient;
  onOpenSettings: () => void;
}

interface DemoScenarioPreset {
  id: string;
  name: string;
  badge: string;
  endpoint: ToolEndpointMode;
  toolName: string;
  argumentsJson: string;
  description: string;
}

const DEMO_PRESETS: DemoScenarioPreset[] = [
  {
    id: 'calc_valid',
    name: '1. Valid Tool Execution (Calculator)',
    badge: 'ALLOW',
    endpoint: 'execute',
    toolName: 'calculator',
    argumentsJson: JSON.stringify({ operation: 'multiply', a: 12.5, b: 4 }, null, 2),
    description: 'Executes allowed calculator tool safely. Output DLP verifies no sensitive data leakage.',
  },
  {
    id: 'sample_valid',
    name: '2. Valid Data Lookup (Item 001)',
    badge: 'ALLOW',
    endpoint: 'execute',
    toolName: 'sample_data_lookup',
    argumentsJson: JSON.stringify({ record_id: 'item_001' }, null, 2),
    description: 'Deterministic lookup in read-only dataset with strict regex-validated identifier.',
  },
  {
    id: 'denied_tool',
    name: '3. Denied Unknown Tool (system_shell)',
    badge: 'BLOCK',
    endpoint: 'execute',
    toolName: 'system_shell',
    argumentsJson: JSON.stringify({ cmd: 'whoami' }, null, 2),
    description: 'Deny-by-default allowlist rejection (TOOL_NOT_ALLOWED). Handler never invoked.',
  },
  {
    id: 'invalid_args_div_zero',
    name: '4. Invalid Arguments (Division by Zero)',
    badge: 'VALIDATION',
    endpoint: 'execute',
    toolName: 'calculator',
    argumentsJson: JSON.stringify({ operation: 'divide', a: 100, b: 0 }, null, 2),
    description: 'Safe structured failure handling without unhandled exceptions or service disruption.',
  },
  {
    id: 'validate_only',
    name: '5. Dry-Run Validation (/validate)',
    badge: 'ISOLATION',
    endpoint: 'validate',
    toolName: 'calculator',
    argumentsJson: JSON.stringify({ operation: 'add', a: 50, b: 25 }, null, 2),
    description: 'Validates schemas and permissions without executing the underlying handler.',
  },
];

export const ToolsView: React.FC<ToolsViewProps> = ({ apiClient, onOpenSettings }) => {
  const [endpointMode, setEndpointMode] = useState<ToolEndpointMode>('execute');
  const [toolName, setToolName] = useState<string>('calculator');
  const [customToolName, setCustomToolName] = useState<string>('');
  const [argumentsText, setArgumentsText] = useState<string>(
    JSON.stringify({ operation: 'multiply', a: 12.5, b: 4 }, null, 2)
  );
  const [callId, setCallId] = useState<string>('demo_call_001');

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<ApiError | null>(null);

  const [validationResult, setValidationResult] = useState<ToolValidationResponse | null>(null);
  const [executionResult, setExecutionResult] = useState<ToolExecutionResponse | null>(null);
  const [showRawJson, setShowRawJson] = useState<boolean>(false);

  const effectiveToolName = toolName === '__custom__' ? customToolName.trim() : toolName;

  const handleApplyPreset = (preset: DemoScenarioPreset) => {
    setEndpointMode(preset.endpoint);
    if (preset.toolName === 'calculator' || preset.toolName === 'sample_data_lookup') {
      setToolName(preset.toolName);
      setCustomToolName('');
    } else {
      setToolName('__custom__');
      setCustomToolName(preset.toolName);
    }
    setArgumentsText(preset.argumentsJson);
    setJsonError(null);
    setApiError(null);
    setValidationResult(null);
    setExecutionResult(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setJsonError(null);
    setApiError(null);
    setValidationResult(null);
    setExecutionResult(null);

    if (!effectiveToolName) {
      setJsonError('Tool name cannot be empty.');
      return;
    }

    let parsedArgs: Record<string, unknown>;
    try {
      parsedArgs = JSON.parse(argumentsText);
      if (typeof parsedArgs !== 'object' || parsedArgs === null || Array.isArray(parsedArgs)) {
        setJsonError('Arguments must be a valid JSON object (dictionary).');
        return;
      }
    } catch (err: unknown) {
      setJsonError(`Malformed JSON syntax: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }

    if (!apiClient.hasApiKey()) {
      setApiError(
        new ApiError(
          401,
          'Authentication required: Please configure a valid API key in Settings.'
        )
      );
      return;
    }

    setIsLoading(true);
    try {
      if (endpointMode === 'validate') {
        const res = await apiClient.validateTool({
          tool_name: effectiveToolName,
          arguments: parsedArgs,
          call_id: callId.trim() || undefined,
        });
        setValidationResult(res);
      } else {
        const res = await apiClient.executeTool({
          tool_name: effectiveToolName,
          arguments: parsedArgs,
          call_id: callId.trim() || undefined,
        });
        setExecutionResult(res);
      }
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setApiError(err);
      } else {
        setApiError(
          new ApiError(0, err instanceof Error ? err.message : 'Unknown tool request failure')
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="view-container">
      <div className="view-header">
        <div>
          <h2>Tool Guard &amp; Output DLP</h2>
          <p className="view-subtitle">
            Deny-by-default allowlisting, strict schema enforcement, and fail-closed Data Loss Prevention
          </p>
        </div>
      </div>

      {/* Preset Demo Scenarios */}
      <div className="preset-bar">
        <span className="preset-label">Safe Demo Scenarios:</span>
        <div className="preset-buttons">
          {DEMO_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              className="preset-btn"
              onClick={() => handleApplyPreset(p)}
              title={p.description}
            >
              <span className={`badge-pill badge-${p.badge.toLowerCase()}`}>{p.badge}</span>
              {p.name}
            </button>
          ))}
        </div>
      </div>

      <div className="workbench-grid">
        {/* Left Panel: Form */}
        <div className="workbench-card">
          <h3>Tool Execution Request</h3>
          <form onSubmit={handleSubmit} className="inspect-form">
            <div className="form-group">
              <label htmlFor="endpoint-mode">Action Mode</label>
              <div className="endpoint-selector">
                <label className="radio-pill">
                  <input
                    type="radio"
                    name="endpoint-mode"
                    value="execute"
                    checked={endpointMode === 'execute'}
                    onChange={() => setEndpointMode('execute')}
                  />
                  <span>Execute Tool (/tools/execute)</span>
                </label>
                <label className="radio-pill">
                  <input
                    type="radio"
                    name="endpoint-mode"
                    value="validate"
                    checked={endpointMode === 'validate'}
                    onChange={() => setEndpointMode('validate')}
                  />
                  <span>Dry-Run Validate (/tools/validate)</span>
                </label>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="tool-select">Target Tool</label>
              <select
                id="tool-select"
                className="select-input"
                value={toolName}
                onChange={(e) => setToolName(e.target.value)}
              >
                <option value="calculator">calculator (Allowlisted Arithmetic)</option>
                <option value="sample_data_lookup">sample_data_lookup (Allowlisted Read-Only Data)</option>
                <option value="__custom__">[Test Rejection] Custom / Untrusted Tool...</option>
              </select>
            </div>

            {toolName === '__custom__' && (
              <div className="form-group">
                <label htmlFor="custom-tool-input">Custom Tool Name</label>
                <input
                  id="custom-tool-input"
                  type="text"
                  className="text-input"
                  placeholder="e.g. bash_shell, eval, sql_query"
                  value={customToolName}
                  onChange={(e) => setCustomToolName(e.target.value)}
                />
              </div>
            )}

            <div className="form-group">
              <label htmlFor="call-id-input">Correlation Call ID (Optional)</label>
              <input
                id="call-id-input"
                type="text"
                className="text-input"
                placeholder="e.g. call_12345"
                value={callId}
                onChange={(e) => setCallId(e.target.value)}
              />
            </div>

            <div className="form-group">
              <div className="label-row">
                <label htmlFor="arguments-input">Tool Arguments (JSON Object)</label>
                <span className="char-count">Max 50 keys</span>
              </div>
              <textarea
                id="arguments-input"
                className="textarea-input mono-font"
                rows={6}
                value={argumentsText}
                onChange={(e) => setArgumentsText(e.target.value)}
              />
              {jsonError && <div className="field-error">{jsonError}</div>}
            </div>

            <div className="form-actions">
              <button
                type="submit"
                className="primary-btn"
                disabled={isLoading}
              >
                {isLoading ? 'Inspecting...' : endpointMode === 'execute' ? 'Execute Tool Guard' : 'Validate Call'}
              </button>
            </div>
          </form>
        </div>

        {/* Right Panel: Security & Execution Decision */}
        <div className="workbench-card">
          <h3>Security &amp; Policy Decision</h3>

          {isLoading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Evaluating allowlist, schemas, and output DLP policies...</p>
            </div>
          )}

          {apiError && !isLoading && (
            <div className="error-card">
              <div className="error-title">
                <span>⚠️</span> Request Failed (HTTP {apiError.status || 'Error'})
              </div>
              <p className="error-message">{apiError.message}</p>
              {apiError.status === 401 && (
                <button
                  type="button"
                  className="settings-link-btn"
                  onClick={onOpenSettings}
                >
                  Configure In-Memory API Key in Settings ➔
                </button>
              )}
            </div>
          )}

          {!isLoading && !apiError && !validationResult && !executionResult && (
            <div className="empty-state">
              <div className="empty-icon">🛡️</div>
              <p>Select a scenario preset above or enter arguments to test tool allowlisting and output DLP.</p>
            </div>
          )}

          {!isLoading && (validationResult || executionResult) && (
            <div className="result-container">
              {/* Decision Banner */}
              <div
                className={`decision-banner decision-${
                  (executionResult?.decision || validationResult?.decision || 'BLOCK').toLowerCase()
                }`}
              >
                <div className="banner-badge">
                  {executionResult?.decision || validationResult?.decision}
                </div>
                <div className="banner-details">
                  <div className="banner-reason">
                    {validationResult?.explanation ||
                      (executionResult?.success
                        ? 'Tool executed safely and passed output DLP scanning.'
                        : executionResult?.error_message || 'Tool call blocked or failed.')}
                  </div>
                  <div className="banner-source">
                    Tool: <code>{executionResult?.tool_name || validationResult?.tool_name}</code>
                    {(executionResult?.error_code || validationResult?.error_code) && (
                      <span className="error-code-tag">
                        Code: {executionResult?.error_code || validationResult?.error_code}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Output DLP Status */}
              {executionResult?.output_status && (
                <div className="dlp-status-card">
                  <h4>Output DLP Inspection</h4>
                  <div className="dlp-details">
                    <span className={`dlp-badge dlp-${executionResult.output_status.toLowerCase()}`}>
                      Status: {executionResult.output_status}
                    </span>
                    {executionResult.output_matched_rule_ids && executionResult.output_matched_rule_ids.length > 0 && (
                      <div className="matched-rules">
                        <span>Matched Rules:</span>
                        {executionResult.output_matched_rule_ids.map((r, i) => (
                          <span key={i} className="rule-tag">{r}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Output Payload */}
              {executionResult && (
                <div className="output-section">
                  <h4>Returned Tool Output</h4>
                  {executionResult.output !== null && executionResult.output !== undefined ? (
                    <pre className="json-dump">
                      {JSON.stringify(executionResult.output, null, 2)}
                    </pre>
                  ) : (
                    <div className="redacted-notice">
                      <span>🔒</span> Output suppressed or null due to security block or rejection.
                    </div>
                  )}
                </div>
              )}

              {/* Raw JSON Toggle */}
              <div className="raw-json-section">
                <button
                  type="button"
                  className="link-btn"
                  onClick={() => setShowRawJson(!showRawJson)}
                >
                  {showRawJson ? '▼ Hide Raw Response' : '▶ Show Raw JSON Response'}
                </button>
                {showRawJson && (
                  <pre className="json-dump">
                    {JSON.stringify(executionResult || validationResult, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
