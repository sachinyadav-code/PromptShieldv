import React, { useState } from 'react';
import { PromptShieldApiClient, getDefaultApiBaseUrl } from '../api/client';

interface ConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  baseUrl: string;
  apiKey: string;
  onSave: (baseUrl: string, apiKey: string) => void;
  apiClient: PromptShieldApiClient;
}

export const ConfigModal: React.FC<ConfigModalProps> = ({
  isOpen,
  onClose,
  baseUrl: initialBaseUrl,
  apiKey: initialApiKey,
  onSave,
}) => {
  const [urlInput, setUrlInput] = useState(initialBaseUrl);
  const [keyInput, setKeyInput] = useState(initialApiKey);
  const [showKey, setShowKey] = useState(false);
  const [testStatus, setTestStatus] = useState<{
    loading: boolean;
    success?: boolean;
    message?: string;
  }>({ loading: false });

  if (!isOpen) return null;

  const handleTestConnection = async () => {
    setTestStatus({ loading: true });
    try {
      const trimmedUrl = urlInput.trim();
      const trimmedKey = keyInput.trim();
      const testClient = new PromptShieldApiClient(trimmedUrl, trimmedKey);

      // 1. Verify general gateway health
      const health = await testClient.getHealth();

      // 2. If an API key is entered, test authentication against protected rules endpoint
      if (trimmedKey) {
        await testClient.getRules();
        setTestStatus({
          loading: false,
          success: true,
          message: `Connected & authenticated successfully! Service: ${health.service} (Phase ${health.phase}, status: ${health.status}). API key validated.`,
        });
      } else {
        setTestStatus({
          loading: false,
          success: true,
          message: `Connected successfully! Service: ${health.service} (Phase ${health.phase}, status: ${health.status}). Note: No API key provided; protected endpoints will require an API key.`,
        });
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown connection error';
      setTestStatus({
        loading: false,
        success: false,
        message: `Connection failed: ${msg}`,
      });
    }
  };

  const handleSave = () => {
    onSave(urlInput.trim(), keyInput.trim());
    onClose();
  };

  const handleReset = () => {
    setUrlInput(getDefaultApiBaseUrl());
    setKeyInput('');
    setTestStatus({ loading: false });
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="config-title">
      <div className="modal-content">
        <div className="modal-header">
          <h2 id="config-title">Gateway Settings</h2>
          <button type="button" className="close-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="alert-notice">
            <span className="notice-icon">🔒</span>
            <div>
              <strong>In-Memory Privacy Guarantee:</strong>
              <p>
                API keys and tokens are held strictly in memory and are never persisted to
                localStorage, cookies, or browser storage.
              </p>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="base-url-input">Backend Gateway Base URL</label>
            <input
              id="base-url-input"
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="http://127.0.0.1:8000"
              className="text-input"
            />
            <span className="field-hint">
              Target address of the PromptShield FastAPI gateway backend.
            </span>
          </div>

          <div className="form-group">
            <label htmlFor="api-key-input">Admin / Gateway API Key</label>
            <div className="input-with-button">
              <input
                id="api-key-input"
                type={showKey ? 'text' : 'password'}
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                placeholder="Enter API key (e.g., test-key-12345)"
                className="text-input"
                autoComplete="off"
              />
              <button
                type="button"
                className="secondary-btn"
                onClick={() => setShowKey(!showKey)}
                aria-label={showKey ? 'Hide key' : 'Show key'}
              >
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
            <span className="field-hint">
              Required for endpoints with rate limits or authentication. Sent via{' '}
              <code>Authorization: Bearer &lt;key&gt;</code>.
            </span>
          </div>

          <div className="connection-test-row">
            <button
              type="button"
              className="action-btn"
              onClick={handleTestConnection}
              disabled={testStatus.loading}
            >
              {testStatus.loading ? 'Testing...' : 'Test Connection & Auth'}
            </button>
            {testStatus.message && (
              <div
                className={`test-result-box ${testStatus.success ? 'test-success' : 'test-failure'}`}
              >
                {testStatus.message}
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="secondary-btn" onClick={handleReset}>
            Reset Defaults
          </button>
          <div className="footer-right">
            <button type="button" className="secondary-btn" onClick={onClose}>
              Cancel
            </button>
            <button type="button" className="primary-btn" onClick={handleSave}>
              Save & Apply
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
