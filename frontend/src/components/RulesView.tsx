import React, { useState, useEffect, useCallback } from 'react';
import { PromptShieldApiClient, ApiError } from '../api/client';
import type { RuleInfo } from '../types/api';

interface RulesViewProps {
  apiClient: PromptShieldApiClient;
  onOpenSettings: () => void;
}

export const RulesView: React.FC<RulesViewProps> = ({ apiClient, onOpenSettings }) => {
  const [rules, setRules] = useState<RuleInfo[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [apiError, setApiError] = useState<ApiError | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [severityFilter, setSeverityFilter] = useState<string>('');
  const [actionFilter, setActionFilter] = useState<string>('');

  const fetchRules = useCallback(async () => {
    setIsLoading(true);
    setApiError(null);
    try {
      const data = await apiClient.getRules();
      setRules(data);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setApiError(err);
      } else {
        setApiError(new ApiError(0, err instanceof Error ? err.message : 'Failed to fetch rules'));
      }
      setRules([]);
    } finally {
      setIsLoading(false);
    }
  }, [apiClient]);

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void fetchRules();
  }, [fetchRules]);

  // Client-side filtering
  const filteredRules = rules.filter((r) => {
    const q = searchQuery.toLowerCase().trim();
    const matchesQuery =
      !q ||
      r.rule_id.toLowerCase().includes(q) ||
      r.name.toLowerCase().includes(q) ||
      r.description.toLowerCase().includes(q);

    const matchesSeverity = !severityFilter || r.severity === severityFilter;
    const matchesAction = !actionFilter || r.action === actionFilter;

    return matchesQuery && matchesSeverity && matchesAction;
  });

  return (
    <div className="rules-view-container">
      <div className="view-header">
        <div>
          <h2>Security Rules Inventory</h2>
          <p className="view-subtitle">
            Active deterministic heuristic and semantic defense rules enforced by the gateway
          </p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="action-btn"
            onClick={fetchRules}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : '🔄 Refresh Rules'}
          </button>
        </div>
      </div>

      {apiError && (
        <div className="alert-box alert-error">
          <div className="alert-title">
            {apiError.status === 401 ? '🔒 Authentication Required' : 'Rule Inventory Error'}
          </div>
          <p>{apiError.message}</p>
          {apiError.status === 401 && (
            <button type="button" className="secondary-btn mt-2" onClick={onOpenSettings}>
              Set In-Memory API Key in Settings
            </button>
          )}
        </div>
      )}

      {/* Filter Bar */}
      <div className="filter-card">
        <div className="filter-grid">
          <div className="filter-field flex-2">
            <label htmlFor="search-rule">Search Rules</label>
            <input
              id="search-rule"
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by ID, rule name, or pattern description..."
              className="text-input"
            />
          </div>

          <div className="filter-field">
            <label htmlFor="filter-severity">Severity</label>
            <select
              id="filter-severity"
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="select-input"
            >
              <option value="">All Severities</option>
              <option value="LOW">LOW</option>
              <option value="MEDIUM">MEDIUM</option>
              <option value="HIGH">HIGH</option>
              <option value="CRITICAL">CRITICAL</option>
            </select>
          </div>

          <div className="filter-field">
            <label htmlFor="filter-action">Action</label>
            <select
              id="filter-action"
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value)}
              className="select-input"
            >
              <option value="">All Actions</option>
              <option value="ALLOW">ALLOW</option>
              <option value="FLAG">FLAG</option>
              <option value="BLOCK">BLOCK</option>
            </select>
          </div>
        </div>

        <div className="filter-actions-row">
          <button
            type="button"
            className="secondary-btn"
            onClick={() => {
              setSearchQuery('');
              setSeverityFilter('');
              setActionFilter('');
            }}
          >
            Clear Filters
          </button>
          <span className="count-label">
            Displaying <strong>{filteredRules.length}</strong> of{' '}
            <strong>{rules.length}</strong> active rules
          </span>
        </div>
      </div>

      {/* Rules Grid */}
      {filteredRules.length === 0 ? (
        <div className="section-card empty-state">
          <p>
            {isLoading
              ? 'Loading security rules from gateway...'
              : 'No security rules match the specified criteria.'}
          </p>
        </div>
      ) : (
        <div className="rules-grid">
          {filteredRules.map((rule) => (
            <div key={rule.rule_id} className="rule-card">
              <div className="rule-card-header">
                <span className="rule-id-badge">{rule.rule_id}</span>
                <div className="rule-badges">
                  <span className={`badge-severity badge-sev-${rule.severity.toLowerCase()}`}>
                    {rule.severity}
                  </span>
                  <span className={`badge-action badge-act-${rule.action.toLowerCase()}`}>
                    {rule.action}
                  </span>
                </div>
              </div>
              <h3 className="rule-title">{rule.name}</h3>
              <p className="rule-description">{rule.description}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
