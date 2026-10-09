import React, { useEffect, useState, useCallback } from 'react';
import { PromptShieldApiClient } from '../api/client';
import type { AuditEvent, HealthResponse } from '../types/api';
import type { TabType } from './Navbar';

interface OverviewViewProps {
  apiClient: PromptShieldApiClient;
  onNavigate: (tab: TabType) => void;
  onOpenSettings: () => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  apiClient,
  onNavigate,
  onOpenSettings,
}) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [recentEvents, setRecentEvents] = useState<AuditEvent[]>([]);
  const [batchEvents, setBatchEvents] = useState<AuditEvent[]>([]);
  const [totalEventsCount, setTotalEventsCount] = useState<number>(0);
  const [auditError, setAuditError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setHealthError(null);
    setAuditError(null);

    // 1. Fetch Health
    try {
      const h = await apiClient.getHealth();
      setHealth(h);
    } catch (err: unknown) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : 'Failed to connect to gateway');
    }

    // 2. Fetch Audit Events Summary (limit: 100 for batch metrics)
    try {
      const auditRes = await apiClient.getAuditEvents({ limit: 100, offset: 0 });
      setRecentEvents(auditRes.events.slice(0, 5));
      setBatchEvents(auditRes.events);
      setTotalEventsCount(auditRes.total_count);
    } catch (err: unknown) {
      setRecentEvents([]);
      setBatchEvents([]);
      setTotalEventsCount(0);
      setAuditError(err instanceof Error ? err.message : 'Failed to retrieve audit events');
    } finally {
      setIsLoading(false);
      setLastRefreshed(new Date());
    }
  }, [apiClient]);

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void fetchData();
  }, [fetchData]);

  // Aggregate decisions from the full retrieved batch (up to 100 events)
  const batchSize = batchEvents.length;
  const allowCount = batchEvents.filter((e) => e.decision === 'ALLOW').length;
  const blockCount = batchEvents.filter((e) => e.decision === 'BLOCK').length;
  const reviewCount = batchEvents.filter((e) => e.decision === 'REVIEW').length;

  return (
    <div className="overview-container">
      <div className="view-header">
        <div>
          <h2>Security Gateway Overview</h2>
          <p className="view-subtitle">
            Real-time health, gateway telemetry, and decision metrics
          </p>
        </div>
        <div className="header-actions">
          {lastRefreshed && (
            <span className="refresh-time">
              Last updated: {lastRefreshed.toLocaleTimeString()}
            </span>
          )}
          <button
            type="button"
            className="action-btn"
            onClick={fetchData}
            disabled={isLoading}
          >
            {isLoading ? 'Refreshing...' : '🔄 Refresh Metrics'}
          </button>
        </div>
      </div>

      {healthError && (
        <div className="alert-box alert-error">
          <div className="alert-title">⚠️ Gateway Connection Error</div>
          <p>{healthError}</p>
          <div className="alert-actions">
            <button type="button" className="secondary-btn" onClick={onOpenSettings}>
              Check Gateway URL / API Key
            </button>
          </div>
        </div>
      )}

      {auditError && (
        <div className="alert-box alert-warning">
          <div className="alert-title">Notice: Audit Telemetry Unavailable</div>
          <p>{auditError}</p>
          <div className="alert-actions">
            <button type="button" className="secondary-btn" onClick={onOpenSettings}>
              Configure API Key
            </button>
          </div>
        </div>
      )}

      {/* Gateway Health & Specs Card */}
      <div className="grid-cards">
        <div className="stat-card">
          <div className="stat-label">Gateway Status</div>
          <div className="stat-value">
            {health ? (
              <span className="health-badge health-ok">● Operational</span>
            ) : (
              <span className="health-badge health-down">● Offline</span>
            )}
          </div>
          <div className="stat-meta">
            Service: <code>{health?.service || 'N/A'}</code>
            {health && <span> | Phase {health.phase}</span>}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Total Persistent Audits</div>
          <div className="stat-value text-blue">{totalEventsCount}</div>
          <div className="stat-meta">Logged across all endpoints</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Sample Allowed ({batchSize})</div>
          <div className="stat-value text-green">{allowCount}</div>
          <div className="stat-meta">In latest retrieved batch ({batchSize} events)</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Sample Blocked ({batchSize})</div>
          <div className="stat-value text-red">{blockCount}</div>
          <div className="stat-meta">In latest retrieved batch ({batchSize} events)</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Sample In-Review ({batchSize})</div>
          <div className="stat-value text-amber">{reviewCount}</div>
          <div className="stat-meta">In latest retrieved batch ({batchSize} events)</div>
        </div>
      </div>

      {/* Quick Launchpad */}
      <div className="launchpad-card">
        <div className="launchpad-item">
          <div className="launchpad-icon">🔍</div>
          <div>
            <h3>Prompt & RAG Inspection</h3>
            <p>Test heuristics, multi-tier RAG checks, and mock LLM proxy with live inputs.</p>
          </div>
          <button
            type="button"
            className="primary-btn"
            onClick={() => onNavigate('inspect')}
          >
            Launch Inspector →
          </button>
        </div>
        <div className="launchpad-divider" />
        <div className="launchpad-item">
          <div className="launchpad-icon">📋</div>
          <div>
            <h3>Audit Log Querying</h3>
            <p>Query persistent audit records with filters for decision, endpoint, and timestamps.</p>
          </div>
          <button
            type="button"
            className="primary-btn"
            onClick={() => onNavigate('audit')}
          >
            View Audit Logs →
          </button>
        </div>
        <div className="launchpad-divider" />
        <div className="launchpad-item">
          <div className="launchpad-icon">📜</div>
          <div>
            <h3>Security Rules</h3>
            <p>Browse active pattern-matching rules, severities, and defensive actions.</p>
          </div>
          <button
            type="button"
            className="primary-btn"
            onClick={() => onNavigate('rules')}
          >
            Explore Rules →
          </button>
        </div>
      </div>

      {/* Recent Events Preview */}
      <div className="section-card">
        <div className="section-card-header">
          <h3>Recent Security Events</h3>
          <button
            type="button"
            className="link-btn"
            onClick={() => onNavigate('audit')}
          >
            View full log table ({totalEventsCount} events) →
          </button>
        </div>

        {recentEvents.length === 0 ? (
          <div className="empty-state">
            <p>No recent audit events found. Send a prompt via the Inspector to generate events.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp (UTC)</th>
                  <th>Endpoint</th>
                  <th>Decision</th>
                  <th>Risk Score</th>
                  <th>Matched Rules</th>
                  <th>Event ID</th>
                </tr>
              </thead>
              <tbody>
                {recentEvents.map((evt) => (
                  <tr key={evt.event_id}>
                    <td className="mono-cell">{new Date(evt.timestamp).toLocaleString()}</td>
                    <td>
                      <code className="endpoint-tag">{evt.endpoint}</code>
                    </td>
                    <td>
                      <span className={`decision-pill decision-${(evt.decision || 'UNKNOWN').toLowerCase()}`}>
                        {evt.decision || 'N/A'}
                      </span>
                    </td>
                    <td>
                      {evt.heuristic_risk_score !== null && evt.heuristic_risk_score !== undefined
                        ? (evt.heuristic_risk_score).toFixed(2)
                        : '—'}
                    </td>
                    <td>
                      {evt.matched_rule_ids && evt.matched_rule_ids.length > 0 ? (
                        <div className="rule-pill-list">
                          {evt.matched_rule_ids.map((r) => (
                            <span key={r} className="rule-tag">
                              {r}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted">None</span>
                      )}
                    </td>
                    <td className="mono-cell text-muted" title={evt.event_id}>
                      {evt.event_id.slice(0, 8)}...
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
