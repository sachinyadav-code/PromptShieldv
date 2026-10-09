import React, { useState, useEffect, useCallback } from 'react';
import { PromptShieldApiClient, ApiError } from '../api/client';
import type { AuditEvent, AuditQueryParams } from '../types/api';

interface AuditEventsViewProps {
  apiClient: PromptShieldApiClient;
  onOpenSettings: () => void;
}

export const AuditEventsView: React.FC<AuditEventsViewProps> = ({
  apiClient,
  onOpenSettings,
}) => {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [limit, setLimit] = useState<number>(25);
  const [offset, setOffset] = useState<number>(0);

  // Filters
  const [filterEndpoint, setFilterEndpoint] = useState<string>('');
  const [filterDecision, setFilterDecision] = useState<string>('');
  const [filterEventType, setFilterEventType] = useState<string>('');
  const [filterStartTime, setFilterStartTime] = useState<string>('');
  const [filterEndTime, setFilterEndTime] = useState<string>('');

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [apiError, setApiError] = useState<ApiError | null>(null);
  const [dateValidationError, setDateValidationError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<AuditEvent | null>(null);

  const fetchEvents = useCallback(async () => {
    setIsLoading(true);
    setApiError(null);
    setDateValidationError(null);

    const params: AuditQueryParams = {
      limit,
      offset,
    };
    if (filterEndpoint) params.endpoint = filterEndpoint;
    if (filterDecision) params.decision = filterDecision;
    if (filterEventType) params.event_type = filterEventType;

    if (filterStartTime.trim()) {
      const parsedStart = new Date(filterStartTime);
      if (!Number.isNaN(parsedStart.getTime())) {
        params.start_time = parsedStart.toISOString();
      } else {
        setDateValidationError('Invalid Start Time format. Please enter a valid timestamp.');
        setIsLoading(false);
        return;
      }
    }

    if (filterEndTime.trim()) {
      const parsedEnd = new Date(filterEndTime);
      if (!Number.isNaN(parsedEnd.getTime())) {
        params.end_time = parsedEnd.toISOString();
      } else {
        setDateValidationError('Invalid End Time format. Please enter a valid timestamp.');
        setIsLoading(false);
        return;
      }
    }

    try {
      const res = await apiClient.getAuditEvents(params);
      setEvents(res.events);
      setTotalCount(res.total_count);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setApiError(err);
      } else {
        setApiError(new ApiError(0, err instanceof Error ? err.message : 'Failed to query audits'));
      }
      setEvents([]);
      setTotalCount(0);
    } finally {
      setIsLoading(false);
    }
  }, [
    apiClient,
    limit,
    offset,
    filterEndpoint,
    filterDecision,
    filterEventType,
    filterStartTime,
    filterEndTime,
  ]);

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void fetchEvents();
  }, [fetchEvents]);

  const handleResetFilters = () => {
    setFilterEndpoint('');
    setFilterDecision('');
    setFilterEventType('');
    setFilterStartTime('');
    setFilterEndTime('');
    setDateValidationError(null);
    setOffset(0);
  };

  const handleNextPage = () => {
    if (offset + limit < totalCount) {
      setOffset(offset + limit);
    }
  };

  const handlePrevPage = () => {
    if (offset > 0) {
      setOffset(Math.max(0, offset - limit));
    }
  };

  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.ceil(totalCount / limit) || 1;

  return (
    <div className="audit-view-container">
      <div className="view-header">
        <div>
          <h2>Persistent Security Audit Logs</h2>
          <p className="view-subtitle">
            Search, filter, and inspect tamper-evident audit records across gateway endpoints
          </p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="action-btn"
            onClick={fetchEvents}
            disabled={isLoading}
          >
            {isLoading ? 'Querying...' : '🔄 Refresh Log Table'}
          </button>
        </div>
      </div>

      {dateValidationError && (
        <div className="alert-box alert-error">
          <div className="alert-title">⚠️ Invalid Date Filter</div>
          <p>{dateValidationError}</p>
        </div>
      )}

      {apiError && (
        <div className="alert-box alert-error">
          <div className="alert-title">
            {apiError.status === 401 ? '🔒 Authentication Required' : 'Audit Query Error'}
          </div>
          <p>{apiError.message}</p>
          {apiError.status === 401 && (
            <button type="button" className="secondary-btn mt-2" onClick={onOpenSettings}>
              Set In-Memory API Key in Settings
            </button>
          )}
        </div>
      )}

      {/* Filter Toolbar */}
      <div className="filter-card">
        <div className="filter-grid">
          <div className="filter-field">
            <label htmlFor="filter-endpoint">Endpoint</label>
            <select
              id="filter-endpoint"
              value={filterEndpoint}
              onChange={(e) => {
                setFilterEndpoint(e.target.value);
                setOffset(0);
              }}
              className="select-input"
            >
              <option value="">All Endpoints</option>
              <option value="/api/v1/inspect">/api/v1/inspect</option>
              <option value="/api/v1/proxy">/api/v1/proxy</option>
              <option value="/api/v1/rag/inspect">/api/v1/rag/inspect</option>
              <option value="/api/v1/rag/proxy">/api/v1/rag/proxy</option>
            </select>
          </div>

          <div className="filter-field">
            <label htmlFor="filter-decision">Decision</label>
            <select
              id="filter-decision"
              value={filterDecision}
              onChange={(e) => {
                setFilterDecision(e.target.value);
                setOffset(0);
              }}
              className="select-input"
            >
              <option value="">All Decisions</option>
              <option value="ALLOW">ALLOW</option>
              <option value="BLOCK">BLOCK</option>
              <option value="REVIEW">REVIEW</option>
            </select>
          </div>

          <div className="filter-field">
            <label htmlFor="filter-event-type">Event Type</label>
            <input
              id="filter-event-type"
              type="text"
              value={filterEventType}
              onChange={(e) => {
                setFilterEventType(e.target.value);
                setOffset(0);
              }}
              placeholder="e.g. INSPECTION_DECISION"
              className="text-input"
            />
          </div>

          <div className="filter-field">
            <label htmlFor="filter-start-time">Start Time</label>
            <input
              id="filter-start-time"
              type="datetime-local"
              value={filterStartTime}
              onChange={(e) => {
                setFilterStartTime(e.target.value);
                setOffset(0);
              }}
              className="text-input"
            />
          </div>

          <div className="filter-field">
            <label htmlFor="filter-end-time">End Time</label>
            <input
              id="filter-end-time"
              type="datetime-local"
              value={filterEndTime}
              onChange={(e) => {
                setFilterEndTime(e.target.value);
                setOffset(0);
              }}
              className="text-input"
            />
          </div>
        </div>

        <div className="filter-actions-row">
          <button
            type="button"
            className="secondary-btn"
            onClick={handleResetFilters}
          >
            Clear Filters
          </button>
          <div className="pagination-limit-wrapper">
            <label htmlFor="limit-select">Rows per page:</label>
            <select
              id="limit-select"
              value={limit}
              onChange={(e) => {
                setLimit(parseInt(e.target.value, 10));
                setOffset(0);
              }}
              className="select-input small-select"
            >
              <option value="10">10</option>
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
          </div>
        </div>
      </div>

      {/* Audit Log Table */}
      <div className="section-card">
        <div className="table-header-meta">
          <span>
            Showing <strong>{events.length > 0 ? offset + 1 : 0}</strong> to{' '}
            <strong>{Math.min(offset + limit, totalCount)}</strong> of{' '}
            <strong>{totalCount}</strong> records
          </span>
          <div className="pagination-buttons">
            <button
              type="button"
              className="secondary-btn"
              onClick={handlePrevPage}
              disabled={offset === 0 || isLoading}
            >
              ← Previous
            </button>
            <span className="page-indicator">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              className="secondary-btn"
              onClick={handleNextPage}
              disabled={offset + limit >= totalCount || isLoading}
            >
              Next →
            </button>
          </div>
        </div>

        {events.length === 0 ? (
          <div className="empty-state">
            <p>
              {isLoading
                ? 'Loading audit records from persistent store...'
                : 'No audit records match the current query filters.'}
            </p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp (UTC)</th>
                  <th>Endpoint</th>
                  <th>Decision</th>
                  <th>Type</th>
                  <th>Risk</th>
                  <th>Matched Rules</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {events.map((evt) => (
                  <tr key={evt.event_id}>
                    <td className="mono-cell">{new Date(evt.timestamp).toLocaleString()}</td>
                    <td>
                      <code className="endpoint-tag">{evt.endpoint}</code>
                    </td>
                    <td>
                      <span className={`decision-pill decision-${(evt.decision || 'UNKNOWN').toLowerCase()}`}>
                        {evt.decision || '—'}
                      </span>
                    </td>
                    <td>
                      <span className="text-secondary">{evt.event_type}</span>
                    </td>
                    <td>
                      {evt.heuristic_risk_score !== null && evt.heuristic_risk_score !== undefined
                        ? (evt.heuristic_risk_score).toFixed(2)
                        : '—'}
                    </td>
                    <td>
                      {evt.matched_rule_ids && evt.matched_rule_ids.length > 0 ? (
                        <div className="rule-pill-list">
                          {evt.matched_rule_ids.slice(0, 3).map((r) => (
                            <span key={r} className="rule-tag">
                              {r}
                            </span>
                          ))}
                          {evt.matched_rule_ids.length > 3 && (
                            <span className="text-muted">
                              +{evt.matched_rule_ids.length - 3} more
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-muted">None</span>
                      )}
                    </td>
                    <td>
                      <button
                        type="button"
                        className="small-action-btn"
                        onClick={() => setSelectedEvent(evt)}
                      >
                        Inspect Record
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedEvent && (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="audit-detail-title"
        >
          <div className="modal-content modal-large">
            <div className="modal-header">
              <h2 id="audit-detail-title">
                Audit Event Detail: <code>{selectedEvent.event_id}</code>
              </h2>
              <button
                type="button"
                className="close-btn"
                onClick={() => setSelectedEvent(null)}
                aria-label="Close"
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-label">Decision:</span>
                  <span className={`decision-pill decision-${(selectedEvent.decision || 'UNKNOWN').toLowerCase()}`}>
                    {selectedEvent.decision || 'N/A'}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Endpoint:</span>
                  <code>{selectedEvent.endpoint}</code>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Timestamp:</span>
                  <span>{new Date(selectedEvent.timestamp).toISOString()}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Event Type:</span>
                  <span>{selectedEvent.event_type}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Risk Score:</span>
                  <span>
                    {selectedEvent.heuristic_risk_score !== null &&
                    selectedEvent.heuristic_risk_score !== undefined
                      ? selectedEvent.heuristic_risk_score.toFixed(2)
                      : 'N/A'}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Forwarded Downstream:</span>
                  <span>{selectedEvent.forwarded ? 'Yes' : 'No'}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Request ID:</span>
                  <code className="mono-cell">{selectedEvent.request_id}</code>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Client Request ID:</span>
                  <span>{selectedEvent.client_request_id || '—'}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Prompt Length:</span>
                  <span>
                    {selectedEvent.prompt_length !== null && selectedEvent.prompt_length !== undefined
                      ? `${selectedEvent.prompt_length} chars`
                      : '—'}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Prompt SHA256:</span>
                  <span className="mono-cell text-muted" title={selectedEvent.prompt_sha256 || ''}>
                    {selectedEvent.prompt_sha256 || '—'}
                  </span>
                </div>
                {selectedEvent.query_sha256 && (
                  <div className="detail-item">
                    <span className="detail-label">Query SHA256:</span>
                    <span className="mono-cell text-muted">{selectedEvent.query_sha256}</span>
                  </div>
                )}
                {selectedEvent.retrieved_doc_ids && selectedEvent.retrieved_doc_ids.length > 0 && (
                  <div className="detail-item full-width">
                    <span className="detail-label">Retrieved Doc IDs:</span>
                    <div className="rule-pill-list">
                      {selectedEvent.retrieved_doc_ids.map((id) => (
                        <span key={id} className="rule-tag">
                          {id}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="detail-item full-width">
                  <span className="detail-label">Matched Rule IDs:</span>
                  {selectedEvent.matched_rule_ids && selectedEvent.matched_rule_ids.length > 0 ? (
                    <div className="rule-pill-list">
                      {selectedEvent.matched_rule_ids.map((r) => (
                        <span key={r} className="rule-tag matched-rule">
                          {r}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-muted">None</span>
                  )}
                </div>
              </div>

              <div className="raw-json-section mt-4">
                <h4>Raw Audit Record (JSON)</h4>
                <pre className="json-dump">
                  {JSON.stringify(selectedEvent, null, 2)}
                </pre>
              </div>
            </div>

            <div className="modal-footer">
              <button
                type="button"
                className="secondary-btn"
                onClick={() => setSelectedEvent(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
