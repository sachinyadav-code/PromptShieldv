import React from 'react';

export type TabType = 'overview' | 'inspect' | 'tools' | 'audit' | 'rules';

interface NavbarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  isConnected: boolean | null;
  hasApiKey: boolean;
  baseUrl: string;
  onOpenSettings: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onTabChange,
  isConnected,
  hasApiKey,
  baseUrl,
  onOpenSettings,
}) => {
  return (
    <header className="navbar">
      <div className="navbar-brand">
        <div className="shield-icon" aria-hidden="true">
          🛡️
        </div>
        <div className="brand-text">
          <span className="brand-title">PromptShield</span>
          <span className="brand-subtitle">Security Gateway</span>
        </div>
      </div>

      <nav className="navbar-nav" aria-label="Main Navigation">
        <button
          type="button"
          className={`nav-tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => onTabChange('overview')}
        >
          Overview
        </button>
        <button
          type="button"
          className={`nav-tab ${activeTab === 'inspect' ? 'active' : ''}`}
          onClick={() => onTabChange('inspect')}
        >
          Inspect & Proxy
        </button>
        <button
          type="button"
          className={`nav-tab ${activeTab === 'tools' ? 'active' : ''}`}
          onClick={() => onTabChange('tools')}
        >
          Tool Guard & DLP
        </button>
        <button
          type="button"
          className={`nav-tab ${activeTab === 'audit' ? 'active' : ''}`}
          onClick={() => onTabChange('audit')}
        >
          Audit Events
        </button>
        <button
          type="button"
          className={`nav-tab ${activeTab === 'rules' ? 'active' : ''}`}
          onClick={() => onTabChange('rules')}
        >
          Rules Inventory
        </button>
      </nav>

      <div className="navbar-actions">
        <div
          className={`status-pill ${
            isConnected === true
              ? 'status-connected'
              : isConnected === false
                ? 'status-disconnected'
                : 'status-checking'
          }`}
          title={`Base URL: ${baseUrl}`}
        >
          <span className="status-dot" aria-hidden="true"></span>
          <span className="status-label">
            {isConnected === true
              ? 'Connected'
              : isConnected === false
                ? 'Offline'
                : 'Connecting...'}
          </span>
        </div>

        <button
          type="button"
          className="settings-button"
          onClick={onOpenSettings}
          title="Configure API Base URL and In-Memory API Key"
        >
          ⚙️ Settings {hasApiKey ? '🔑' : ''}
        </button>
      </div>
    </header>
  );
};
