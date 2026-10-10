import React, { useState, useEffect, useMemo } from 'react';
import { PromptShieldApiClient, getDefaultApiBaseUrl } from './api/client';
import { Navbar } from './components/Navbar';
import type { TabType } from './components/Navbar';
import { ConfigModal } from './components/ConfigModal';
import { OverviewView } from './components/OverviewView';
import { InspectView } from './components/InspectView';
import { AuditEventsView } from './components/AuditEventsView';
import { RulesView } from './components/RulesView';
import './App.css';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [baseUrl, setBaseUrl] = useState<string>(getDefaultApiBaseUrl());
  const [apiKey, setApiKey] = useState<string>('');
  const [isConfigOpen, setIsConfigOpen] = useState<boolean>(false);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);

  // Instantiates the API client with current in-memory credentials
  const apiClient = useMemo(() => {
    return new PromptShieldApiClient(baseUrl, apiKey);
  }, [baseUrl, apiKey]);

  // Check health periodically or on baseUrl change
  useEffect(() => {
    let isCancelled = false;
    const check = async () => {
      try {
        const health = await apiClient.getHealth();
        if (!isCancelled) {
          setIsConnected(health.status === 'ok');
        }
      } catch {
        if (!isCancelled) {
          setIsConnected(false);
        }
      }
    };

    void check();
    const timer = setInterval(check, 30000);
    return () => {
      isCancelled = true;
      clearInterval(timer);
    };
  }, [apiClient]);

  const handleSaveConfig = (newBaseUrl: string, newApiKey: string) => {
    setBaseUrl(newBaseUrl);
    setApiKey(newApiKey);
  };

  return (
    <div className="app-layout">
      <Navbar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isConnected={isConnected}
        hasApiKey={apiKey.length > 0}
        baseUrl={baseUrl}
        onOpenSettings={() => setIsConfigOpen(true)}
      />

      <main className="main-content" role="main">
        {activeTab === 'overview' && (
          <OverviewView
            apiClient={apiClient}
            onNavigate={setActiveTab}
            onOpenSettings={() => setIsConfigOpen(true)}
          />
        )}

        {activeTab === 'inspect' && (
          <InspectView
            apiClient={apiClient}
            onOpenSettings={() => setIsConfigOpen(true)}
          />
        )}

        {activeTab === 'audit' && (
          <AuditEventsView
            apiClient={apiClient}
            onOpenSettings={() => setIsConfigOpen(true)}
          />
        )}

        {activeTab === 'rules' && (
          <RulesView
            apiClient={apiClient}
            onOpenSettings={() => setIsConfigOpen(true)}
          />
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <span>PromptShield Enterprise Security Gateway &copy; 2026</span>
          <span className="footer-status">
            Connected to <code>{baseUrl}</code> &bull; In-memory security isolation active
          </span>
        </div>
      </footer>

      {isConfigOpen && (
        <ConfigModal
          isOpen={isConfigOpen}
          onClose={() => setIsConfigOpen(false)}
          baseUrl={baseUrl}
          apiKey={apiKey}
          onSave={handleSaveConfig}
          apiClient={apiClient}
        />
      )}
    </div>
  );
};

export default App;
