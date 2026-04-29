import { useState } from "react";
import GeneralSection from "../components/settings/GeneralSection";
import AppearanceSection from "../components/settings/AppearanceSection";
import DiscordRPCSection from "../components/settings/DiscordRPCSection";
import SceneDetectionSection from "../components/settings/SceneDetectionSection";
import DeveloperSection from "../components/settings/DeveloperSection";
import { type ThemeSettings } from "../settings/themeSettings";
import { type GeneralSettings } from "../settings/generalSettings";

type SettingsProps = {
  themeSettings: ThemeSettings;
  setThemeSettings: React.Dispatch<React.SetStateAction<ThemeSettings>>;
  generalSettings: GeneralSettings;
  setGeneralSettings: React.Dispatch<React.SetStateAction<GeneralSettings>>;
  onGeneralSettingsReset: () => void;
  onEpisodesPathChanged: (oldPath: string, newPath: string) => void;
  onThemeReset: () => void;
};

export default function Settings({
  themeSettings,
  setThemeSettings,
  generalSettings,
  setGeneralSettings,
  onGeneralSettingsReset,
  onEpisodesPathChanged,
  onThemeReset,
}: SettingsProps) {
  const [activeTab, setActiveTab] = useState("general");

  const PAGES = [
    { key: "general", label: "General" },
    { key: "appearance", label: "Appearance" },
    { key: "scene_detection", label: "Scene Detection" },
    { key: "discord", label: "Discord RPC" },
    ...(generalSettings.developerMode ? [{ key: "developer", label: "Developer" }] : []),
  ];

  return (
    <div className="menu-page">
      <div className="menu-header">
        <h2 className="menu-title">Settings</h2>

        <div className="menu-nav">
          {PAGES.map((page) => (
            <button
              key={page.key}
              className={`menu-nav-btn${activeTab === page.key ? " active" : ""}`}
              onClick={() => setActiveTab(page.key)}
            >
              {page.label}
            </button>
          ))}
        </div>
      </div>

      <div className="menu-content">
        <div className="menu-section">
          <div className="tab-content" style={{ flex: 1 }}>
            {activeTab === "general" && (
              <GeneralSection
                generalSettings={generalSettings}
                setGeneralSettings={setGeneralSettings}
                onGeneralSettingsReset={onGeneralSettingsReset}
                onEpisodesPathChanged={onEpisodesPathChanged}
              />
            )}

            {activeTab === "appearance" && (
              <AppearanceSection
                themeSettings={themeSettings}
                setThemeSettings={setThemeSettings}
                onThemeReset={onThemeReset}
              />
            )}

            {activeTab === "scene_detection" && (
              <SceneDetectionSection
                generalSettings={generalSettings}
                setGeneralSettings={setGeneralSettings}
              />
            )}

            {activeTab === "discord" && (
              <DiscordRPCSection
                generalSettings={generalSettings}
                setGeneralSettings={setGeneralSettings}
              />
            )}

            {activeTab === "developer" && generalSettings.developerMode && (
              <DeveloperSection />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}