import React, { useState } from 'react'
import {
  Activity,
  BarChart3,
  Box,
  Cpu,
  FlaskConical,
  LayoutDashboard,
  Play,
  RotateCcw,
  Settings,
  ShieldAlert,
  Sliders,
  Square,
  Terminal,
} from 'lucide-react'
import { api } from './services/api'
import { useSimulationSocket } from './hooks/useSimulationSocket'
import type { NavSection, ViewportOptions } from './types/robotics'
import { FrankaPandaViewport } from './features/simulation/FrankaPandaViewport'
import { RobotControlPanel } from './features/robot-control/RobotControlPanel'
import { TaskExecutionPanel } from './features/task-execution/TaskExecutionPanel'
import { EnvironmentPanel } from './features/environment/EnvironmentPanel'
import { PoliciesPanel } from './features/reinforcement-learning/PoliciesPanel'
import { ExperimentsAndEvalPanel } from './features/experiments/ExperimentsAndEvalPanel'
import { DiagnosticsAndOverviewPanel } from './features/diagnostics/DiagnosticsAndOverviewPanel'

const NAV_ITEMS: { id: NavSection; label: string; icon: React.ReactNode }[] = [
  { id: 'overview', label: 'Overview', icon: <LayoutDashboard className="h-3.5 w-3.5" /> },
  { id: 'simulation', label: 'Simulation', icon: <Box className="h-3.5 w-3.5" /> },
  { id: 'robot-control', label: 'Robot Control', icon: <Sliders className="h-3.5 w-3.5" /> },
  { id: 'task-execution', label: 'Task Execution', icon: <Play className="h-3.5 w-3.5" /> },
  { id: 'rl', label: 'Reinforcement Learning', icon: <Cpu className="h-3.5 w-3.5" /> },
  { id: 'environment', label: 'Environment Config', icon: <Activity className="h-3.5 w-3.5" /> },
  { id: 'experiments', label: 'Experiments', icon: <FlaskConical className="h-3.5 w-3.5" /> },
  { id: 'evaluation', label: 'Evaluation', icon: <BarChart3 className="h-3.5 w-3.5" /> },
  { id: 'diagnostics', label: 'Logs & Diagnostics', icon: <Terminal className="h-3.5 w-3.5" /> },
  { id: 'settings', label: 'Settings', icon: <Settings className="h-3.5 w-3.5" /> },
]

export function App() {
  const [section, setSection] = useState<NavSection>('simulation')
  const { snapshot, events, connected, wsError } = useSimulationSocket()
  const [bannerMsg, setBannerMsg] = useState<{ text: string; isErr: boolean } | null>(null)
  const [vpOpts, setVpOpts] = useState<ViewportOptions>({
    showGrid: true,
    showFrames: true,
    showCollision: false,
    showTrajectory: true,
    showLabels: true,
    showEeFrame: true,
  })

  const notify = (text: string, isErr = false) => setBannerMsg({ text, isErr })

  const show3DWorkspace =
    section === 'overview' ||
    section === 'simulation' ||
    section === 'robot-control' ||
    section === 'task-execution' ||
    section === 'environment'

  const simStatus = snapshot?.status ?? 'DISCONNECTED'
  const isPaused = simStatus === 'PAUSED'
  const isEstop = snapshot?.emergency_stopped ?? false

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-ws-bg text-ws-text">
      {/* TOP SYSTEM-STATUS BAR (40px) */}
      <header className="flex h-10 shrink-0 items-center justify-between border-b border-ws-border bg-ws-panel px-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                isEstop
                  ? 'bg-ws-error'
                  : connected
                    ? 'bg-ws-success'
                    : 'bg-ws-warn'
              }`}
            />
            <span className="font-mono text-xs font-semibold tracking-tight text-ws-text">
              DEXTEROUS-ROBOT-LEARNING
            </span>
          </div>
          <span className="hidden border-l border-ws-border pl-3 font-mono text-[11px] text-ws-muted md:inline">
            STATUS:{' '}
            <span
              className={
                isEstop
                  ? 'text-ws-error font-semibold'
                  : simStatus === 'RUNNING'
                    ? 'text-ws-success'
                    : 'text-ws-warn'
              }
            >
              {simStatus}
            </span>
          </span>
          <span className="hidden font-mono text-[11px] text-ws-muted lg:inline">
            t={snapshot?.sim_time_s.toFixed(1) ?? '0.0'}s | step={snapshot?.step_count ?? 0} | SIM-ONLY
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {isPaused ? (
            <button
              onClick={() => api.simResume().catch((e) => notify(e.message, true))}
              className="flex items-center gap-1 border border-ws-success bg-ws-success/15 px-2.5 py-1 font-mono text-[11px] text-ws-text hover:bg-ws-success/25"
            >
              <Play className="h-3 w-3" /> Resume
            </button>
          ) : (
            <button
              onClick={() => api.simPause().catch((e) => notify(e.message, true))}
              className="flex items-center gap-1 border border-ws-border bg-ws-panel2 px-2.5 py-1 font-mono text-[11px] text-ws-text hover:border-ws-accent"
            >
              <Square className="h-3 w-3" /> Pause
            </button>
          )}
          <button
            onClick={() => api.simReset().then(() => notify('Simulation reset')).catch((e) => notify(e.message, true))}
            className="flex items-center gap-1 border border-ws-border bg-ws-panel2 px-2.5 py-1 font-mono text-[11px] text-ws-text hover:border-ws-accent"
          >
            <RotateCcw className="h-3 w-3" /> Reset
          </button>
          <button
            onClick={() => api.emergencyStop().then(() => notify('EMERGENCY STOP ENGAGED', true))}
            className="flex items-center gap-1 border border-ws-error bg-ws-error/20 px-2.5 py-1 font-mono text-[11px] font-semibold text-ws-error hover:bg-ws-error/30"
          >
            <ShieldAlert className="h-3 w-3" /> E-STOP
          </button>
        </div>
      </header>

      {/* MAIN WORKBENCH AREA */}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* COMPACT LEFT SIDEBAR */}
        <nav className="flex w-44 shrink-0 flex-col border-r border-ws-border bg-ws-panel py-2 md:w-48">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => setSection(item.id)}
              className={`flex items-center gap-2 border-l-2 px-3 py-2 text-left text-xs transition-colors ${
                section === item.id
                  ? 'border-ws-accent bg-ws-panel2 font-medium text-ws-text'
                  : 'border-transparent text-ws-muted hover:bg-ws-panel2/50 hover:text-ws-text'
              }`}
            >
              {item.icon}
              <span className="truncate">{item.label}</span>
            </button>
          ))}
          <div className="mt-auto border-t border-ws-border px-3 pt-2 font-mono text-[10px] text-ws-muted">
            <div>Franka Panda 7-DOF</div>
            <div className={connected ? 'text-ws-success' : 'text-ws-error'}>
              WS: {connected ? 'LIVE 20 Hz' : 'OFFLINE'}
            </div>
          </div>
        </nav>

        {/* CENTER + RIGHT VIEWPORT WORKSPACE */}
        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          {show3DWorkspace ? (
            <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
              {/* CENTRAL 3D VIEWPORT (Primary Focus) */}
              <div className="min-h-[320px] flex-1 p-2">
                <FrankaPandaViewport snapshot={snapshot} options={vpOpts} setOptions={setVpOpts} />
              </div>

              {/* RIGHT CONTEXTUAL CONTROL PANEL */}
              <aside className="h-80 shrink-0 border-t border-ws-border bg-ws-panel lg:h-full lg:w-[340px] lg:border-l lg:border-t-0">
                {section === 'overview' && (
                  <DiagnosticsAndOverviewPanel
                    mode="overview"
                    snapshot={snapshot}
                    events={events}
                    connected={connected}
                    wsError={wsError}
                  />
                )}
                {(section === 'simulation' || section === 'robot-control') && (
                  <RobotControlPanel snapshot={snapshot} onActionLog={notify} />
                )}
                {section === 'task-execution' && (
                  <TaskExecutionPanel snapshot={snapshot} onActionLog={notify} />
                )}
                {section === 'environment' && (
                  <EnvironmentPanel snapshot={snapshot} onActionLog={notify} />
                )}
              </aside>
            </div>
          ) : (
            <main className="min-h-0 flex-1 overflow-hidden bg-ws-bg">
              {section === 'rl' && <PoliciesPanel onActionLog={notify} />}
              {section === 'experiments' && <ExperimentsAndEvalPanel mode="experiments" />}
              {section === 'evaluation' && <ExperimentsAndEvalPanel mode="evaluation" />}
              {section === 'diagnostics' && (
                <DiagnosticsAndOverviewPanel
                  mode="diagnostics"
                  snapshot={snapshot}
                  events={events}
                  connected={connected}
                  wsError={wsError}
                />
              )}
              {section === 'settings' && (
                <DiagnosticsAndOverviewPanel
                  mode="settings"
                  snapshot={snapshot}
                  events={events}
                  connected={connected}
                  wsError={wsError}
                />
              )}
            </main>
          )}

          {/* BOTTOM TELEMETRY & EXECUTION STRIP */}
          <footer className="flex h-24 shrink-0 items-center justify-between gap-4 border-t border-ws-border bg-ws-panel px-3 font-mono text-[11px]">
            <div className="grid flex-1 grid-cols-2 gap-x-6 gap-y-1 md:grid-cols-4">
              <div>
                <div className="text-[10px] text-ws-muted">END-EFFECTOR XYZ (m)</div>
                <div className="text-ws-text">
                  {snapshot ? snapshot.ee_position.map((v) => v.toFixed(3)).join(', ') : '-'}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-ws-muted">GRIPPER APERTURE</div>
                <div className="text-ws-success">
                  {snapshot ? `${(snapshot.gripper.width * 100).toFixed(2)} cm` : '-'}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-ws-muted">TASK STATUS</div>
                <div className="truncate text-ws-accent">
                  {snapshot?.task.task_id ? `${snapshot.task.task_id} (${snapshot.task.status})` : 'IDLE'}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-ws-muted">JOINTS q1..q7 (rad)</div>
                <div className="truncate text-ws-muted">
                  {snapshot ? snapshot.joint_positions.map((j) => j.toFixed(2)).join(' ') : '-'}
                </div>
              </div>
            </div>

            {(bannerMsg || snapshot?.last_error || wsError) && (
              <div
                className={`max-w-xs truncate border px-2.5 py-1 text-[10px] ${
                  bannerMsg?.isErr || snapshot?.last_error || wsError
                    ? 'border-ws-error bg-ws-error/15 text-ws-error'
                    : 'border-ws-border bg-ws-panel2 text-ws-text'
                }`}
              >
                {snapshot?.last_error || wsError || bannerMsg?.text}
              </div>
            )}
          </footer>
        </div>
      </div>
    </div>
  )
}

export default App

