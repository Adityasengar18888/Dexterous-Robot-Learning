import React, { useEffect, useState } from 'react'
import { api } from '../../services/api'
import type { SimulationSnapshot } from '../../types/robotics'

export const EnvironmentPanel: React.FC<{
  snapshot: SimulationSnapshot | null
  onActionLog: (msg: string, isErr?: boolean) => void
}> = ({ snapshot, onActionLog }) => {
  const [envs, setEnvs] = useState<string[]>([])
  const [envName, setEnvName] = useState(snapshot?.active_env ?? 'PandaPickPlaceEnv')
  const [seed, setSeed] = useState(snapshot?.seed ?? 42)
  const [drTier, setDrTier] = useState(snapshot?.dr_tier ?? 'DR0')
  const [drEnabled, setDrEnabled] = useState(snapshot?.dr_enabled ?? false)

  useEffect(() => {
    api.getEnvConfig().then((cfg) => {
      setEnvs(cfg.available_environments)
      setEnvName(cfg.environment)
      setSeed(cfg.seed)
      setDrTier(cfg.dr_tier)
      setDrEnabled(cfg.dr_enabled)
    }).catch(() => {})
  }, [])

  const applyConfig = async () => {
    try {
      await api.setEnvConfig({ environment: envName, seed, dr_tier: drTier, dr_enabled: drEnabled })
      onActionLog(`Environment reset -> ${envName} (seed=${seed}, ${drEnabled ? drTier : 'Nominal'})`)
    } catch (e) {
      onActionLog(e instanceof Error ? e.message : String(e), true)
    }
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3 text-xs">
      <div className="border-b border-ws-border pb-1.5 font-semibold text-ws-text">
        Environment & Domain Randomization
      </div>

      <label className="flex flex-col gap-1">
        <span className="font-mono text-[10px] text-ws-muted">GYMNASIUM ENVIRONMENT</span>
        <select
          value={envName}
          onChange={(e) => setEnvName(e.target.value)}
          className="border border-ws-border bg-ws-panel2 px-2 py-1.5 font-mono text-[11px] text-ws-text"
        >
          {envs.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>

      <div className="grid grid-cols-2 gap-2">
        <label className="border border-ws-border bg-ws-panel2 p-2">
          <div className="font-mono text-[10px] text-ws-muted">RESET SEED</div>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value, 10) || 42)}
            className="mt-1 w-full border border-ws-border bg-ws-bg px-1.5 py-1 font-mono text-[11px] text-ws-text"
          />
        </label>
        <label className="border border-ws-border bg-ws-panel2 p-2">
          <div className="font-mono text-[10px] text-ws-muted">DR TIER</div>
          <select
            value={drTier}
            disabled={!drEnabled}
            onChange={(e) => setDrTier(e.target.value)}
            className="mt-1 w-full border border-ws-border bg-ws-bg px-1 py-1 font-mono text-[11px] text-ws-text disabled:opacity-40"
          >
            {['DR0', 'DR1', 'DR2', 'DR3', 'DR4'].map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
      </div>

      <label className="flex items-center gap-2 border border-ws-border bg-ws-panel2 p-2 font-mono text-[11px] text-ws-text">
        <input
          type="checkbox"
          checked={drEnabled}
          onChange={(e) => setDrEnabled(e.target.checked)}
        />
        Enable Physics / Spatial Domain Randomization
      </label>

      <button
        onClick={applyConfig}
        className="border border-ws-accent bg-ws-accent/20 py-1.5 font-mono text-[11px] font-medium text-ws-text hover:bg-ws-accent/30"
      >
        Apply Config & Reload World
      </button>

      <div className="border border-ws-border bg-ws-panel2 p-2.5 font-mono text-[10px] text-ws-muted">
        <div className="mb-1 font-semibold text-ws-text">Workspace Safety Limits (m)</div>
        <div>X: [{snapshot?.workspace_bounds.x.join(', ')}]</div>
        <div>Y: [{snapshot?.workspace_bounds.y.join(', ')}]</div>
        <div>Z: [{snapshot?.workspace_bounds.z.join(', ')}]</div>
      </div>
    </div>
  )
}

