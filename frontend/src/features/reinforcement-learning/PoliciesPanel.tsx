import React, { useEffect, useState } from 'react'
import { api } from '../../services/api'
import type { PolicySpec } from '../../types/robotics'

interface TrainingRunSummary {
  run: string
  steps: number
  actor_loss: number
  ent_coef: number | null
  fps: number | null
}

export const PoliciesPanel: React.FC<{
  onActionLog: (msg: string, isErr?: boolean) => void
}> = ({ onActionLog }) => {
  const [policies, setPolicies] = useState<PolicySpec[]>([])
  const [loadedId, setLoadedId] = useState<string | null>(null)
  const [activeEnv, setActiveEnv] = useState<string>('')
  const [tbRuns, setTbRuns] = useState<TrainingRunSummary[]>([])

  const refresh = () => {
    api.listPolicies().then((r) => {
      setPolicies(r.policies)
      setLoadedId(r.loaded_policy_id)
      setActiveEnv(r.active_env)
    }).catch(() => {})
  }

  useEffect(() => {
    refresh()
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const base = import.meta.env.VITE_WS_BASE_URL || `${proto}//${window.location.host}`
    const ws = new WebSocket(`${base}/ws/training`)
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'training_archive_snapshot' && Array.isArray(msg.runs)) {
          setTbRuns(msg.runs)
        }
      } catch {
        // ignore
      }
    }
    return () => ws.close()
  }, [])

  const provColor = (p: PolicySpec['provenance']) => {
    if (p === 'SCRIPTED') return 'border-ws-accent text-ws-accent'
    if (p === 'BEHAVIOR_CLONING') return 'border-ws-warn text-ws-warn'
    if (p === 'HYBRID_BC_SAC') return 'border-ws-success text-ws-success'
    return 'border-ws-success text-ws-success'
  }

  const handleLoad = async (pid: string) => {
    try {
      await api.loadPolicy(pid)
      refresh()
      onActionLog(`Validated & loaded policy '${pid}'`)
    } catch (e) {
      onActionLog(e instanceof Error ? e.message : String(e), true)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4 text-xs">
      <div className="flex items-center justify-between border-b border-ws-border pb-2">
        <div>
          <h2 className="text-sm font-semibold text-ws-text">Policies & Reinforcement Learning Registry</h2>
          <p className="font-mono text-[11px] text-ws-muted">
            Strict provenance separation (SCRIPTED vs LEARNED) | Active Env: {activeEnv} | Training jobs auto-start: DISABLED
          </p>
        </div>
        <button
          onClick={refresh}
          className="border border-ws-border bg-ws-panel2 px-3 py-1 font-mono text-[11px] text-ws-text hover:border-ws-accent"
        >
          Refresh Checkpoints
        </button>
      </div>

      <div className="grid gap-2.5 lg:grid-cols-2">
        {policies.map((p) => (
          <div
            key={p.policy_id}
            className={`border p-3 ${
              loadedId === p.policy_id ? 'border-ws-success bg-ws-panel2' : 'border-ws-border bg-ws-panel'
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="font-semibold text-ws-text">{p.name}</div>
                <div className="mt-0.5 font-mono text-[10px] text-ws-muted">{p.algorithm}</div>
              </div>
              <span className={`border px-1.5 py-0.5 font-mono text-[10px] ${provColor(p.provenance)}`}>
                {p.provenance}
              </span>
            </div>

            <div className="mt-2 grid grid-cols-2 gap-1 border-t border-ws-border pt-2 font-mono text-[10px]">
              <span className="text-ws-muted">Checkpoint:</span>
              <span className={p.exists ? 'text-ws-success' : 'text-ws-error'}>
                {p.exists ? `${p.file_size_kb} KB` : 'MISSING'} ({p.resolved_path.split('/').pop()})
              </span>
              <span className="text-ws-muted">Dims (Obs/Goal/Act):</span>
              <span className="text-ws-text">
                {p.obs_dim}D / {p.goal_dim}D / {p.action_dim}D
              </span>
              <span className="text-ws-muted">Benchmark Metric:</span>
              <span className="text-ws-text">{p.verified_benchmark}</span>
            </div>

            <div className="mt-2.5 flex items-center justify-between">
              <span
                className={`font-mono text-[10px] ${
                  p.compatible_with_active_env ? 'text-ws-success' : 'text-ws-warn'
                }`}
              >
                {p.compatible_with_active_env
                  ? 'Compatible with active env'
                  : `Targets: ${p.compatible_envs.join(', ')}`}
              </span>
              <button
                disabled={!p.exists && p.provenance !== 'SCRIPTED'}
                onClick={() => handleLoad(p.policy_id)}
                className="border border-ws-accent bg-ws-accent/15 px-3 py-1 font-mono text-[11px] text-ws-text hover:bg-ws-accent/30 disabled:opacity-40"
              >
                {loadedId === p.policy_id ? 'Loaded' : 'Validate & Load'}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Real Archived TensorBoard Runs Table */}
      <div className="border border-ws-border bg-ws-panel p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="font-semibold text-ws-text">Archived TensorBoard Training Ledger (`/ws/training`)</span>
          <span className="font-mono text-[10px] text-ws-muted">{tbRuns.length} Real Runs Streamed</span>
        </div>
        <div className="max-h-64 overflow-auto border border-ws-border">
          <table className="w-full text-left font-mono text-[10px]">
            <thead className="sticky top-0 bg-ws-panel2 text-ws-muted">
              <tr>
                <th className="p-2">Log Run</th>
                <th className="p-2">Final Step</th>
                <th className="p-2">Actor Loss</th>
                <th className="p-2">Ent Coef</th>
                <th className="p-2">CPU FPS</th>
              </tr>
            </thead>
            <tbody>
              {tbRuns.map((r) => (
                <tr key={r.run} className="border-t border-ws-border hover:bg-ws-panel2/60">
                  <td className="p-1.5 text-ws-text">{r.run}</td>
                  <td className="p-1.5 text-ws-muted">{r.steps.toLocaleString()}</td>
                  <td className="p-1.5 text-ws-accent">{r.actor_loss}</td>
                  <td className="p-1.5 text-ws-muted">{r.ent_coef ?? '-'}</td>
                  <td className="p-1.5 text-ws-success">{r.fps ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

