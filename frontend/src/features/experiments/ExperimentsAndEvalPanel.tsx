import React, { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../../services/api'
import type { ExperimentRecord } from '../../types/robotics'

export const ExperimentsAndEvalPanel: React.FC<{
  mode: 'experiments' | 'evaluation'
}> = ({ mode }) => {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([])
  const [search, setSearch] = useState('')
  const [provFilter, setProvFilter] = useState('ALL')
  const [selectedExp, setSelectedExp] = useState<string>('EXP-P12-1M')
  const [evalData, setEvalData] = useState<{
    metadata: ExperimentRecord
    series: Record<string, unknown>[]
    summary: Record<string, unknown>
    tb_run?: string
  } | null>(null)

  useEffect(() => {
    api.listExperiments().then((r) => setExperiments(r.experiments)).catch(() => {})
  }, [])

  useEffect(() => {
    if (!selectedExp) return
    api.getEvaluation(selectedExp).then(setEvalData).catch(() => setEvalData(null))
  }, [selectedExp])

  const filtered = useMemo(() => {
    return experiments.filter((e) => {
      const okProv = provFilter === 'ALL' || e.provenance === provFilter
      const q = search.trim().toLowerCase()
      const okQ =
        !q ||
        e.exp_id.toLowerCase().includes(q) ||
        e.phase.toLowerCase().includes(q) ||
        e.environment.toLowerCase().includes(q) ||
        e.method.toLowerCase().includes(q) ||
        e.verified_outcome.toLowerCase().includes(q)
      return okProv && okQ
    })
  }, [experiments, provFilter, search])

  const exportCsv = () => {
    const hdr = ['exp_id', 'phase', 'environment', 'method', 'provenance', 'hardware', 'budget_or_eval', 'seeds', 'verified_outcome']
    const lines = [hdr.join(',')]
    for (const r of filtered) {
      lines.push(hdr.map((k) => `"${String((r as unknown as Record<string, string>)[k] ?? '').replace(/"/g, '""')}"`).join(','))
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'dexterous_experiments_export.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  interface EvalChartRow {
    x: string
    reach?: number
    grasp?: number
    success?: number
    actor_loss?: number
  }

  // Convert real Phase 12 or Phase 14 or Phase 15.1 summaries into chartable rows without synthesizing data
  const chartRows = useMemo<EvalChartRow[]>(() => {
    if (!evalData) return []
    if (evalData.series && evalData.series.length > 0) {
      return evalData.series.map((s) => ({
        x: String(s['Total Steps'] ?? s['step'] ?? s['evaluation_condition'] ?? ''),
        reach: s['Reach'] !== undefined ? Number(s['Reach']) * 100 : undefined,
        grasp: s['Grasp'] !== undefined ? Number(s['Grasp']) * 100 : undefined,
        success: s['End-to-End Success'] !== undefined ? Number(s['End-to-End Success']) * 100 : undefined,
        actor_loss: s['actor_loss'] !== undefined ? Number(s['actor_loss']) : undefined,
      }))
    }
    const sumList = (evalData.summary as { summaries?: Record<string, unknown>[] })?.summaries
    if (Array.isArray(sumList)) {
      return sumList.slice(0, 14).map((s) => ({
        x: `${String(s.controller ?? s.condition ?? '').slice(0, 12)} ${s.regime ?? ''} s${s.training_seed ?? ''}`.trim(),
        success: Number(s.success_rate ?? 0) * 100,
        grasp: Number(s.grasp_rate ?? 0) * 100,
      }))
    }
    return []
  }, [evalData])

  if (mode === 'evaluation') {
    return (
      <div className="flex h-full flex-col gap-3 overflow-y-auto p-4 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ws-border pb-2">
          <div>
            <h2 className="text-sm font-semibold text-ws-text">Experiment Evaluation & Genuine Benchmarks</h2>
            <p className="font-mono text-[10px] text-ws-muted">
              Strict separation of SCRIPTED vs LEARNED checkpoints | Backed by archived CSV/JSON/TB logs
            </p>
          </div>
          <select
            value={selectedExp}
            onChange={(e) => setSelectedExp(e.target.value)}
            className="border border-ws-border bg-ws-panel2 px-2.5 py-1.5 font-mono text-[11px] text-ws-text"
          >
            {experiments.map((e) => (
              <option key={e.exp_id} value={e.exp_id}>
                {e.exp_id} — {e.phase} [{e.provenance}]
              </option>
            ))}
          </select>
        </div>

        {evalData && (
          <div className="grid grid-cols-2 gap-2 border border-ws-border bg-ws-panel p-3 font-mono text-[11px] lg:grid-cols-4">
            <div>
              <span className="text-ws-muted">ID: </span>
              <span className="text-ws-accent">{evalData.metadata.exp_id}</span>
            </div>
            <div>
              <span className="text-ws-muted">Provenance: </span>
              <span className="text-ws-text">{evalData.metadata.provenance}</span>
            </div>
            <div>
              <span className="text-ws-muted">Env: </span>
              <span className="text-ws-text">{evalData.metadata.environment}</span>
            </div>
            <div>
              <span className="text-ws-muted">Budget/Eval: </span>
              <span className="text-ws-success">{evalData.metadata.budget_or_eval}</span>
            </div>
            <div className="col-span-2 lg:col-span-4 text-ws-text">
              Outcome: {evalData.metadata.verified_outcome}
            </div>
          </div>
        )}

        <div className="h-80 border border-ws-border bg-ws-panel p-3">
          {chartRows.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              {chartRows[0].actor_loss !== undefined ? (
                <LineChart data={chartRows}>
                  <CartesianGrid stroke="#22272C" strokeDasharray="3 3" />
                  <XAxis dataKey="x" stroke="#929AA3" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#929AA3" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1B1F23', borderColor: '#343B42', fontSize: 11 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="actor_loss" name="SAC Actor Loss" stroke="#789DC4" dot={false} strokeWidth={2} />
                </LineChart>
              ) : (
                <BarChart data={chartRows}>
                  <CartesianGrid stroke="#22272C" strokeDasharray="3 3" />
                  <XAxis dataKey="x" stroke="#929AA3" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#929AA3" domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1B1F23', borderColor: '#343B42', fontSize: 11 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {chartRows[0].reach !== undefined && <Bar dataKey="reach" name="Reach %" fill="#789DC4" />}
                  <Bar dataKey="grasp" name="Grasp %" fill="#D6A65D" />
                  <Bar dataKey="success" name="Success %" fill="#78B89A" />
                </BarChart>
              )}
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center justify-center font-mono text-ws-muted">
              Select EXP-P12-1M, EXP-P14-4, EXP-P15-1, or a TB-logged run to plot raw empirical series.
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-4 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ws-border pb-2">
        <div>
          <h2 className="text-sm font-semibold text-ws-text">Historical & Active Experiment Registry ({filtered.length})</h2>
          <p className="font-mono text-[10px] text-ws-muted">
            Source: publication/experiment_inventory.csv | Zero fabricated metrics
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            placeholder="Search ID, phase, env, method..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="border border-ws-border bg-ws-panel2 px-2.5 py-1 font-mono text-[11px] text-ws-text"
          />
          <select
            value={provFilter}
            onChange={(e) => setProvFilter(e.target.value)}
            className="border border-ws-border bg-ws-panel2 px-2 py-1 font-mono text-[11px] text-ws-text"
          >
            <option value="ALL">ALL PROVENANCE</option>
            <option value="LEARNED_SAC_HER">LEARNED_SAC_HER</option>
            <option value="BEHAVIOR_CLONING">BEHAVIOR_CLONING</option>
            <option value="HYBRID_BC_SAC">HYBRID_BC_SAC</option>
            <option value="SCRIPTED">SCRIPTED</option>
          </select>
          <button
            onClick={exportCsv}
            className="border border-ws-accent bg-ws-accent/15 px-3 py-1 font-mono text-[11px] text-ws-text hover:bg-ws-accent/30"
          >
            Export CSV
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto border border-ws-border bg-ws-panel">
        <table className="w-full text-left font-mono text-[11px]">
          <thead className="sticky top-0 bg-ws-panel2 text-ws-muted">
            <tr>
              <th className="p-2">Exp ID</th>
              <th className="p-2">Phase</th>
              <th className="p-2">Environment</th>
              <th className="p-2">Provenance</th>
              <th className="p-2">Method</th>
              <th className="p-2">Budget / Eval</th>
              <th className="p-2">Verified Outcome</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr
                key={r.exp_id}
                onClick={() => setSelectedExp(r.exp_id)}
                className={`cursor-pointer border-t border-ws-border hover:bg-ws-panel2 ${
                  selectedExp === r.exp_id ? 'bg-ws-accent/10' : ''
                }`}
              >
                <td className="p-2 font-semibold text-ws-accent">{r.exp_id}</td>
                <td className="p-2 text-ws-text">{r.phase}</td>
                <td className="p-2 text-ws-text">{r.environment}</td>
                <td className="p-2 text-ws-warn">{r.provenance}</td>
                <td className="p-2 text-ws-muted">{r.method}</td>
                <td className="p-2 text-ws-text">{r.budget_or_eval}</td>
                <td className="p-2 text-ws-success">{r.verified_outcome}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
