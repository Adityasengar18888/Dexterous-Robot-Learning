import React, { useEffect, useState } from 'react'
import { api } from '../../services/api'
import type { PolicySpec, SimulationSnapshot, TaskSpec } from '../../types/robotics'

export const TaskExecutionPanel: React.FC<{
  snapshot: SimulationSnapshot | null
  onActionLog: (msg: string, isErr?: boolean) => void
}> = ({ snapshot, onActionLog }) => {
  const [tasks, setTasks] = useState<TaskSpec[]>([])
  const [policies, setPolicies] = useState<PolicySpec[]>([])
  const [selectedTask, setSelectedTask] = useState('pick_place')
  const [mode, setMode] = useState<'SCRIPTED' | 'POLICY'>('SCRIPTED')
  const [policyId, setPolicyId] = useState<string>('')
  const [seed, setSeed] = useState(42)
  const [maxSteps, setMaxSteps] = useState(150)

  useEffect(() => {
    api.listTasks().then((r) => setTasks(r.tasks)).catch(() => {})
    api.listPolicies().then((r) => {
      setPolicies(r.policies)
      const firstLearned = r.policies.find((p) => p.provenance !== 'SCRIPTED' && p.exists)
      if (firstLearned) setPolicyId(firstLearned.policy_id)
    }).catch(() => {})
  }, [])

  const activeSpec = tasks.find((t) => t.task_id === selectedTask)
  const implemented = activeSpec?.status === 'IMPLEMENTED'

  const handleExecute = async () => {
    try {
      await api.executeTask({
        task_id: selectedTask,
        controller_mode: mode,
        policy_id: mode === 'POLICY' ? policyId : null,
        seed,
        max_steps: maxSteps,
      })
      onActionLog(`Task '${selectedTask}' launched (${mode}, seed=${seed})`)
    } catch (e) {
      onActionLog(e instanceof Error ? e.message : String(e), true)
    }
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3 text-xs">
      <div className="border-b border-ws-border pb-1.5 font-semibold text-ws-text">
        Manipulation Task Runner
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="font-mono text-[10px] text-ws-muted">SELECT TASK</label>
        <select
          value={selectedTask}
          onChange={(e) => {
            setSelectedTask(e.target.value)
            const sp = tasks.find((t) => t.task_id === e.target.value)
            if (sp && !sp.controllers.includes(mode)) setMode('SCRIPTED')
          }}
          className="border border-ws-border bg-ws-panel2 px-2 py-1.5 font-mono text-[11px] text-ws-text"
        >
          {tasks.map((t) => (
            <option key={t.task_id} value={t.task_id}>
              {t.name} [{t.status}]
            </option>
          ))}
        </select>

        {activeSpec && (
          <div className="border border-ws-border bg-ws-panel2 p-2 text-[11px]">
            <div className="flex items-center justify-between font-mono text-[10px]">
              <span className="text-ws-muted">Env: {activeSpec.env}</span>
              <span className={implemented ? 'text-ws-success' : 'text-ws-error font-semibold'}>
                {activeSpec.status}
              </span>
            </div>
            <p className="mt-1 text-ws-muted">{activeSpec.description}</p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <label className="border border-ws-border bg-ws-panel2 p-2">
          <div className="font-mono text-[10px] text-ws-muted">CONTROLLER</div>
          <select
            value={mode}
            disabled={!implemented}
            onChange={(e) => setMode(e.target.value as 'SCRIPTED' | 'POLICY')}
            className="mt-1 w-full border border-ws-border bg-ws-bg px-1 py-1 font-mono text-[11px] text-ws-text"
          >
            <option value="SCRIPTED">SCRIPTED</option>
            {activeSpec?.controllers.includes('POLICY') && <option value="POLICY">POLICY (Checkpoint)</option>}
          </select>
        </label>
        <label className="border border-ws-border bg-ws-panel2 p-2">
          <div className="font-mono text-[10px] text-ws-muted">SEED</div>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value, 10) || 42)}
            className="mt-1 w-full border border-ws-border bg-ws-bg px-1.5 py-1 font-mono text-[11px] text-ws-text"
          />
        </label>
      </div>

      {mode === 'POLICY' && (
        <label className="border border-ws-border bg-ws-panel2 p-2">
          <div className="font-mono text-[10px] text-ws-muted">LOADED / TARGET CHECKPOINT</div>
          <select
            value={policyId}
            onChange={(e) => setPolicyId(e.target.value)}
            className="mt-1 w-full border border-ws-border bg-ws-bg px-1.5 py-1 font-mono text-[11px] text-ws-text"
          >
            {policies
              .filter((p) => p.provenance !== 'SCRIPTED')
              .map((p) => (
                <option key={p.policy_id} value={p.policy_id}>
                  [{p.provenance}] {p.policy_id}
                </option>
              ))}
          </select>
        </label>
      )}

      <label className="border border-ws-border bg-ws-panel2 p-2">
        <div className="flex justify-between font-mono text-[10px] text-ws-muted">
          <span>MAX STEPS</span>
          <span>{maxSteps}</span>
        </div>
        <input
          type="range"
          min={30}
          max={300}
          step={10}
          value={maxSteps}
          onChange={(e) => setMaxSteps(parseInt(e.target.value, 10))}
          className="mt-1 w-full"
        />
      </label>

      <div className="grid grid-cols-3 gap-1.5">
        <button
          disabled={!implemented}
          onClick={handleExecute}
          className="border border-ws-success bg-ws-success/20 py-1.5 font-mono text-[11px] font-medium text-ws-text hover:bg-ws-success/30 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Execute
        </button>
        <button
          onClick={() => api.stopTask().then(() => onActionLog('Stopped task runner'))}
          className="border border-ws-warn bg-ws-warn/15 py-1.5 font-mono text-[11px] text-ws-text hover:bg-ws-warn/25"
        >
          Stop Task
        </button>
        <button
          onClick={() => api.simReset().then(() => onActionLog('Reset simulation & task'))}
          className="border border-ws-border bg-ws-panel2 py-1.5 font-mono text-[11px] text-ws-text hover:border-ws-accent"
        >
          Reset
        </button>
      </div>

      {/* Live Execution Telemetry */}
      <div className="border border-ws-border bg-ws-panel2 p-2.5 font-mono text-[10px]">
        <div className="mb-1 font-semibold text-ws-text">Live Task Execution Telemetry</div>
        <div className="grid grid-cols-2 gap-1 text-ws-muted">
          <span>Active Task:</span>
          <span className="text-right text-ws-text">{snapshot?.task.task_id ?? 'NONE'}</span>
          <span>Status:</span>
          <span
            className={`text-right ${
              snapshot?.task.status === 'COMPLETED_SUCCESS'
                ? 'text-ws-success'
                : snapshot?.task.status.includes('FAIL') || snapshot?.task.status === 'EMERGENCY_STOPPED'
                  ? 'text-ws-error'
                  : 'text-ws-accent'
            }`}
          >
            {snapshot?.task.status ?? 'IDLE'}
          </span>
          <span>Progress Step:</span>
          <span className="text-right text-ws-text">
            {snapshot?.task.step ?? 0} / {snapshot?.task.max_steps ?? 150}
          </span>
          <span>Controller Mode:</span>
          <span className="text-right text-ws-text">{snapshot?.task.mode ?? 'SCRIPTED'}</span>
          <span>is_success / grasp:</span>
          <span className="text-right text-ws-text">
            {String(snapshot?.metrics?.is_success ?? false)} / {String(snapshot?.metrics?.grasp_success ?? false)}
          </span>
        </div>
      </div>
    </div>
  )
}

