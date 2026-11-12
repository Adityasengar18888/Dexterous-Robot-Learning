const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  const text = await res.text()
  let parsed: unknown = {}
  if (text) {
    try {
      parsed = JSON.parse(text)
    } catch {
      parsed = { detail: text }
    }
  }
  if (!res.ok) {
    const detail = (parsed as { detail?: string }).detail || `HTTP ${res.status}`
    throw new Error(detail)
  }
  return parsed as T
}

export const api = {
  health: () => request<{ status: string; physics_engine: string; physical_robot: string }>('/api/health'),
  systemStatus: () => request<Record<string, unknown>>('/api/system/status'),
  robotInfo: () => request<Record<string, unknown>>('/api/robot/info'),
  simStatus: () => request('/api/simulation/status'),
  simStart: () => request('/api/simulation/start', { method: 'POST' }),
  simPause: () => request('/api/simulation/pause', { method: 'POST' }),
  simResume: () => request('/api/simulation/resume', { method: 'POST' }),
  simReset: () => request('/api/simulation/reset', { method: 'POST' }),
  setJoints: (positions: number[], steps = 60) =>
    request<{ achieved_joints: number[] }>('/api/robot/joints', {
      method: 'POST',
      body: JSON.stringify({ positions, steps }),
    }),
  moveCartesian: (payload: {
    x: number
    y: number
    z: number
    roll?: number
    pitch?: number
    yaw?: number
    steps?: number
  }) =>
    request<Record<string, unknown>>('/api/robot/cartesian-target', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  commandGripper: (action: 'open' | 'close' | 'set_width', width?: number) =>
    request<{ actual_width: number; status: string }>('/api/robot/gripper', {
      method: 'POST',
      body: JSON.stringify({ action, width }),
    }),
  emergencyStop: () => request('/api/robot/emergency-stop', { method: 'POST' }),
  listTasks: () => request<{ tasks: import('../types/robotics').TaskSpec[] }>('/api/tasks'),
  executeTask: (payload: {
    task_id: string
    controller_mode: 'SCRIPTED' | 'POLICY'
    policy_id?: string | null
    seed: number
    max_steps: number
  }) =>
    request('/api/tasks/execute', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  stopTask: () => request('/api/tasks/stop', { method: 'POST' }),
  listPolicies: () =>
    request<{
      active_env: string
      loaded_policy_id: string | null
      policies: import('../types/robotics').PolicySpec[]
    }>('/api/policies'),
  loadPolicy: (policy_id: string) =>
    request<{ policy: import('../types/robotics').PolicySpec }>('/api/policies/load', {
      method: 'POST',
      body: JSON.stringify({ policy_id }),
    }),
  listExperiments: () =>
    request<{
      experiments: import('../types/robotics').ExperimentRecord[]
      phases: Record<string, string>[]
      debugging_history: Record<string, string>[]
    }>('/api/experiments'),
  getEvaluation: (expId: string) =>
    request<{
      metadata: import('../types/robotics').ExperimentRecord
      series: Record<string, unknown>[]
      summary: Record<string, unknown>
      tb_run?: string
    }>(`/api/evaluation/${encodeURIComponent(expId)}`),
  getEnvConfig: () =>
    request<{
      environment: string
      seed: number
      dr_tier: string
      dr_enabled: boolean
      available_environments: string[]
    }>('/api/environment/config'),
  setEnvConfig: (payload: {
    environment: string
    seed: number
    dr_tier: string
    dr_enabled: boolean
  }) =>
    request('/api/environment/config', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
}

