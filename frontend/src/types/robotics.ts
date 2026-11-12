export type NavSection =
  | 'overview'
  | 'simulation'
  | 'robot-control'
  | 'task-execution'
  | 'rl'
  | 'environment'
  | 'experiments'
  | 'evaluation'
  | 'diagnostics'
  | 'settings'

export interface LinkTransform {
  name: string
  link_idx: number
  visual_mesh: string
  collision_mesh: string
  mesh_rpy: [number, number, number]
  position: [number, number, number]
  quaternion: [number, number, number, number]
}

export interface SceneObject {
  id: string
  kind: 'cube' | 'peg'
  size: number
  color: string
  position: [number, number, number]
  quaternion: [number, number, number, number]
}

export interface SceneTarget {
  id: string
  kind: 'target' | 'slot'
  position: [number, number, number]
  color: string
}

export interface SceneBin {
  id: string
  position: [number, number, number]
  color: string
}

export interface IKResult {
  requested_pose?: { x: number; y: number; z: number; quat: number[] }
  achieved_pose?: { x: number; y: number; z: number; quat: number[] }
  position_error_m?: number
  orientation_error_rad?: number
  ik_converged?: boolean
}

export interface SimulationSnapshot {
  timestamp: number
  status: 'IDLE' | 'RUNNING' | 'PAUSED' | 'STOPPED' | 'EMERGENCY_STOPPED'
  step_count: number
  sim_time_s: number
  active_env: string
  seed: number
  dr_tier: string
  dr_enabled: boolean
  emergency_stopped: boolean
  joint_names: string[]
  joint_positions: number[]
  joint_velocities: number[]
  joint_limits: { lower: number[]; upper: number[] }
  workspace_bounds: { x: [number, number]; y: [number, number]; z: [number, number] }
  ee_position: [number, number, number]
  ee_orientation: [number, number, number, number]
  ee_rpy: [number, number, number]
  gripper: { width: number; is_open: boolean }
  link_transforms: LinkTransform[]
  objects: SceneObject[]
  targets: SceneTarget[]
  bins: SceneBin[]
  ee_trajectory: [number, number, number][]
  task: {
    task_id: string | null
    status: string
    step: number
    max_steps: number
    mode: string
  }
  policy_id: string | null
  metrics: Record<string, unknown>
  last_ik_result: IKResult
  last_error: string | null
}

export interface TaskSpec {
  task_id: string
  name: string
  env: string
  status: 'IMPLEMENTED' | 'NOT IMPLEMENTED'
  controllers: string[]
  description: string
}

export interface PolicySpec {
  policy_id: string
  name: string
  provenance: 'SCRIPTED' | 'BEHAVIOR_CLONING' | 'LEARNED_SAC_HER' | 'HYBRID_BC_SAC'
  algorithm: string
  compatible_envs: string[]
  obs_dim: number
  goal_dim: number
  action_dim: number
  checkpoint_path: string
  resolved_path: string
  exists: boolean
  file_size_kb: number
  compatible_with_active_env: boolean
  verified_benchmark: string
}

export interface ExperimentRecord {
  exp_id: string
  phase: string
  environment: string
  method: string
  provenance: string
  hardware: string
  budget_or_eval: string
  seeds: string
  verified_outcome: string
}

export interface DiagnosticEvent {
  timestamp: number
  level: string
  source: string
  message: string
}

export interface ViewportOptions {
  showGrid: boolean
  showFrames: boolean
  showCollision: boolean
  showTrajectory: boolean
  showLabels: boolean
  showEeFrame: boolean
}

