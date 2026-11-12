import React, { useEffect, useState } from 'react'
import { api } from '../../services/api'
import type { SimulationSnapshot } from '../../types/robotics'

const HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

export const RobotControlPanel: React.FC<{
  snapshot: SimulationSnapshot | null
  onActionLog: (msg: string, isErr?: boolean) => void
}> = ({ snapshot, onActionLog }) => {
  const [tab, setTab] = useState<'joints' | 'cartesian' | 'gripper'>('joints')
  const [targets, setTargets] = useState<number[]>(HOME_JOINTS)
  const [cart, setCart] = useState({ x: 0.45, y: 0.0, z: 0.25, roll: 3.1416, pitch: 0.0, yaw: 0.0, useRpy: false })
  const [gripWidth, setGripWidth] = useState(0.04)
  const [busy, setBusy] = useState(false)
  const [gripStatus, setGripStatus] = useState<string>('IDLE')

  useEffect(() => {
    if (snapshot?.joint_positions && snapshot.joint_positions.length === 7 && tab !== 'joints') {
      setTargets(snapshot.joint_positions)
    }
  }, [snapshot?.active_env, tab])

  const lower = snapshot?.joint_limits.lower ?? [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
  const upper = snapshot?.joint_limits.upper ?? [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
  const currentJoints = snapshot?.joint_positions ?? HOME_JOINTS
  const names = snapshot?.joint_names ?? ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

  const runCmd = async (fn: () => Promise<void>) => {
    setBusy(true)
    try {
      await fn()
    } catch (e) {
      onActionLog(e instanceof Error ? e.message : String(e), true)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3 text-xs">
      {/* Sub-tabs */}
      <div className="grid grid-cols-3 border border-ws-border bg-ws-panel2">
        {(['joints', 'cartesian', 'gripper'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`py-1.5 font-mono text-[11px] uppercase transition-colors ${
              tab === t ? 'bg-ws-accent/20 text-ws-text font-semibold' : 'text-ws-muted hover:text-ws-text'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* JOINT CONTROL */}
      {tab === 'joints' && (
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center justify-between border-b border-ws-border pb-1.5">
            <span className="font-semibold text-ws-text">7-DOF Joint Control (rad)</span>
            <span className="font-mono text-[10px] text-ws-muted">Limit Checked</span>
          </div>
          {names.map((jn, i) => (
            <div key={jn} className="border border-ws-border bg-ws-panel2 p-2">
              <div className="mb-1 flex items-center justify-between font-mono text-[11px]">
                <span className="text-ws-text">{jn}</span>
                <span className="text-ws-muted">
                  cur <span className="text-ws-success">{currentJoints[i]?.toFixed(3)}</span> | tgt{' '}
                  <span className="text-ws-accent">{targets[i]?.toFixed(3)}</span>
                </span>
              </div>
              <input
                type="range"
                min={lower[i]}
                max={upper[i]}
                step={0.01}
                value={targets[i]}
                onChange={(e) => {
                  const next = [...targets]
                  next[i] = parseFloat(e.target.value)
                  setTargets(next)
                }}
                className="w-full"
              />
              <div className="mt-0.5 flex justify-between font-mono text-[9px] text-ws-muted">
                <span>min {lower[i].toFixed(3)}</span>
                <span>max {upper[i].toFixed(3)}</span>
              </div>
            </div>
          ))}
          <div className="grid grid-cols-3 gap-1.5 pt-1">
            <button
              disabled={busy}
              onClick={() =>
                runCmd(async () => {
                  await api.setJoints(targets)
                  onActionLog('Applied 7-DOF joint target')
                })
              }
              className="border border-ws-accent bg-ws-accent/20 py-1.5 font-mono text-[11px] font-medium text-ws-text hover:bg-ws-accent/30 disabled:opacity-50"
            >
              Apply Target
            </button>
            <button
              disabled={busy}
              onClick={() => setTargets([...currentJoints])}
              className="border border-ws-border bg-ws-panel2 py-1.5 font-mono text-[11px] text-ws-text hover:border-ws-accent"
            >
              Reset Target
            </button>
            <button
              disabled={busy}
              onClick={() =>
                runCmd(async () => {
                  setTargets(HOME_JOINTS)
                  await api.setJoints(HOME_JOINTS)
                  onActionLog('Moved Panda to neutral home configuration')
                })
              }
              className="border border-ws-border bg-ws-panel2 py-1.5 font-mono text-[11px] text-ws-text hover:border-ws-accent"
            >
              Home Pose
            </button>
          </div>
        </div>
      )}

      {/* CARTESIAN IK CONTROL */}
      {tab === 'cartesian' && (
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center justify-between border-b border-ws-border pb-1.5">
            <span className="font-semibold text-ws-text">Cartesian Pose & DLS IK</span>
            <span className="font-mono text-[10px] text-ws-muted">PyBullet IK</span>
          </div>

          <div className="grid grid-cols-3 gap-2">
            {(
              [
                ['x', 0.2, 0.8],
                ['y', -0.45, 0.45],
                ['z', 0.01, 0.65],
              ] as const
            ).map(([axis, minV, maxV]) => (
              <label key={axis} className="border border-ws-border bg-ws-panel2 p-2">
                <div className="font-mono text-[10px] uppercase text-ws-muted">{axis} (m)</div>
                <input
                  type="number"
                  step={0.01}
                  min={minV}
                  max={maxV}
                  value={cart[axis]}
                  onChange={(e) => setCart((c) => ({ ...c, [axis]: parseFloat(e.target.value) || 0 }))}
                  className="mt-1 w-full border border-ws-border bg-ws-bg px-1.5 py-1 font-mono text-[11px] text-ws-text"
                />
              </label>
            ))}
          </div>

          <label className="flex items-center gap-2 font-mono text-[11px] text-ws-muted">
            <input
              type="checkbox"
              checked={cart.useRpy}
              onChange={(e) => setCart((c) => ({ ...c, useRpy: e.target.checked }))}
            />
            Override EE Orientation (Roll / Pitch / Yaw rad)
          </label>

          {cart.useRpy && (
            <div className="grid grid-cols-3 gap-2">
              {(['roll', 'pitch', 'yaw'] as const).map((ang) => (
                <label key={ang} className="border border-ws-border bg-ws-panel2 p-2">
                  <div className="font-mono text-[10px] uppercase text-ws-muted">{ang}</div>
                  <input
                    type="number"
                    step={0.05}
                    min={-3.1416}
                    max={3.1416}
                    value={cart[ang]}
                    onChange={(e) => setCart((c) => ({ ...c, [ang]: parseFloat(e.target.value) || 0 }))}
                    className="mt-1 w-full border border-ws-border bg-ws-bg px-1.5 py-1 font-mono text-[11px] text-ws-text"
                  />
                </label>
              ))}
            </div>
          )}

          <button
            disabled={busy}
            onClick={() =>
              runCmd(async () => {
                await api.moveCartesian({
                  x: cart.x,
                  y: cart.y,
                  z: cart.z,
                  ...(cart.useRpy ? { roll: cart.roll, pitch: cart.pitch, yaw: cart.yaw } : {}),
                })
                onActionLog(`Cartesian IK command (${cart.x.toFixed(2)}, ${cart.y.toFixed(2)}, ${cart.z.toFixed(2)})`)
              })
            }
            className="border border-ws-accent bg-ws-accent/20 py-1.5 font-mono text-[11px] font-medium text-ws-text hover:bg-ws-accent/30 disabled:opacity-50"
          >
            Solve IK & Move End-Effector
          </button>

          {/* Achieved vs Requested Pose Telemetry */}
          <div className="border border-ws-border bg-ws-panel2 p-2.5 font-mono text-[10px]">
            <div className="mb-1 font-semibold text-ws-text">IK Verification Telemetry</div>
            <div className="grid grid-cols-2 gap-1 text-ws-muted">
              <span>Live EE Pos (m):</span>
              <span className="text-right text-ws-text">
                {snapshot?.ee_position.map((v) => v.toFixed(4)).join(', ') ?? '-'}
              </span>
              <span>Requested (m):</span>
              <span className="text-right text-ws-accent">
                {snapshot?.last_ik_result?.requested_pose
                  ? `${snapshot.last_ik_result.requested_pose.x.toFixed(4)}, ${snapshot.last_ik_result.requested_pose.y.toFixed(4)}, ${snapshot.last_ik_result.requested_pose.z.toFixed(4)}`
                  : '-'}
              </span>
              <span>Achieved (m):</span>
              <span className="text-right text-ws-success">
                {snapshot?.last_ik_result?.achieved_pose
                  ? `${snapshot.last_ik_result.achieved_pose.x.toFixed(4)}, ${snapshot.last_ik_result.achieved_pose.y.toFixed(4)}, ${snapshot.last_ik_result.achieved_pose.z.toFixed(4)}`
                  : '-'}
              </span>
              <span>Position Error:</span>
              <span className="text-right text-ws-text">
                {snapshot?.last_ik_result?.position_error_m !== undefined
                  ? `${(snapshot.last_ik_result.position_error_m * 1000).toFixed(2)} mm`
                  : '-'}
              </span>
              <span>Orientation Error:</span>
              <span className="text-right text-ws-text">
                {snapshot?.last_ik_result?.orientation_error_rad !== undefined
                  ? `${snapshot.last_ik_result.orientation_error_rad.toFixed(4)} rad`
                  : '-'}
              </span>
              <span>IK Status:</span>
              <span
                className={`text-right ${
                  snapshot?.last_ik_result?.ik_converged === undefined
                    ? 'text-ws-muted'
                    : snapshot.last_ik_result.ik_converged
                      ? 'text-ws-success'
                      : 'text-ws-warn'
                }`}
              >
                {snapshot?.last_ik_result?.ik_converged === undefined
                  ? 'NO_CMD'
                  : snapshot.last_ik_result.ik_converged
                    ? 'CONVERGED'
                    : 'CLAMPED / TOL'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* GRIPPER CONTROL */}
      {tab === 'gripper' && (
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center justify-between border-b border-ws-border pb-1.5">
            <span className="font-semibold text-ws-text">Parallel-Jaw Gripper (0.00–0.08 m)</span>
            <span className="font-mono text-[10px] text-ws-muted">{gripStatus}</span>
          </div>

          <div className="border border-ws-border bg-ws-panel2 p-2.5 font-mono text-[11px]">
            <div className="flex justify-between">
              <span className="text-ws-muted">Actual Width:</span>
              <span className="text-ws-success">
                {snapshot ? `${(snapshot.gripper.width * 100).toFixed(2)} cm (${snapshot.gripper.width.toFixed(4)} m)` : '-'}
              </span>
            </div>
            <div className="mt-1 flex justify-between">
              <span className="text-ws-muted">Aperture State:</span>
              <span className="text-ws-text">{snapshot?.gripper.is_open ? 'OPEN' : 'CLOSED / GRIPPING'}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              disabled={busy}
              onClick={() =>
                runCmd(async () => {
                  const r = await api.commandGripper('open')
                  setGripStatus(r.status)
                  onActionLog(`Gripper OPEN -> ${(r.actual_width * 100).toFixed(2)} cm`)
                })
              }
              className="border border-ws-border bg-ws-panel2 py-1.5 font-mono text-[11px] text-ws-text hover:border-ws-accent"
            >
              Open (8.0 cm)
            </button>
            <button
              disabled={busy}
              onClick={() =>
                runCmd(async () => {
                  const r = await api.commandGripper('close')
                  setGripStatus(r.status)
                  onActionLog(`Gripper CLOSE -> ${(r.actual_width * 100).toFixed(2)} cm`)
                })
              }
              className="border border-ws-border bg-ws-panel2 py-1.5 font-mono text-[11px] text-ws-text hover:border-ws-accent"
            >
              Close (0.0 cm)
            </button>
          </div>

          <div className="border border-ws-border bg-ws-panel2 p-2">
            <div className="mb-1 flex justify-between font-mono text-[10px]">
              <span className="text-ws-muted">Target Aperture (m)</span>
              <span className="text-ws-accent">{gripWidth.toFixed(3)} m</span>
            </div>
            <input
              type="range"
              min={0}
              max={0.08}
              step={0.002}
              value={gripWidth}
              onChange={(e) => setGripWidth(parseFloat(e.target.value))}
              className="w-full"
            />
            <button
              disabled={busy}
              onClick={() =>
                runCmd(async () => {
                  const r = await api.commandGripper('set_width', gripWidth)
                  setGripStatus(r.status)
                  onActionLog(`Gripper width=${gripWidth.toFixed(3)}m`)
                })
              }
              className="mt-2 w-full border border-ws-accent bg-ws-accent/20 py-1 font-mono text-[11px] text-ws-text hover:bg-ws-accent/30"
            >
              Apply Gripper Width
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

