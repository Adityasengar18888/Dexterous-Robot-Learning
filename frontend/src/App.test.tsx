import { describe, expect, it, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import App from './App'

// Mock R3F Canvas in jsdom unit tests while verifying UI navigation, controls, and state wiring
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="three-canvas">{children}</div>,
  useFrame: () => {},
}))

vi.mock('@react-three/drei', () => ({
  Grid: () => null,
  Html: () => null,
  Line: () => null,
  OrbitControls: () => null,
}))

beforeEach(() => {
  globalThis.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      text: () =>
        Promise.resolve(
          JSON.stringify({
            tasks: [
              { task_id: 'pick_place', name: 'Pick and Place', env: 'PandaPickPlaceEnv', status: 'IMPLEMENTED', controllers: ['SCRIPTED', 'POLICY'], description: 'PickPlace' },
              { task_id: 'physical_fci_execution', name: 'Physical FCI', env: 'N/A', status: 'NOT IMPLEMENTED', controllers: [], description: 'Not attempted' },
            ],
            policies: [],
            experiments: [],
            phases: [],
            debugging_history: [],
            available_environments: ['PandaPickPlaceEnv'],
          }),
        ),
    } as Response),
  )

  class MockWS {
    onopen: (() => void) | null = null
    onmessage: ((e: { data: string }) => void) | null = null
    onerror: (() => void) | null = null
    onclose: (() => void) | null = null
    constructor() {
      setTimeout(() => this.onopen && this.onopen(), 0)
    }
    close() {}
  }
  vi.stubGlobal('WebSocket', MockWS)
})

describe('Dexterous-Robot-Learning Workstation UI', () => {
  it('renders layout, navigation sections, 3D viewport, and controls cleanly', () => {
    render(<App />)
    expect(screen.getByText('DEXTEROUS-ROBOT-LEARNING')).toBeTruthy()
    expect(screen.getByTestId('three-canvas')).toBeTruthy()
    expect(screen.getByText('7-DOF Joint Control (rad)')).toBeTruthy()
    expect(screen.getByText('E-STOP')).toBeTruthy()
  })

  it('switches across Cartesian, Gripper, Task Execution, Experiments, RL, and Settings', async () => {
    render(<App />)
    fireEvent.click(screen.getByText('cartesian'))
    expect(screen.getByText('Cartesian Pose & DLS IK')).toBeTruthy()

    fireEvent.click(screen.getByText('gripper'))
    expect(screen.getByText('Parallel-Jaw Gripper (0.00–0.08 m)')).toBeTruthy()

    fireEvent.click(screen.getByText('Task Execution'))
    expect(screen.getByText('Manipulation Task Runner')).toBeTruthy()

    fireEvent.click(screen.getByText('Reinforcement Learning'))
    expect(screen.getByText('Policies & Reinforcement Learning Registry')).toBeTruthy()

    fireEvent.click(screen.getByText('Experiments'))
    expect(screen.getByText(/Historical & Active Experiment Registry/)).toBeTruthy()

    fireEvent.click(screen.getByText('Logs & Diagnostics'))
    expect(screen.getByText(/Verified Debugging Chronicle/)).toBeTruthy()

    fireEvent.click(screen.getByText('Settings'))
    expect(screen.getByText('Workstation Settings & Hardware Boundary Declaration')).toBeTruthy()
  })
})

