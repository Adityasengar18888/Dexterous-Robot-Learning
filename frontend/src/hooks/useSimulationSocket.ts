import { useEffect, useRef, useState } from 'react'
import type { DiagnosticEvent, SimulationSnapshot } from '../types/robotics'

const WS_BASE = import.meta.env.VITE_WS_BASE_URL || ''

function buildWsUrl(path: string): string {
  if (WS_BASE) return `${WS_BASE}${path}`
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}${path}`
}

export function useSimulationSocket() {
  const [snapshot, setSnapshot] = useState<SimulationSnapshot | null>(null)
  const [events, setEvents] = useState<DiagnosticEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [wsError, setWsError] = useState<string | null>(null)
  const retryRef = useRef<number>(1000)

  useEffect(() => {
    let closedByEffect = false
    let wsSim: WebSocket | null = null
    let wsEv: WebSocket | null = null
    let timerSim: number | undefined
    let timerEv: number | undefined

    const connectSim = () => {
      if (closedByEffect) return
      try {
        wsSim = new WebSocket(buildWsUrl('/ws/simulation'))
        wsSim.onopen = () => {
          setConnected(true)
          setWsError(null)
          retryRef.current = 1000
        }
        wsSim.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data)
            if (msg.type === 'simulation_state' && msg.payload) {
              setSnapshot(msg.payload as SimulationSnapshot)
            }
          } catch {
            // ignore malformed frame
          }
        }
        wsSim.onerror = () => {
          setWsError('WebSocket connection error (/ws/simulation)')
        }
        wsSim.onclose = () => {
          setConnected(false)
          if (!closedByEffect) {
            const wait = retryRef.current
            retryRef.current = Math.min(8000, wait * 1.5)
            timerSim = window.setTimeout(connectSim, wait)
          }
        }
      } catch (e) {
        setWsError(String(e))
      }
    }

    const connectEvents = () => {
      if (closedByEffect) return
      try {
        wsEv = new WebSocket(buildWsUrl('/ws/events'))
        wsEv.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data)
            if (msg.type === 'events_snapshot' && Array.isArray(msg.events)) {
              setEvents(msg.events as DiagnosticEvent[])
            }
          } catch {
            // ignore
          }
        }
        wsEv.onclose = () => {
          if (!closedByEffect) {
            timerEv = window.setTimeout(connectEvents, 2500)
          }
        }
      } catch {
        // ignore
      }
    }

    connectSim()
    connectEvents()

    return () => {
      closedByEffect = true
      if (timerSim) clearTimeout(timerSim)
      if (timerEv) clearTimeout(timerEv)
      wsSim?.close()
      wsEv?.close()
    }
  }, [])

  return { snapshot, events, connected, wsError }
}

