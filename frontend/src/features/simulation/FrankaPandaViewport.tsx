import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Grid, Html, Line, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js'
import type { LinkTransform, SimulationSnapshot, ViewportOptions } from '../../types/robotics'

const ASSETS_BASE = (import.meta.env.VITE_API_BASE_URL || '') + '/assets/franka_panda/meshes'

interface MeshCacheEntry {
  obj: THREE.Group
}

const meshCache = new Map<string, Promise<THREE.Group>>()

function loadUrdfObjMesh(relMeshPath: string, wireframe = false): Promise<THREE.Group> {
  const key = `${relMeshPath}::${wireframe ? 'wire' : 'solid'}`
  const hit = meshCache.get(key)
  if (hit) return hit

  const p = new Promise<THREE.Group>((resolve) => {
    const url = `${ASSETS_BASE}/${relMeshPath}`
    const mtlPath = relMeshPath.replace(/\.obj$/i, '.mtl')
    const mtlUrl = `${ASSETS_BASE}/${mtlPath}`
    const folder = url.substring(0, url.lastIndexOf('/') + 1)
    const fileName = url.substring(url.lastIndexOf('/') + 1)

    const fallbackMat = new THREE.MeshStandardMaterial({
      color: wireframe ? '#789DC4' : '#DCE1E6',
      roughness: 0.45,
      metalness: 0.15,
      wireframe,
    })

    const loadObjWithMat = (materials?: MTLLoader.MaterialCreator) => {
      const objLoader = new OBJLoader()
      if (materials) {
        materials.preload()
        objLoader.setMaterials(materials)
      }
      objLoader.setPath(folder).load(
        fileName,
        (group) => {
          group.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              const m = child as THREE.Mesh
              if (wireframe) {
                m.material = fallbackMat
              } else {
                m.castShadow = true
                m.receiveShadow = true
              }
            }
          })
          resolve(group)
        },
        undefined,
        () => resolve(new THREE.Group()),
      )
    }

    if (!wireframe && relMeshPath.startsWith('visual/')) {
      const mtlLoader = new MTLLoader()
      mtlLoader.setPath(folder).load(
        mtlUrl.substring(mtlUrl.lastIndexOf('/') + 1),
        (mats) => loadObjWithMat(mats),
        undefined,
        () => loadObjWithMat(),
      )
    } else {
      loadObjWithMat()
    }
  })
  meshCache.set(key, p)
  return p
}

function ArticulatedLinkMesh({
  link,
  showCollision,
}: {
  link: LinkTransform
  showCollision: boolean
}) {
  const groupRef = useRef<THREE.Group>(null)
  const [solidMesh, setSolidMesh] = useState<THREE.Group | null>(null)
  const [colMesh, setColMesh] = useState<THREE.Group | null>(null)

  useEffect(() => {
    let active = true
    loadUrdfObjMesh(link.visual_mesh, false).then((g) => {
      if (active) setSolidMesh(g.clone())
    })
    return () => {
      active = false
    }
  }, [link.visual_mesh])

  useEffect(() => {
    if (!showCollision) {
      setColMesh(null)
      return
    }
    let active = true
    loadUrdfObjMesh(link.collision_mesh, true).then((g) => {
      if (active) setColMesh(g.clone())
    })
    return () => {
      active = false
    }
  }, [link.collision_mesh, showCollision])

  const targetPos = useMemo(() => new THREE.Vector3(...link.position), [link.position])
  const targetQuat = useMemo(
    () => new THREE.Quaternion(link.quaternion[0], link.quaternion[1], link.quaternion[2], link.quaternion[3]),
    [link.quaternion],
  )

  useFrame((_, delta) => {
    if (!groupRef.current) return
    const alpha = Math.min(1, delta * 24)
    groupRef.current.position.lerp(targetPos, alpha)
    groupRef.current.quaternion.slerp(targetQuat, alpha)
  })

  return (
    <group ref={groupRef} position={link.position} quaternion={link.quaternion}>
      <group rotation={link.mesh_rpy}>
        {solidMesh && <primitive object={solidMesh} />}
        {showCollision && colMesh && <primitive object={colMesh} />}
      </group>
    </group>
  )
}

function SceneContents({
  snapshot,
  options,
  orbitRef,
}: {
  snapshot: SimulationSnapshot | null
  options: ViewportOptions
  orbitRef: React.RefObject<any>
}) {
  const links = snapshot?.link_transforms ?? []
  const objects = snapshot?.objects ?? []
  const targets = snapshot?.targets ?? []
  const bins = snapshot?.bins ?? []
  const traj = snapshot?.ee_trajectory ?? []
  const eePos = snapshot?.ee_position ?? [0.307, 0, 0.485]
  const eeQuat = snapshot?.ee_orientation ?? [1, 0, 0, 0]

  return (
    <>
      <color attach="background" args={['#111315']} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[1.8, 2.5, 1.5]} intensity={1.15} castShadow />
      <directionalLight position={[-1.5, 1.2, -1.2]} intensity={0.4} color="#789DC4" />

      {/* Ground grid in standard Three.js XZ plane aligned under Z-up PyBullet root */}
      {options.showGrid && (
        <Grid
          args={[3.0, 3.0]}
          cellSize={0.1}
          cellThickness={0.5}
          cellColor="#22272C"
          sectionSize={0.5}
          sectionThickness={1.0}
          sectionColor="#343B42"
          fadeDistance={4}
          position={[0.3, 0, 0]}
        />
      )}

      {/* PyBullet Z-up coordinate space rotated cleanly into Three.js Y-up */}
      <group rotation={[-Math.PI / 2, 0, 0]}>
        {/* Tabletop surface */}
        <mesh position={[0.45, 0, -0.004]} receiveShadow>
          <boxGeometry args={[1.2, 1.0, 0.008]} />
          <meshStandardMaterial color="#1B1F23" roughness={0.88} metalness={0.05} />
        </mesh>

        {/* Base world axes frame */}
        {options.showFrames && <axesHelper args={[0.25]} />}

        {/* Authoritative articulated Franka Panda links */}
        {links.map((lk) => (
          <ArticulatedLinkMesh key={lk.name} link={lk} showCollision={options.showCollision} />
        ))}

        {/* End-effector coordinate triad */}
        {options.showEeFrame && (
          <group position={eePos} quaternion={eeQuat}>
            <axesHelper args={[0.09]} />
            {options.showLabels && (
              <Html distanceFactor={1.4} position={[0, 0, 0.06]}>
                <div className="pointer-events-none whitespace-nowrap rounded-sm border border-ws-border bg-ws-panel/95 px-1.5 py-0.5 font-mono text-[10px] text-ws-accent">
                  EE [{eePos.map((v) => v.toFixed(2)).join(', ')}]
                </div>
              </Html>
            )}
          </group>
        )}

        {/* Dynamic scene objects (cubes, peg) streamed from PyBullet */}
        {objects.map((ob) => (
          <group key={ob.id} position={ob.position} quaternion={ob.quaternion}>
            <mesh castShadow receiveShadow>
              <boxGeometry args={[ob.size, ob.size, ob.size]} />
              <meshStandardMaterial color={ob.color} roughness={0.4} metalness={0.1} />
            </mesh>
            {options.showLabels && (
              <Html distanceFactor={1.4} position={[0, 0, ob.size]}>
                <div className="pointer-events-none whitespace-nowrap rounded-sm border border-ws-border bg-ws-bg/90 px-1 py-0.5 font-mono text-[9px] text-ws-text">
                  {ob.id}
                </div>
              </Html>
            )}
          </group>
        ))}

        {/* Sorting bins */}
        {bins.map((bn) => (
          <group key={bn.id} position={bn.position}>
            <mesh position={[0, 0, 0.015]} receiveShadow>
              <boxGeometry args={[0.14, 0.14, 0.03]} />
              <meshStandardMaterial color={bn.color} opacity={0.35} transparent />
            </mesh>
            {options.showLabels && (
              <Html distanceFactor={1.6} position={[0, 0, 0.05]}>
                <div className="pointer-events-none rounded-sm border border-ws-border bg-ws-panel px-1 py-0.5 font-mono text-[9px] text-ws-muted">
                  {bn.id}
                </div>
              </Html>
            )}
          </group>
        ))}

        {/* Target markers */}
        {targets.map((tg) => (
          <group key={tg.id} position={tg.position}>
            <mesh>
              <sphereGeometry args={[0.022, 16, 16]} />
              <meshStandardMaterial color={tg.color} opacity={0.65} transparent wireframe={tg.kind === 'slot'} />
            </mesh>
            {options.showLabels && (
              <Html distanceFactor={1.5} position={[0, 0, 0.04]}>
                <div className="pointer-events-none rounded-sm border border-ws-border bg-ws-panel px-1 py-0.5 font-mono text-[9px] text-ws-success">
                  {tg.id}
                </div>
              </Html>
            )}
          </group>
        ))}

        {/* Authoritative EE 3D Trajectory */}
        {options.showTrajectory && traj.length >= 2 && (
          <Line points={traj} color="#789DC4" lineWidth={1.6} transparent opacity={0.85} />
        )}
      </group>

      <OrbitControls ref={orbitRef} makeDefault target={[0.38, 0.18, 0]} enableDamping dampingFactor={0.08} />
    </>
  )
}

export const FrankaPandaViewport: React.FC<{
  snapshot: SimulationSnapshot | null
  options: ViewportOptions
  setOptions: React.Dispatch<React.SetStateAction<ViewportOptions>>
}> = ({ snapshot, options, setOptions }) => {
  const orbitRef = useRef<any>(null)

  const handleResetView = () => {
    if (orbitRef.current) {
      orbitRef.current.object.position.set(1.15, 0.78, 0.88)
      orbitRef.current.target.set(0.38, 0.18, 0)
      orbitRef.current.update()
    }
  }

  return (
    <div className="relative h-full w-full overflow-hidden border border-ws-border bg-ws-bg">
      {/* Top-left compact viewport HUD */}
      <div className="pointer-events-none absolute left-2.5 top-2.5 z-10 flex items-center gap-2">
        <span className="border border-ws-border bg-ws-panel/90 px-2 py-0.5 font-mono text-[10px] text-ws-text">
          PYBULLET AUTHORITATIVE STREAM
        </span>
        <span className="border border-ws-border bg-ws-panel/90 px-2 py-0.5 font-mono text-[10px] text-ws-muted">
          ENV: {snapshot?.active_env ?? 'PandaPickPlaceEnv'} | LINKS: {snapshot?.link_transforms.length ?? 0}/11
        </span>
      </div>

      {/* Top-right compact visibility toggles */}
      <div className="absolute right-2.5 top-2.5 z-10 flex flex-wrap items-center gap-1">
        {(
          [
            ['showGrid', 'Grid'],
            ['showFrames', 'Frames'],
            ['showEeFrame', 'EE'],
            ['showCollision', 'Collision'],
            ['showTrajectory', 'Traj'],
            ['showLabels', 'Labels'],
          ] as [keyof ViewportOptions, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setOptions((o) => ({ ...o, [key]: !o[key] }))}
            className={`border px-2 py-0.5 font-mono text-[10px] transition-colors ${
              options[key]
                ? 'border-ws-accent bg-ws-accent/15 text-ws-text'
                : 'border-ws-border bg-ws-panel/90 text-ws-muted hover:text-ws-text'
            }`}
          >
            {label}
          </button>
        ))}
        <button
          onClick={handleResetView}
          className="border border-ws-border bg-ws-panel/90 px-2 py-0.5 font-mono text-[10px] text-ws-text hover:border-ws-accent"
        >
          Reset Cam
        </button>
      </div>

      <Canvas shadows camera={{ position: [1.15, 0.78, 0.88], fov: 42, near: 0.02, far: 25 }}>
        <SceneContents snapshot={snapshot} options={options} orbitRef={orbitRef} />
      </Canvas>
    </div>
  )
}

