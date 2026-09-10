/**
 * AstroActAi — 3D Microgravity Space Operations & Mission Control
 * WebGL 3D Space Station Payload Visualizer & Real-Time Telemetry Engine
 * Features:
 *   - Dual-Hand 3D Tracking (Left Hand & Right Hand Ambidextrous Manipulation)
 *   - In-Hand AI Object Identification & Dynamic 3D Grasp Attachment
 *   - Live In-Hand Payload Specifications Telemetry
 *   - Three.js WebGL OrbitControls & Space Station Glovebox Environment
 */

// MediaPipe 33-Keypoint Skeletal Connections
const POSE_CONNECTIONS = [
  // Torso
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  // Left Arm
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["left_wrist", "left_pinky"],
  ["left_wrist", "left_index"],
  ["left_wrist", "left_thumb"],
  // Right Arm
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["right_wrist", "right_pinky"],
  ["right_wrist", "right_index"],
  ["right_wrist", "right_thumb"],
  // Left Leg
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["left_ankle", "left_heel"],
  ["left_ankle", "left_foot_index"],
  // Right Leg
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["right_ankle", "right_heel"],
  ["right_ankle", "right_foot_index"],
  // Head
  ["nose", "left_eye"],
  ["nose", "right_eye"],
  ["left_eye", "left_ear"],
  ["right_eye", "right_ear"]
];

const LANDMARK_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
  "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
  "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
  "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
  "right_heel", "left_foot_index", "right_foot_index"
];

let activeProtocol = null;
let currentStepIdx = 1;
let ws = null;
let deviationsCount = 0;
let lastPose = null;
let audioCtx = null;

// Three.js 3D Engine Globals
let scene, camera, renderer, controls;
let jointMeshes = {};
let boneMeshes = [];
let targetMeshes = {};
let defaultTargetPositions = {};
let leftWristPulseSphere = null;
let rightWristPulseSphere = null;
let leftKneePulseSphere = null;
let rightKneePulseSphere = null;
let wristTrajectoryLine = null;
let wristHistory = [];

// ==============================================================================
// 1. THREE.JS 3D SPACE STATION & DUAL-HAND VISUALIZER
// ==============================================================================

function init3DScene() {
  const container = document.getElementById("three-container");
  if (!container) return;

  const width = container.clientWidth || 640;
  const height = container.clientHeight || 480;

  // Scene
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x060910, 0.08);

  // Camera
  camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 100);
  camera.position.set(0, 0.2, 2.8);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  container.appendChild(renderer.domElement);

  // OrbitControls
  if (typeof THREE.OrbitControls !== "undefined") {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 6.0;
    controls.minDistance = 0.8;
  }

  // Lighting
  const ambientLight = new THREE.AmbientLight(0x233348, 1.8);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0x00e5ff, 2.2);
  keyLight.position.set(2, 4, 3);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xff9100, 1.0);
  fillLight.position.set(-3, -2, 2);
  scene.add(fillLight);

  // 1. Deep Space Starfield Particles
  createStarfield();

  // 2. Space Station Payload Rack Frame
  createPayloadRackStructure();

  // 3. 3D Payload Target Components
  create3DPayloadTargets();

  // 4. 3D Astronaut Skeletal Rig (Dual-Hand Ambidextrous Tracking)
  createAstronautSkeletalRig();

  window.addEventListener("resize", onWindowResize);
  animate3D();
}

function createStarfield() {
  const starsCount = 1200;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(starsCount * 3);

  for (let i = 0; i < starsCount * 3; i += 3) {
    positions[i] = (Math.random() - 0.5) * 40;
    positions[i + 1] = (Math.random() - 0.5) * 40;
    positions[i + 2] = (Math.random() - 0.5) * 40;
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0x8494b2,
    size: 0.08,
    transparent: true,
    opacity: 0.65
  });

  const starfield = new THREE.Points(geometry, material);
  scene.add(starfield);
}

function createPayloadRackStructure() {
  const rackGroup = new THREE.Group();

  const boxGeom = new THREE.BoxGeometry(2.2, 1.7, 1.0);
  const edges = new THREE.EdgesGeometry(boxGeom);
  const lineMat = new THREE.LineBasicMaterial({ color: 0x1a273f, linewidth: 1 });
  const rackBox = new THREE.LineSegments(edges, lineMat);
  rackGroup.add(rackBox);

  const grid = new THREE.GridHelper(2.0, 16, 0x00e5ff, 0x101b2d);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.5;
  rackGroup.add(grid);

  const fidGeom = new THREE.BoxGeometry(0.06, 0.06, 0.06);
  const fidMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
  const corners = [
    [-1.0, 0.75, -0.48],
    [1.0, 0.75, -0.48],
    [-1.0, -0.75, -0.48],
    [1.0, -0.75, -0.48]
  ];
  // Station Foot Restraint Rails (Station Deck Floor)
  const restraintMat = new THREE.MeshStandardMaterial({
    color: 0x00e676,
    metalness: 0.8,
    roughness: 0.2,
    emissive: 0x003311,
    emissiveIntensity: 0.6
  });
  const railGeom = new THREE.BoxGeometry(0.35, 0.04, 0.08);
  const leftRail = new THREE.Mesh(railGeom, restraintMat);
  leftRail.position.set(-0.35, -0.82, 0.0);
  rackGroup.add(leftRail);

  const rightRail = new THREE.Mesh(railGeom, restraintMat);
  rightRail.position.set(0.35, -0.82, 0.0);
  rackGroup.add(rightRail);

  scene.add(rackGroup);
}

function create3DPayloadTargets(proto = activeProtocol) {
  if (!scene) return;

  // 1. Remove previous target meshes from scene
  Object.keys(targetMeshes).forEach((key) => {
    if (targetMeshes[key]) {
      scene.remove(targetMeshes[key]);
      if (targetMeshes[key].geometry) targetMeshes[key].geometry.dispose();
      if (targetMeshes[key].material) targetMeshes[key].material.dispose();
    }
  });
  targetMeshes = {};
  defaultTargetPositions = {};

  const protoId = proto ? proto.protocol_id : "BAS-EXP-BIO-2026";

  if (protoId === "BAS-EXP-FLUID-2026") {
    // SCENARIO 2: FLUID PHYSICS WETLAB
    // 1. Syringe Injector
    const syrGroup = new THREE.Group();
    const barrelGeom = new THREE.CylinderGeometry(0.045, 0.045, 0.26, 16);
    const barrelMat = new THREE.MeshStandardMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.85,
      metalness: 0.3,
      roughness: 0.1,
      emissive: 0x004455,
      emissiveIntensity: 0.6
    });
    const barrel = new THREE.Mesh(barrelGeom, barrelMat);
    syrGroup.add(barrel);
    const tipGeom = new THREE.ConeGeometry(0.015, 0.08, 12);
    const tipMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.9, roughness: 0.1 });
    const tip = new THREE.Mesh(tipGeom, tipMat);
    tip.position.set(0, 0.16, 0);
    syrGroup.add(tip);
    const posSyr = new THREE.Vector3(-0.60, 0.35, -0.25);
    syrGroup.position.copy(posSyr);
    scene.add(syrGroup);
    targetMeshes["Syringe_Injector"] = syrGroup;
    defaultTargetPositions["Syringe_Injector"] = posSyr.clone();

    // 2. Sample Vial
    const vialGroup = new THREE.Group();
    const bodyGeom = new THREE.CylinderGeometry(0.05, 0.05, 0.18, 16);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0xffab00,
      metalness: 0.2,
      roughness: 0.2,
      emissive: 0x4a2a00,
      emissiveIntensity: 0.5
    });
    const vBody = new THREE.Mesh(bodyGeom, bodyMat);
    vialGroup.add(vBody);
    const capGeom = new THREE.CylinderGeometry(0.052, 0.052, 0.035, 16);
    const capMat = new THREE.MeshStandardMaterial({ color: 0x78909c, metalness: 0.8, roughness: 0.3 });
    const vCap = new THREE.Mesh(capGeom, capMat);
    vCap.position.set(0, 0.10, 0);
    vialGroup.add(vCap);
    const posVial = new THREE.Vector3(0.0, 0.35, -0.25);
    vialGroup.position.copy(posVial);
    scene.add(vialGroup);
    targetMeshes["Sample_Vial"] = vialGroup;
    defaultTargetPositions["Sample_Vial"] = posVial.clone();

    // 3. Manifold Port A
    const manifoldGeom = new THREE.BoxGeometry(0.22, 0.22, 0.14);
    const manifoldMat = new THREE.MeshStandardMaterial({ color: 0x1e3a5f, metalness: 0.8, roughness: 0.3 });
    const manifold = new THREE.Mesh(manifoldGeom, manifoldMat);
    manifold.position.set(0.60, -0.25, -0.42);
    scene.add(manifold);
    targetMeshes["Manifold_Port_A"] = manifold;

    // 4. Pinch Valve
    const pinchGeom = new THREE.TorusGeometry(0.08, 0.025, 8, 24);
    const pinchMat = new THREE.MeshStandardMaterial({
      color: 0x00bcd4,
      metalness: 0.7,
      roughness: 0.2,
      emissive: 0x004d40,
      emissiveIntensity: 0.5
    });
    const pinch = new THREE.Mesh(pinchGeom, pinchMat);
    pinch.position.set(0.60, -0.55, -0.40);
    scene.add(pinch);
    targetMeshes["Pinch_Valve"] = pinch;

  } else if (protoId === "BAS-EXP-AVIONICS-2026") {
    // SCENARIO 3: AVIONICS MAINTENANCE
    // 1. SpaceWire Harness
    const harnessGroup = new THREE.Group();
    const connGeom = new THREE.CylinderGeometry(0.065, 0.065, 0.15, 16);
    const connMat = new THREE.MeshStandardMaterial({
      color: 0xffd700,
      metalness: 0.9,
      roughness: 0.2,
      emissive: 0x665500,
      emissiveIntensity: 0.6
    });
    const conn = new THREE.Mesh(connGeom, connMat);
    harnessGroup.add(conn);
    const wireGeom = new THREE.TorusGeometry(0.12, 0.02, 8, 24, Math.PI);
    const wireMat = new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 0.5, roughness: 0.5 });
    const wire = new THREE.Mesh(wireGeom, wireMat);
    wire.rotation.x = Math.PI / 2;
    wire.position.set(0, -0.07, 0);
    harnessGroup.add(wire);
    const posHarn = new THREE.Vector3(-0.60, 0.35, -0.25);
    harnessGroup.position.copy(posHarn);
    scene.add(harnessGroup);
    targetMeshes["SpaceWire_Harness"] = harnessGroup;
    defaultTargetPositions["SpaceWire_Harness"] = posHarn.clone();

    // 2. Avionics Port 4
    const portGeom = new THREE.CylinderGeometry(0.08, 0.08, 0.08, 16);
    const portMat = new THREE.MeshStandardMaterial({
      color: 0x0288d1,
      metalness: 0.8,
      roughness: 0.3,
      emissive: 0x01579b,
      emissiveIntensity: 0.6
    });
    const port = new THREE.Mesh(portGeom, portMat);
    port.position.set(0.0, 0.35, -0.25);
    scene.add(port);
    targetMeshes["Avionics_Port_4"] = port;
    defaultTargetPositions["Avionics_Port_4"] = port.position.clone();

    // 3. Torque Wrench
    const torqueGroup = new THREE.Group();
    const shaftGeom = new THREE.CylinderGeometry(0.02, 0.02, 0.32, 12);
    const shaftMat = new THREE.MeshStandardMaterial({ color: 0xdcdcdc, metalness: 0.9, roughness: 0.2 });
    const shaft = new THREE.Mesh(shaftGeom, shaftMat);
    torqueGroup.add(shaft);
    const headGeom = new THREE.BoxGeometry(0.06, 0.05, 0.08);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x212121, metalness: 0.7, roughness: 0.4 });
    const head = new THREE.Mesh(headGeom, headMat);
    head.position.set(0, 0.16, 0.02);
    torqueGroup.add(head);
    torqueGroup.position.set(0.60, -0.25, -0.42);
    scene.add(torqueGroup);
    targetMeshes["Torque_Wrench"] = torqueGroup;

    // 4. Breaker Toggle
    const breakerGroup = new THREE.Group();
    const bBaseGeom = new THREE.BoxGeometry(0.12, 0.16, 0.06);
    const bBaseMat = new THREE.MeshStandardMaterial({ color: 0x37474f, metalness: 0.6, roughness: 0.4 });
    const bBase = new THREE.Mesh(bBaseGeom, bBaseMat);
    breakerGroup.add(bBase);
    const leverGeom = new THREE.CylinderGeometry(0.015, 0.015, 0.10, 8);
    const leverMat = new THREE.MeshStandardMaterial({
      color: 0xff1744,
      metalness: 0.8,
      roughness: 0.2,
      emissive: 0x880000,
      emissiveIntensity: 0.7
    });
    const lever = new THREE.Mesh(leverGeom, leverMat);
    lever.position.set(0, 0.03, 0.04);
    lever.rotation.x = Math.PI / 4;
    breakerGroup.add(lever);
    breakerGroup.position.set(0.60, -0.55, -0.40);
    scene.add(breakerGroup);
    targetMeshes["Breaker_Toggle"] = breakerGroup;

  } else if (protoId === "BAS-EMERGENCY-01") {
    // SCENARIO 4: EMERGENCY AIRLOCK DEPRESS & HATCH SEAL
    // 1. Breather Mask
    const maskGroup = new THREE.Group();
    const domeGeom = new THREE.SphereGeometry(0.08, 16, 16, 0, Math.PI * 2, 0, Math.PI * 0.6);
    const domeMat = new THREE.MeshStandardMaterial({
      color: 0xffea00,
      metalness: 0.4,
      roughness: 0.3,
      emissive: 0x665500,
      emissiveIntensity: 0.6
    });
    const dome = new THREE.Mesh(domeGeom, domeMat);
    maskGroup.add(dome);
    const posMask = new THREE.Vector3(-0.60, 0.35, -0.25);
    maskGroup.position.copy(posMask);
    scene.add(maskGroup);
    targetMeshes["Breather_Mask"] = maskGroup;
    defaultTargetPositions["Breather_Mask"] = posMask.clone();

    // 2. Equalization Valve
    const valveGroup = new THREE.Group();
    const wheelGeom = new THREE.TorusGeometry(0.085, 0.02, 8, 24);
    const wheelMat = new THREE.MeshStandardMaterial({
      color: 0xff9100,
      metalness: 0.8,
      roughness: 0.3,
      emissive: 0x663300,
      emissiveIntensity: 0.5
    });
    const wheel = new THREE.Mesh(wheelGeom, wheelMat);
    valveGroup.add(wheel);
    const hubGeom = new THREE.CylinderGeometry(0.025, 0.025, 0.04, 12);
    const hubMat = new THREE.MeshStandardMaterial({ color: 0x263238, metalness: 0.9, roughness: 0.2 });
    const hub = new THREE.Mesh(hubGeom, hubMat);
    valveGroup.add(hub);
    const posValve = new THREE.Vector3(0.0, 0.35, -0.25);
    valveGroup.position.copy(posValve);
    scene.add(valveGroup);
    targetMeshes["Equalization_Valve"] = valveGroup;
    defaultTargetPositions["Equalization_Valve"] = posValve.clone();

    // 3. Hatch Dog Handle
    const dogGroup = new THREE.Group();
    const dogHandleGeom = new THREE.CylinderGeometry(0.025, 0.025, 0.34, 12);
    const dogHandleMat = new THREE.MeshStandardMaterial({
      color: 0xd50000,
      metalness: 0.8,
      roughness: 0.2,
      emissive: 0x550000,
      emissiveIntensity: 0.8
    });
    const dogHandle = new THREE.Mesh(dogHandleGeom, dogHandleMat);
    dogHandle.rotation.z = Math.PI / 4;
    dogGroup.add(dogHandle);
    dogGroup.position.set(0.60, -0.25, -0.42);
    scene.add(dogGroup);
    targetMeshes["Hatch_Dog_Handle"] = dogGroup;

    // 4. Secondary Lock Bar
    const secLockGeom = new THREE.BoxGeometry(0.28, 0.06, 0.06);
    const secLockMat = new THREE.MeshStandardMaterial({
      color: 0x00e676,
      metalness: 0.8,
      roughness: 0.3,
      emissive: 0x004d20,
      emissiveIntensity: 0.6
    });
    const secLock = new THREE.Mesh(secLockGeom, secLockMat);
    secLock.position.set(0.60, -0.55, -0.40);
    scene.add(secLock);
    targetMeshes["Secondary_Lock"] = secLock;

  } else {
    // DEFAULT: SCENARIO 1: BIO PROTOCOL
    const compAGeom = new THREE.CylinderGeometry(0.08, 0.08, 0.22, 16);
    const compAMat = new THREE.MeshStandardMaterial({
      color: 0x00e5ff,
      metalness: 0.8,
      roughness: 0.2,
      emissive: 0x003b4f,
      emissiveIntensity: 0.6
    });
    const compA = new THREE.Mesh(compAGeom, compAMat);
    const posA = new THREE.Vector3(-0.60, 0.35, -0.25);
    compA.position.copy(posA);
    scene.add(compA);
    targetMeshes["Component_A"] = compA;
    defaultTargetPositions["Component_A"] = posA.clone();

    const compBGeom = new THREE.CylinderGeometry(0.08, 0.08, 0.22, 16);
    const compBMat = new THREE.MeshStandardMaterial({
      color: 0xffab00,
      metalness: 0.8,
      roughness: 0.3,
      emissive: 0x4a2a00,
      emissiveIntensity: 0.5
    });
    const compB = new THREE.Mesh(compBGeom, compBMat);
    const posB = new THREE.Vector3(0.0, 0.35, -0.25);
    compB.position.copy(posB);
    scene.add(compB);
    targetMeshes["Component_B"] = compB;
    defaultTargetPositions["Component_B"] = posB.clone();

    const slotGeom = new THREE.BoxGeometry(0.24, 0.28, 0.12);
    const slotMat = new THREE.MeshStandardMaterial({ color: 0x233758, metalness: 0.9, roughness: 0.4 });
    const slot = new THREE.Mesh(slotGeom, slotMat);
    slot.position.set(0.60, -0.25, -0.42);
    scene.add(slot);
    targetMeshes["Rack_Slot_1"] = slot;

    const latchGeom = new THREE.TorusGeometry(0.09, 0.025, 8, 24);
    const latchMat = new THREE.MeshStandardMaterial({ color: 0x00e676, metalness: 0.7, roughness: 0.3 });
    const latch = new THREE.Mesh(latchGeom, latchMat);
    latch.position.set(0.60, -0.55, -0.40);
    scene.add(latch);
    targetMeshes["Latch_Mechanism"] = latch;
  }
}

function createAstronautSkeletalRig() {
  const jointGroup = new THREE.Group();

  const defaultJointMat = new THREE.MeshStandardMaterial({
    color: 0x00e676,
    emissive: 0x006633,
    emissiveIntensity: 0.8,
    roughness: 0.2
  });

  // Distinct Materials for Both Hands
  const leftHandMat = new THREE.MeshStandardMaterial({
    color: 0xda70d6, // Violet
    emissive: 0x8b008b,
    emissiveIntensity: 1.0,
    roughness: 0.1
  });

  const rightHandMat = new THREE.MeshStandardMaterial({
    color: 0xff9100, // Orange
    emissive: 0x994400,
    emissiveIntensity: 1.0,
    roughness: 0.1
  });

  const headMat = new THREE.MeshStandardMaterial({
    color: 0x00e5ff,
    emissive: 0x0088aa,
    emissiveIntensity: 0.9
  });

  const leftLegMat = new THREE.MeshStandardMaterial({
    color: 0x00e676,
    emissive: 0x00c853,
    emissiveIntensity: 0.95,
    roughness: 0.1
  });

  const rightLegMat = new THREE.MeshStandardMaterial({
    color: 0x00e5ff,
    emissive: 0x00b0ff,
    emissiveIntensity: 0.95,
    roughness: 0.1
  });

  // Create 33 Joint Spheres (Enlarged & High-Vis Glow)
  LANDMARK_NAMES.forEach((name) => {
    const isLeftWrist = name === "left_wrist";
    const isRightWrist = name === "right_wrist";
    const isLeftLeg = name.includes("left_knee") || name.includes("left_ankle") || name.includes("left_foot") || name === "left_heel";
    const isRightLeg = name.includes("right_knee") || name.includes("right_ankle") || name.includes("right_foot") || name === "right_heel";
    const isHead = name === "nose" || name.includes("eye") || name.includes("ear");

    const radius = (isLeftWrist || isRightWrist) ? 0.052 : (isLeftLeg || isRightLeg) ? 0.044 : isHead ? 0.040 : 0.032;
    const geom = new THREE.SphereGeometry(radius, 16, 16);
    const mat = isLeftWrist ? leftHandMat : isRightWrist ? rightHandMat : isLeftLeg ? leftLegMat : isRightLeg ? rightLegMat : isHead ? headMat : defaultJointMat;
    const sphere = new THREE.Mesh(geom, mat);
    sphere.visible = true;
    jointMeshes[name] = sphere;
    jointGroup.add(sphere);
  });

  // 3D Pulse Spheres for BOTH Hands
  const pulseGeom = new THREE.SphereGeometry(0.14, 16, 16);
  leftWristPulseSphere = new THREE.Mesh(
    pulseGeom,
    new THREE.MeshBasicMaterial({ color: 0xda70d6, wireframe: true, transparent: true, opacity: 0.45 })
  );
  leftWristPulseSphere.visible = true;
  jointGroup.add(leftWristPulseSphere);

  rightWristPulseSphere = new THREE.Mesh(
    pulseGeom,
    new THREE.MeshBasicMaterial({ color: 0xff9100, wireframe: true, transparent: true, opacity: 0.45 })
  );
  rightWristPulseSphere.visible = true;
  jointGroup.add(rightWristPulseSphere);

  // 3D Pulse Spheres for BOTH Knees (Foot Restraints)
  const kneePulseGeom = new THREE.SphereGeometry(0.11, 16, 16);
  leftKneePulseSphere = new THREE.Mesh(
    kneePulseGeom,
    new THREE.MeshBasicMaterial({ color: 0x00e676, wireframe: true, transparent: true, opacity: 0.40 })
  );
  leftKneePulseSphere.visible = true;
  jointGroup.add(leftKneePulseSphere);

  rightKneePulseSphere = new THREE.Mesh(
    kneePulseGeom,
    new THREE.MeshBasicMaterial({ color: 0x00e5ff, wireframe: true, transparent: true, opacity: 0.40 })
  );
  rightKneePulseSphere.visible = true;
  jointGroup.add(rightKneePulseSphere);

  // 3D Bone Connections (High-Thickness Luminous Rods)
  const boneMat = new THREE.MeshStandardMaterial({
    color: 0x00e5ff,
    metalness: 0.1,
    roughness: 0.1,
    emissive: 0x00d4ff,
    emissiveIntensity: 0.85
  });

  POSE_CONNECTIONS.forEach(() => {
    const geom = new THREE.CylinderGeometry(0.022, 0.022, 1.0, 10);
    const bone = new THREE.Mesh(geom, boneMat);
    bone.visible = true;
    boneMeshes.push(bone);
    jointGroup.add(bone);
  });

  // 3D Trajectory Ribbon
  const trajGeom = new THREE.BufferGeometry();
  wristTrajectoryLine = new THREE.Line(
    trajGeom,
    new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 3, transparent: true, opacity: 0.85 })
  );
  jointGroup.add(wristTrajectoryLine);

  scene.add(jointGroup);

  // Initialize with prominent default microgravity Neutral Body Posture (NBP)
  renderDefaultBaselineSkeleton();
}

function renderDefaultBaselineSkeleton() {
  const nbpKeypoints = {
    "nose": { rack_x: 0.0, rack_y: -0.28, rack_z: 0.05 },
    "left_eye": { rack_x: -0.04, rack_y: -0.32, rack_z: 0.05 },
    "right_eye": { rack_x: 0.04, rack_y: -0.32, rack_z: 0.05 },
    "left_ear": { rack_x: -0.09, rack_y: -0.31, rack_z: 0.02 },
    "right_ear": { rack_x: 0.09, rack_y: -0.31, rack_z: 0.02 },
    "left_shoulder": { rack_x: -0.18, rack_y: -0.16, rack_z: 0.0 },
    "right_shoulder": { rack_x: 0.18, rack_y: -0.16, rack_z: 0.0 },
    "left_elbow": { rack_x: -0.28, rack_y: 0.02, rack_z: 0.10 },
    "right_elbow": { rack_x: 0.28, rack_y: 0.02, rack_z: 0.10 },
    "left_wrist": { rack_x: -0.32, rack_y: 0.20, rack_z: 0.18 },
    "right_wrist": { rack_x: 0.32, rack_y: 0.20, rack_z: 0.18 },
    "left_pinky": { rack_x: -0.35, rack_y: 0.23, rack_z: 0.18 },
    "right_pinky": { rack_x: 0.35, rack_y: 0.23, rack_z: 0.18 },
    "left_index": { rack_x: -0.33, rack_y: 0.25, rack_z: 0.18 },
    "right_index": { rack_x: 0.33, rack_y: 0.25, rack_z: 0.18 },
    "left_thumb": { rack_x: -0.30, rack_y: 0.22, rack_z: 0.18 },
    "right_thumb": { rack_x: 0.30, rack_y: 0.22, rack_z: 0.18 },
    "left_hip": { rack_x: -0.11, rack_y: 0.18, rack_z: -0.05 },
    "right_hip": { rack_x: 0.11, rack_y: 0.18, rack_z: -0.05 },
    "left_knee": { rack_x: -0.14, rack_y: 0.42, rack_z: 0.10 },
    "right_knee": { rack_x: 0.14, rack_y: 0.42, rack_z: 0.10 },
    "left_ankle": { rack_x: -0.15, rack_y: 0.65, rack_z: 0.12 },
    "right_ankle": { rack_x: 0.15, rack_y: 0.65, rack_z: 0.12 },
    "left_heel": { rack_x: -0.17, rack_y: 0.68, rack_z: 0.10 },
    "right_heel": { rack_x: 0.17, rack_y: 0.68, rack_z: 0.10 },
    "left_foot_index": { rack_x: -0.14, rack_y: 0.70, rack_z: 0.16 },
    "right_foot_index": { rack_x: 0.14, rack_y: 0.70, rack_z: 0.16 }
  };
  update3DSkeleton({ keypoints: nbpKeypoints });
}

function update3DSkeleton(pose) {
  if (!pose || !pose.keypoints) return;
  const kps = pose.keypoints;

  function mapTo3D(pt) {
    const rx = (pt.rack_x !== undefined) ? pt.rack_x : (pt.x !== undefined ? (pt.x - 0.5) : 0);
    const ry = (pt.rack_y !== undefined) ? pt.rack_y : (pt.y !== undefined ? (pt.y - 0.5) : 0);
    const rz = (pt.rack_z !== undefined) ? pt.rack_z : (pt.z !== undefined ? pt.z : 0.0);

    const x3d = rx * 2.5;
    const y3d = -ry * 2.3 + 0.15;
    const z3d = -rz * 1.6;
    return new THREE.Vector3(x3d, y3d, z3d);
  }

  let leftWristPos3D = null;
  let rightWristPos3D = null;

  // Update Joint Positions
  Object.keys(jointMeshes).forEach(name => {
    const mesh = jointMeshes[name];
    if (kps[name]) {
      const pos = mapTo3D(kps[name]);
      mesh.position.copy(pos);
      mesh.visible = true;

      // Left Hand pulse & tracking
      if (name === "left_wrist") {
        leftWristPos3D = pos.clone();
        if (leftWristPulseSphere) {
          leftWristPulseSphere.position.copy(pos);
          leftWristPulseSphere.visible = true;
          const s = 1.0 + Math.sin(Date.now() * 0.009) * 0.25;
          leftWristPulseSphere.scale.set(s, s, s);
        }
      }

      // Right Hand pulse & tracking
      if (name === "right_wrist") {
        rightWristPos3D = pos.clone();
        if (rightWristPulseSphere) {
          rightWristPulseSphere.position.copy(pos);
          rightWristPulseSphere.visible = true;
          const s = 1.0 + Math.cos(Date.now() * 0.009) * 0.25;
          rightWristPulseSphere.scale.set(s, s, s);
        }

        // Active trajectory
        wristHistory.push(pos.clone());
        if (wristHistory.length > 25) wristHistory.shift();

        if (wristTrajectoryLine) {
          const pts = [];
          wristHistory.forEach(p => pts.push(p.x, p.y, p.z));
          wristTrajectoryLine.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
        }
      }

      // Left Knee pulse (Microgravity Foot Restraint anchor)
      if (name === "left_knee" && leftKneePulseSphere) {
        leftKneePulseSphere.position.copy(pos);
        leftKneePulseSphere.visible = true;
        const sk = 0.9 + Math.sin(Date.now() * 0.006) * 0.15;
        leftKneePulseSphere.scale.set(sk, sk, sk);
      }

      // Right Knee pulse (Microgravity Foot Restraint anchor)
      if (name === "right_knee" && rightKneePulseSphere) {
        rightKneePulseSphere.position.copy(pos);
        rightKneePulseSphere.visible = true;
        const sk = 0.9 + Math.cos(Date.now() * 0.006) * 0.15;
        rightKneePulseSphere.scale.set(sk, sk, sk);
      }
    } else {
      mesh.visible = false;
      if (name === "left_knee" && leftKneePulseSphere) leftKneePulseSphere.visible = false;
      if (name === "right_knee" && rightKneePulseSphere) rightKneePulseSphere.visible = false;
    }
  });

  // Update Bones
  POSE_CONNECTIONS.forEach(([nameA, nameB], idx) => {
    const bone = boneMeshes[idx];
    if (kps[nameA] && kps[nameB] && bone) {
      const pA = mapTo3D(kps[nameA]);
      const pB = mapTo3D(kps[nameB]);
      const dist = pA.distanceTo(pB);
      if (dist > 0.01) {
        bone.position.copy(pA).add(pB).multiplyScalar(0.5);
        bone.scale.set(1, dist, 1);
        const dir = pB.clone().sub(pA).normalize();
        if (dir.lengthSq() > 0.5) {
          bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
          bone.visible = true;
        }
      } else {
        bone.visible = false;
      }
    } else if (bone) {
      bone.visible = false;
    }
  });

  // DYNAMIC 3D OBJECT-IN-HAND ATTACHMENT
  // If an object is picked up, attach the 3D model to that grasping hand!
  const held = pose.held_payload;
  if (held && held.object_name) {
    const targetObjMesh = targetMeshes[held.object_name];
    if (targetObjMesh) {
      const graspingPos = held.hand === "LEFT" ? leftWristPos3D : rightWristPos3D;
      if (graspingPos) {
        // Move object along with astronaut's hand in 3D space
        targetObjMesh.position.set(graspingPos.x, graspingPos.y, graspingPos.z - 0.05);
      }
    }
  } else {
    // Reset resting positions if released
    Object.keys(defaultTargetPositions).forEach(key => {
      if (targetMeshes[key]) {
        targetMeshes[key].position.copy(defaultTargetPositions[key]);
      }
    });
  }
}

function onWindowResize() {
  const container = document.getElementById("three-container");
  if (!container || !renderer || !camera) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function animate3D() {
  requestAnimationFrame(animate3D);
  if (controls) controls.update();
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

// Camera View Presets
function setCameraView(mode) {
  document.querySelectorAll(".view-btn").forEach(b => b.classList.remove("active"));
  event?.target?.classList?.add("active");

  if (!camera || !controls) return;

  if (mode === "front") {
    camera.position.set(0, 0, 2.8);
    controls.target.set(0, 0, 0);
  } else if (mode === "top") {
    camera.position.set(0, 2.9, 0.2);
    controls.target.set(0, 0, -0.2);
  } else if (mode === "side") {
    camera.position.set(2.8, 0, 0.1);
    controls.target.set(0, 0, -0.2);
  } else if (mode === "orbit") {
    camera.position.set(1.2, 1.0, 2.2);
    controls.target.set(0, 0, 0);
  }
  controls.update();
}

function reset3DCamera() {
  if (camera && controls) {
    camera.position.set(0, 0.2, 2.8);
    controls.target.set(0, 0, 0);
    controls.update();
  }
}

// ==============================================================================
// 2. MISSION TELEMETRY & DUAL-HAND WEBSOCKET ENGINE
// ==============================================================================

function playAlertChime() {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.setValueAtTime(440, audioCtx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.4);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.45);
  } catch (e) {
    console.log("Audio chime:", e);
  }
}

function updateMissionClock() {
  const now = new Date();
  const utcStr = now.toISOString().substring(11, 19) + " UTC";
  const el = document.getElementById("mission-clock");
  if (el) el.textContent = utcStr;
}
setInterval(updateMissionClock, 1000);
updateMissionClock();

async function loadProtocolsList() {
  try {
    const res = await fetch("/api/v1/protocols");
    const data = await res.json();
    const select = document.getElementById("scenario-select");
    if (!select || !data.protocols) return;

    select.innerHTML = "";
    data.protocols.forEach((p, idx) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      const icon = p.category.includes("BIO") ? "🔬" : p.category.includes("FLUID") ? "💧" : p.category.includes("AVIONICS") ? "⚡" : "🚨";
      opt.textContent = `${icon} Scenario ${idx + 1}: ${p.experiment_name}`;
      select.appendChild(opt);
    });

    if (activeProtocol) {
      select.value = activeProtocol.protocol_id;
    }
  } catch (err) {
    console.error("Failed to load protocols list:", err);
  }
}

const PROTOCOL_CAMERA_MAP = {
  "BAS-EXP-BIO-2026": {
    camera_tag: "CAM-01 • GLOVEBOX BIO-RACK",
    camera_channel: "CAM-MSG-01-A",
    facility: "Bharatiya Antariksh Station (BAS) - Microgravity Science Glovebox",
    location: "Microgravity Science Glovebox Bay"
  },
  "BAS-EXP-FLUID-2026": {
    camera_tag: "CAM-02 • FLUID BENCH PORT-A",
    camera_channel: "CAM-FPB-02-B",
    facility: "Bharatiya Antariksh Station (BAS) - Glovebox Fluid Physics Bench",
    location: "Fluid Physics Bench Workstation"
  },
  "BAS-EXP-AVIONICS-2026": {
    camera_tag: "CAM-03 • AVIONICS BAY 3 RACK",
    camera_channel: "CAM-AVN-03-C",
    facility: "Bharatiya Antariksh Station (BAS) - Avionics Bay 3 Rack",
    location: "Avionics Bay 3 Component Rack"
  },
  "BAS-EMERGENCY-01": {
    camera_tag: "CAM-04 • AIRLOCK FORWARD HATCH",
    camera_channel: "CAM-EMG-04-D",
    facility: "Bharatiya Antariksh Station (BAS) - Node 1 Forward Airlock Tunnel",
    location: "Node 1 Forward Airlock Bulkhead"
  }
};

function updateScenarioCameraTags(protocolOrData) {
  if (!protocolOrData) return;
  const protoId = protocolOrData.protocol_id || (activeProtocol && activeProtocol.protocol_id);
  const fallback = PROTOCOL_CAMERA_MAP[protoId] || PROTOCOL_CAMERA_MAP["BAS-EXP-BIO-2026"];

  const camTag = protocolOrData.camera_tag || fallback.camera_tag;
  const camChannel = protocolOrData.camera_channel || fallback.camera_channel;

  // 1. PIP Camera Box Window Header
  const pipTitle = document.getElementById("pip-camera-title");
  if (pipTitle) {
    pipTitle.textContent = `● ${camTag.replace(/^●\s*/, '')}`;
  }

  // 2. 3D Viewport Header Live Camera Tag
  const streamBadge = document.getElementById("camera-tag-stream-badge");
  if (streamBadge) {
    streamBadge.textContent = `● ${camTag.replace(/^●\s*/, '')}`;
  }

  // 3. Top Header Camera Channel
  const chanBadge = document.getElementById("cam-channel-badge");
  if (chanBadge) {
    chanBadge.textContent = camChannel;
  }

  // 4. Enlarged Camera Tag & Theater Subtitle
  const ecamTag = document.getElementById("enlarged-cam-tag");
  if (ecamTag) {
    ecamTag.textContent = `${camTag.replace(/^●\s*/, '')} [${camChannel}]`;
  }
  const thSub = document.getElementById("theater-cam-sub");
  if (thSub) {
    thSub.textContent = `${camTag.replace(/^●\s*/, '')} [${camChannel}] | Real-Time Kinematics & Dual-Hand HUD`;
  }
}

async function loadProtocol() {
  try {
    const res = await fetch("/api/v1/protocol");
    activeProtocol = await res.json();
    document.getElementById("protocol-id-badge").textContent = activeProtocol.protocol_id;
    document.getElementById("experiment-name").textContent = activeProtocol.experiment_name;
    const facEl = document.getElementById("experiment-facility");
    if (facEl && activeProtocol.facility) facEl.textContent = activeProtocol.facility;
    const selEl = document.getElementById("scenario-select");
    if (selEl) selEl.value = activeProtocol.protocol_id;
    updateScenarioCameraTags(activeProtocol);
    renderProtocolSteps();
    create3DPayloadTargets(activeProtocol);
  } catch (err) {
    console.error("Failed to load protocol:", err);
  }
}

async function switchProtocol(protocolId) {
  try {
    addTerminalLog(`Uplinking command: Switch scenario to [${protocolId}]...`, "sys");
    const res = await fetch(`/api/v1/protocol/select/${protocolId}`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    activeProtocol = data.protocol;
    currentStepIdx = 1;
    deviationsCount = 0;
    document.getElementById("counter-deviations").textContent = `0 DEVIATIONS`;
    hideDeviationBanner();
    document.getElementById("protocol-id-badge").textContent = activeProtocol.protocol_id;
    document.getElementById("experiment-name").textContent = activeProtocol.experiment_name;
    const facEl = document.getElementById("experiment-facility");
    if (facEl && activeProtocol.facility) facEl.textContent = activeProtocol.facility;
    const selEl = document.getElementById("scenario-select");
    if (selEl) selEl.value = activeProtocol.protocol_id;
    updateScenarioCameraTags(activeProtocol);
    renderProtocolSteps();
    create3DPayloadTargets(activeProtocol);
    addTerminalLog(`[SCENARIO ACTIVE] Switched to: ${activeProtocol.experiment_name} (${activeProtocol.protocol_id})`, "step");
  } catch (err) {
    console.error("Failed to switch scenario:", err);
    addTerminalLog(`[ERROR] Scenario switch failed: ${err.message}`, "alert");
  }
}

function renderProtocolSteps() {
  const container = document.getElementById("steps-container");
  if (!container || !activeProtocol) return;

  container.innerHTML = "";
  activeProtocol.steps.forEach((step) => {
    const isPast = step.step_id < currentStepIdx;
    const isCurrent = step.step_id === currentStepIdx;

    let statusClass = "";
    let badgeContent = step.step_id;
    if (isPast) {
      statusClass = "verified";
      badgeContent = "✓";
    } else if (isCurrent) {
      statusClass = "active";
      badgeContent = "⟳";
    }

    const critClass = step.criticality === "CRITICAL" ? "crit-critical" : "crit-high";

    const card = document.createElement("div");
    card.className = `step-card ${statusClass}`;
    card.id = `step-card-${step.step_id}`;
    card.innerHTML = `
      <div class="step-badge">${badgeContent}</div>
      <div class="step-details">
        <div class="step-title-row">
          <span class="step-title font-mono">${step.expected_action} ${step.expected_target}</span>
          <span class="step-criticality ${critClass}">${step.criticality}</span>
        </div>
        <div class="step-desc">${step.description}</div>
      </div>
    `;
    container.appendChild(card);
  });

  const currentStep = activeProtocol.steps.find(s => s.step_id === currentStepIdx);
  const expEl = document.getElementById("expected-action-display");
  if (expEl) {
    expEl.textContent = currentStep ? `${currentStep.expected_action} ${currentStep.expected_target}` : "EXPERIMENT COMPLETE";
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    addTerminalLog("Telemetry downlink synchronized with Bharatiya Antariksh Station.", "sys");
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleTelemetryMessage(msg);
    } catch (e) {
      console.error("Malformed WebSocket packet:", e);
    }
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 3000);
  };
}

function handleTelemetryMessage(msg) {
  if (msg.type === "CAMERA_OFFLINE") {
    const feed = document.getElementById("live-camera-feed");
    const offMsg = document.getElementById("camera-offline-msg");
    if (feed) feed.style.display = "none";
    if (offMsg) offMsg.style.display = "flex";
    addTerminalLog("[CAMERA FEED] Edge camera monitoring program closed.", "sys");
    return;
  }

  if (msg.type === "PROTOCOL_SWITCH") {
    activeProtocol = msg.protocol;
    currentStepIdx = 1;
    deviationsCount = 0;
    document.getElementById("counter-deviations").textContent = `0 DEVIATIONS`;
    hideDeviationBanner();
    document.getElementById("protocol-id-badge").textContent = activeProtocol.protocol_id;
    document.getElementById("experiment-name").textContent = activeProtocol.experiment_name;
    const facEl = document.getElementById("experiment-facility");
    if (facEl && activeProtocol.facility) facEl.textContent = activeProtocol.facility;
    const selEl = document.getElementById("scenario-select");
    if (selEl) selEl.value = activeProtocol.protocol_id;
    updateScenarioCameraTags(activeProtocol || msg.state);
    renderProtocolSteps();
    create3DPayloadTargets(activeProtocol);
    addTerminalLog(`[UPLINK BROADCAST] Mission scenario switched: ${activeProtocol.experiment_name}`, "sys");
    return;
  }

  if (msg.type === "INIT" || msg.type === "RESET") {
    currentStepIdx = msg.state.current_step;
    deviationsCount = msg.state.deviations_count || 0;
    document.getElementById("counter-deviations").textContent = `${deviationsCount} DEVIATIONS`;
    hideDeviationBanner();
    if (msg.state) updateScenarioCameraTags(msg.state);
    renderProtocolSteps();

    updateDualHandUI(msg.state);
    updateLegsUI(msg.state);
    updateInHandPayloadCard(msg.state.held_payload);

    if (msg.state.astronaut_pose) {
      lastPose = msg.state.astronaut_pose;
      updateKinematicsUI(lastPose);
      update3DSkeleton(lastPose);
      const legV = ((msg.state.left_leg && msg.state.left_leg.velocity) || 0) + ((msg.state.right_leg && msg.state.right_leg.velocity) || 0);
      pushMotionData(lastPose.wrist_velocity || 0, lastPose.body_roll_degrees || 0, legV);
    }
    return;
  }

  if (msg.type === "HEARTBEAT") {
    const data = msg.data;
    if (data && (data.camera_tag || data.camera_channel)) {
      updateScenarioCameraTags(data);
    }
    currentStepIdx = data.current_step;
    renderProtocolSteps();

    const detEl = document.getElementById("detected-action-display");
    if (detEl) detEl.textContent = data.active_action || "MONITORING";

    updateDualHandUI(data);
    updateLegsUI(data);
    updateInHandPayloadCard(data.held_payload);

    // Sync Theater HUD
    const thAct = document.getElementById("th-action-val");
    if (thAct) thAct.textContent = data.active_action || "STANDBY";
    const thHand = document.getElementById("th-hand-val");
    if (thHand) thHand.textContent = data.active_hand || "DUAL-READY";
    const thStep = document.getElementById("th-step-val");
    if (thStep) thStep.textContent = `STEP ${data.current_step || 1}`;
    const thPayload = document.getElementById("th-payload-val");
    if (thPayload) thPayload.textContent = data.held_payload || "NONE";

    if (data.astronaut_pose) {
      lastPose = data.astronaut_pose;
      // Pass held payload & legs to 3D skeleton
      lastPose.held_payload = data.held_payload;
      lastPose.left_leg = data.left_leg;
      lastPose.right_leg = data.right_leg;
      lastPose.lower_body_activity = data.lower_body_activity;
      updateKinematicsUI(lastPose);
      update3DSkeleton(lastPose);

      const legV = ((data.left_leg && data.left_leg.velocity) || 0) + ((data.right_leg && data.right_leg.velocity) || 0);
      pushMotionData(lastPose.wrist_velocity || 0, lastPose.body_roll_degrees || 0, legV);
    }

    if (data.status === "STEP_VERIFIED") {
      addTerminalLog(`[STEP VERIFIED] Step ${data.current_step}: ${data.active_action} (${data.active_hand || 'HAND'})`, "step");
      updateComplianceScore();
    } else if (data.status === "EXPERIMENT_SUCCESSFUL") {
      addTerminalLog(`[MISSION SUCCESS] Protocol complete. 100% compliance verified across all steps!`, "step");
      document.getElementById("compliance-pct").textContent = "100%";
    }
  }

  if (msg.type === "DEVIATION") {
    const alert = msg.data;
    deviationsCount++;
    document.getElementById("counter-deviations").textContent = `${deviationsCount} DEVIATIONS`;
    showDeviationBanner(alert);
    playAlertChime();

    addTerminalLog(`🚨 [DEVIATION] ${alert.deviation_type}: ${alert.alert_message}`, "alert");

    const card = document.getElementById(`step-card-${alert.current_step}`);
    if (card) {
      card.classList.add("deviated");
      const badge = card.querySelector(".step-badge");
      if (badge) badge.textContent = "⚠";
    }

    if (alert.astronaut_pose) {
      lastPose = alert.astronaut_pose;
      lastPose.held_payload = alert.held_payload;
      updateKinematicsUI(lastPose);
      update3DSkeleton(lastPose);
      updateLegsUI(alert);
      const legV = ((alert.left_leg && alert.left_leg.velocity) || 0) + ((alert.right_leg && alert.right_leg.velocity) || 0);
      pushMotionData(lastPose.wrist_velocity || 0, lastPose.body_roll_degrees || 0, legV);
    }

    updateComplianceScore();
  }
}

// Update Left Hand & Right Hand UI Cards
function updateDualHandUI(stateOrData) {
  const lh = stateOrData.left_hand;
  const rh = stateOrData.right_hand;

  if (lh) {
    const lhCard = document.getElementById("left-hand-card");
    const lhTag = document.getElementById("left-grip-tag");
    const lhObj = document.getElementById("left-hand-obj");
    const lhVel = document.getElementById("left-hand-vel");
    const lhCoords = document.getElementById("left-hand-coords");

    if (lh.held_object) {
      lhCard?.classList.add("holding");
      if (lhTag) lhTag.textContent = "HOLDING";
      if (lhObj) lhObj.textContent = lh.held_object;
    } else {
      lhCard?.classList.remove("holding");
      if (lhTag) lhTag.textContent = lh.detected ? "EMPTY" : "STANDBY";
      if (lhObj) lhObj.textContent = lh.detected ? "HAND EMPTY" : "NOT DETECTED";
    }

    if (lhVel) lhVel.textContent = `${(lh.velocity || 0).toFixed(2)} m/s`;
    if (lhCoords && lh.coords) {
      lhCoords.textContent = `[${lh.coords[0].toFixed(2)}, ${lh.coords[1].toFixed(2)}, ${lh.coords[2].toFixed(2)}]`;
    }
  }

  if (rh) {
    const rhCard = document.getElementById("right-hand-card");
    const rhTag = document.getElementById("right-grip-tag");
    const rhObj = document.getElementById("right-hand-obj");
    const rhVel = document.getElementById("right-hand-vel");
    const rhCoords = document.getElementById("right-hand-coords");

    if (rh.held_object) {
      rhCard?.classList.add("holding");
      if (rhTag) rhTag.textContent = "HOLDING";
      if (rhObj) rhObj.textContent = rh.held_object;
    } else {
      rhCard?.classList.remove("holding");
      if (rhTag) rhTag.textContent = rh.detected ? "EMPTY" : "STANDBY";
      if (rhObj) rhObj.textContent = rh.detected ? "HAND EMPTY" : "NOT DETECTED";
    }

    if (rhVel) rhVel.textContent = `${(rh.velocity || 0).toFixed(2)} m/s`;
    if (rhCoords && rh.coords) {
      rhCoords.textContent = `[${rh.coords[0].toFixed(2)}, ${rh.coords[1].toFixed(2)}, ${rh.coords[2].toFixed(2)}]`;
    }
  }
}

// ==============================================================================
// LOWER BODY & LEGS TELEMETRY
// ==============================================================================
function updateLegsUI(stateOrData) {
  if (!stateOrData) return;
  const ll = stateOrData.left_leg;
  const rl = stateOrData.right_leg;
  const lowerBodyAct = stateOrData.lower_body_activity || "ANCHORED IN FOOT RESTRAINT";

  // Update 3D Overlay Lower Body Readout
  const ovLb = document.getElementById("overlay-lower-body");
  if (ovLb) {
    ovLb.textContent = lowerBodyAct;
    if (lowerBodyAct.includes("FLEXION") || lowerBodyAct.includes("BENT")) {
      ovLb.className = "font-mono text-amber";
    } else if (lowerBodyAct.includes("FLOATING")) {
      ovLb.className = "font-mono text-cyan";
    } else {
      ovLb.className = "font-mono text-green";
    }
  }

  // Left Leg Card
  if (ll) {
    const lCard = document.getElementById("left-leg-card");
    const lTag = document.getElementById("left-leg-tag");
    const lAct = document.getElementById("left-leg-act");
    const lAngle = document.getElementById("left-knee-angle");
    const lVel = document.getElementById("left-leg-vel");

    const lAngleVal = ll.knee_angle_deg !== undefined ? ll.knee_angle_deg : (ll.knee_angle !== undefined ? ll.knee_angle : 180);
    if (lTag) lTag.textContent = ll.activity || (ll.detected ? "ANCHORED" : "STANDBY");
    if (lAct) lAct.textContent = (ll.activity === "KNEE FLEXION") ? "KNEE FLEXED (BENT)" : ((ll.activity === "MICROGRAVITY FLOATING") ? "ZERO-G FLOAT" : "RESTRAINT NOMINAL");
    if (lAngle) lAngle.textContent = `${Math.round(lAngleVal)}°`;
    if (lVel) lVel.textContent = `${(ll.velocity || 0).toFixed(2)} m/s`;

    if (lCard) {
      lCard.classList.toggle("flexed", ll.activity === "KNEE FLEXION");
      lCard.classList.toggle("floating", ll.activity === "MICROGRAVITY FLOATING");
    }
  }

  // Right Leg Card
  if (rl) {
    const rCard = document.getElementById("right-leg-card");
    const rTag = document.getElementById("right-leg-tag");
    const rAct = document.getElementById("right-leg-act");
    const rAngle = document.getElementById("right-knee-angle");
    const rVel = document.getElementById("right-leg-vel");

    const rAngleVal = rl.knee_angle_deg !== undefined ? rl.knee_angle_deg : (rl.knee_angle !== undefined ? rl.knee_angle : 180);
    if (rTag) rTag.textContent = rl.activity || (rl.detected ? "ANCHORED" : "STANDBY");
    if (rAct) rAct.textContent = (rl.activity === "KNEE FLEXION") ? "KNEE FLEXED (BENT)" : ((rl.activity === "MICROGRAVITY FLOATING") ? "ZERO-G FLOAT" : "RESTRAINT NOMINAL");
    if (rAngle) rAngle.textContent = `${Math.round(rAngleVal)}°`;
    if (rVel) rVel.textContent = `${(rl.velocity || 0).toFixed(2)} m/s`;

    if (rCard) {
      rCard.classList.toggle("flexed", rl.activity === "KNEE FLEXION");
      rCard.classList.toggle("floating", rl.activity === "MICROGRAVITY FLOATING");
    }
  }
}

// ==============================================================================
// REAL-TIME MOTION OSCILLOSCOPE GRAPH (MATCHED TO CAMERA FPS)
// ==============================================================================
const MOTION_WAVEFORM_CAPACITY = 80;
const motionHistory = [];
for (let i = 0; i < MOTION_WAVEFORM_CAPACITY; i++) {
  motionHistory.push({ vel: 0, roll: 0, legVel: 0 });
}

function pushMotionData(vel, roll, legVel) {
  motionHistory.push({
    vel: Math.max(0, Math.min(2.0, vel || 0)),
    roll: Math.max(-90, Math.min(90, roll || 0)),
    legVel: Math.max(0, Math.min(2.0, legVel || 0))
  });
  if (motionHistory.length > MOTION_WAVEFORM_CAPACITY) {
    motionHistory.shift();
  }
  renderMotionWaveform();
}

function renderMotionWaveform() {
  const canvas = document.getElementById("kinematics-waveform-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Background Grid Lines
  ctx.strokeStyle = "rgba(0, 229, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h * 0.25); ctx.lineTo(w, h * 0.25);
  ctx.moveTo(0, h * 0.50); ctx.lineTo(w, h * 0.50);
  ctx.moveTo(0, h * 0.75); ctx.lineTo(w, h * 0.75);
  ctx.stroke();

  const stepX = w / (MOTION_WAVEFORM_CAPACITY - 1);

  // 1. Draw Leg Motion (Green)
  ctx.strokeStyle = "#00e676";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < motionHistory.length; i++) {
    const x = i * stepX;
    const y = h - 5 - (motionHistory[i].legVel / 1.5) * (h - 12);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // 2. Draw Body Roll (Amber)
  ctx.strokeStyle = "#ff9100";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < motionHistory.length; i++) {
    const x = i * stepX;
    const normalizedRoll = (motionHistory[i].roll + 90) / 180;
    const y = h - 5 - normalizedRoll * (h - 12);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // 3. Draw Wrist Speed (Cyan, prominent)
  ctx.strokeStyle = "#00e5ff";
  ctx.lineWidth = 2.0;
  ctx.shadowColor = "#00e5ff";
  ctx.shadowBlur = 4;
  ctx.beginPath();
  for (let i = 0; i < motionHistory.length; i++) {
    const x = i * stepX;
    const y = h - 5 - (motionHistory[i].vel / 1.5) * (h - 12);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

// Update In-Hand Active Payload Telemetry Card
function updateInHandPayloadCard(heldPayload) {
  const card = document.getElementById("inhand-payload-card");
  if (!card) return;

  if (!heldPayload || !heldPayload.object_name) {
    card.classList.add("hidden");
    return;
  }

  card.classList.remove("hidden");
  document.getElementById("inhand-obj-name").textContent = heldPayload.object_name.replace(/_/g, " ");

  const badge = document.getElementById("inhand-actor-badge");
  if (badge) {
    badge.textContent = `${heldPayload.hand} HAND`;
    badge.className = `inhand-hand-badge ${heldPayload.hand === 'LEFT' ? 'badge-l' : 'badge-r'}`;
  }

  const grid = document.getElementById("inhand-specs-grid");
  const specs = heldPayload.specs || {};

  grid.innerHTML = `
    <div class="spec-item">
      <span class="spec-key">GRASP CONFIDENCE</span>
      <span class="spec-val text-green">${heldPayload.grasp_confidence || 95}% (${heldPayload.contact_distance_m || 0.02}m)</span>
    </div>
    <div class="spec-item">
      <span class="spec-key">MICROG MASS</span>
      <span class="spec-val">${specs.microg_mass || '320 g (Nominal Zero-G)'}</span>
    </div>
    <div class="spec-item">
      <span class="spec-key">HAZARD LEVEL</span>
      <span class="spec-val text-amber">${specs.hazard_level || 'Biosafety Level 2 (Hermetic)'}</span>
    </div>
    <div class="spec-item">
      <span class="spec-key">TARGET DESTINATION</span>
      <span class="spec-val">${specs.destination_bay || 'Payload Slot 1'}</span>
    </div>
    <div class="spec-item">
      <span class="spec-key">ALIGNMENT PIN</span>
      <span class="spec-val">${specs.alignment_pin || '90° Radial Offset'}</span>
    </div>
    <div class="spec-item">
      <span class="spec-key">HANDLING DIRECTIVE</span>
      <span class="spec-val text-cyan">${specs.handling || 'Ambidextrous grasp permitted.'}</span>
    </div>
  `;
}

function updateKinematicsUI(pose) {
  const roll = pose.body_roll_degrees || 0.0;
  const vel = pose.wrist_velocity || 0.0;

  document.getElementById("overlay-roll").textContent = `${roll.toFixed(1)}°`;
  document.getElementById("overlay-vel").textContent = `${vel.toFixed(2)} m/s`;
  document.getElementById("val-roll").textContent = `${roll.toFixed(1)}°`;
  document.getElementById("val-vel").textContent = `${vel.toFixed(2)} m/s`;

  if (pose.keypoints && (pose.keypoints.right_wrist || pose.keypoints.left_wrist)) {
    const w = pose.keypoints.right_wrist || pose.keypoints.left_wrist;
    const x = w.rack_x !== undefined ? w.rack_x.toFixed(2) : w.x.toFixed(2);
    const y = w.rack_y !== undefined ? w.rack_y.toFixed(2) : w.y.toFixed(2);
    const z = w.rack_z !== undefined ? w.rack_z.toFixed(2) : (w.z || 0).toFixed(2);
    document.getElementById("overlay-wrist-3d").textContent = `[${x}, ${y}, ${z}]`;
  }

  const rollPct = Math.min(100, Math.max(0, ((roll + 90) / 180) * 100));
  const velPct = Math.min(100, (vel / 1.0) * 100);
  document.getElementById("bar-roll").style.width = `${rollPct}%`;
  document.getElementById("bar-vel").style.width = `${velPct}%`;
}

function updateComplianceScore() {
  if (deviationsCount === 0) {
    document.getElementById("compliance-pct").textContent = "100%";
    document.getElementById("compliance-pct").className = "stat-val text-green";
  } else {
    const penalty = Math.min(80, deviationsCount * 15);
    const score = 100 - penalty;
    document.getElementById("compliance-pct").textContent = `${score}%`;
    document.getElementById("compliance-pct").className = "stat-val text-amber";
  }
}

function showDeviationBanner(alert) {
  const banner = document.getElementById("incident-banner");
  banner.classList.remove("hidden");
  document.getElementById("alert-type").textContent = alert.deviation_type || "DEVIATION DETECTED";
  document.getElementById("alert-step-id").textContent = `STEP ${alert.current_step}`;
  document.getElementById("alert-message").textContent = alert.alert_message;
}

function hideDeviationBanner() {
  const banner = document.getElementById("incident-banner");
  banner.classList.add("hidden");
}

document.getElementById("alert-ack-btn")?.addEventListener("click", hideDeviationBanner);

function addTerminalLog(text, type = "sys") {
  const container = document.getElementById("terminal-logs");
  if (!container) return;

  const now = new Date().toISOString().substring(11, 19);
  const entry = document.createElement("div");
  entry.className = `log-entry ${type === "alert" ? "log-alert" : type === "step" ? "log-step" : "log-heartbeat"}`;
  entry.textContent = `[${now}] ${text}`;
  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;
}

// Load Recorded Sessions List
async function loadRecordingsList() {
  const listContainer = document.getElementById("recordings-list");
  if (!listContainer) return;

  try {
    const res = await fetch("/api/v1/recordings");
    const data = await res.json();
    if (!data.recordings || data.recordings.length === 0) {
      listContainer.innerHTML = `<div class="rec-placeholder">No saved sessions yet. Edge camera creates recordings automatically.</div>`;
      return;
    }

    listContainer.innerHTML = "";
    data.recordings.forEach(rec => {
      const item = document.createElement("div");
      item.className = "rec-item";
      item.innerHTML = `
        <a href="/api/v1/recordings/${rec.filename}" target="_blank">🎬 ${rec.filename}</a>
        <span>${rec.size_mb} MB</span>
      `;
      listContainer.appendChild(item);
    });
  } catch (err) {
    console.error("Failed to load recordings:", err);
  }
}

// Simulations
async function runScenario(scenarioKey) {
  const label = document.getElementById("sim-status-label");
  if (label) label.textContent = `Running ${scenarioKey}...`;

  try {
    const res = await fetch(`/api/v1/simulate/${scenarioKey}`, { method: "POST" });
    const data = await res.json();
    addTerminalLog(`Simulation launched: Scenario ${scenarioKey.toUpperCase()} (PID ${data.pid})`, "sys");
    setTimeout(() => {
      if (label) label.textContent = "Standby";
      loadRecordingsList();
    }, 4000);
  } catch (err) {
    console.error("Failed to run scenario:", err);
    if (label) label.textContent = "Error launching sim";
  }
}

async function resetTelemetryState() {
  try {
    await fetch("/api/v1/telemetry/reset", { method: "POST" });
    addTerminalLog("Telemetry state reset to initial baseline.", "sys");
  } catch (err) {
    console.error("Reset failed:", err);
  }
}

// Window Load Init
window.addEventListener("DOMContentLoaded", () => {
  init3DScene();
  loadProtocol();
  loadProtocolsList();
  connectWebSocket();
  loadRecordingsList();

  setInterval(loadRecordingsList, 10000);
});

let standaloneCameraWindow = null;

function openStandaloneCamera() {
  if (standaloneCameraWindow && !standaloneCameraWindow.closed) {
    standaloneCameraWindow.focus();
    return standaloneCameraWindow;
  }
  standaloneCameraWindow = window.open(
    '/camera',
    'AstroActAiDedicatedCamera',
    'width=1280,height=760,menubar=no,toolbar=no,location=no,status=no,resizable=yes'
  );
  return standaloneCameraWindow;
}

function closeStandaloneCamera() {
  if (standaloneCameraWindow && !standaloneCameraWindow.closed) {
    try {
      standaloneCameraWindow.close();
    } catch (e) {
      console.warn("Could not close camera window:", e);
    }
    standaloneCameraWindow = null;
  }
}

// Automatically close camera window when dashboard/program window is closed or unloaded
window.addEventListener("beforeunload", () => {
  closeStandaloneCamera();
});

window.addEventListener("unload", () => {
  closeStandaloneCamera();
});

window.addEventListener("pagehide", () => {
  closeStandaloneCamera();
});


