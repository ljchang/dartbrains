// Net Magnetization Widget - Three.js 3D
// Shows ensemble of proton spins aligning when B0 is toggled on

const THREE_URL = "https://esm.sh/three@0.170.0";
const ORBIT_URL = "https://esm.sh/three@0.170.0/addons/controls/OrbitControls.js";

export default {
  async render({ model, el }) {
    const THREE = await import(THREE_URL);
    const { OrbitControls } = await import(ORBIT_URL);

    const WIDTH = 600;
    const HEIGHT = 480;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(WIDTH, HEIGHT);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8f9fa);

    const camera = new THREE.PerspectiveCamera(40, WIDTH / HEIGHT, 0.1, 100);
    camera.position.set(2.5, 2.0, 2.5);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dl = new THREE.DirectionalLight(0xffffff, 0.6);
    dl.position.set(3, 5, 3);
    scene.add(dl);

    // Axes
    function addAxis(from, to, color) {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...from), new THREE.Vector3(...to)
      ]);
      scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color })));
    }
    addAxis([-1.3, 0, 0], [1.3, 0, 0], 0xcc4444);
    addAxis([0, -1.3, 0], [0, 1.3, 0], 0x44aa44);
    addAxis([0, 0, -1.3], [0, 0, 1.3], 0x4444cc);

    // B0 label
    function makeLabel(text, pos, color) {
      const c = document.createElement("canvas");
      c.width = 128; c.height = 64;
      const x = c.getContext("2d");
      x.font = "bold 32px Arial";
      x.fillStyle = color;
      x.textAlign = "center";
      x.fillText(text, 64, 42);
      const t = new THREE.CanvasTexture(c);
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true }));
      s.position.set(...pos);
      s.scale.set(0.4, 0.2, 1);
      return s;
    }
    scene.add(makeLabel("z (B\u2080)", [0, 0, 1.45], "#4444cc"));

    // --- Spins ---
    const nSpins = model.get("n_protons");
    const spins = [];
    const rng = mulberry32(42); // deterministic RNG

    function mulberry32(a) {
      return function() {
        let t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      };
    }

    // Create spin arrows as small cones
    const spinGeo = new THREE.ConeGeometry(0.015, 0.12, 6);
    const spinShaftGeo = new THREE.CylinderGeometry(0.005, 0.005, 0.12, 4);

    for (let i = 0; i < nSpins; i++) {
      // Random initial direction on unit sphere
      const theta = Math.acos(2 * rng() - 1);
      const phi = rng() * Math.PI * 2;

      const group = new THREE.Group();
      const hue = (i / nSpins);
      const color = new THREE.Color().setHSL(hue, 0.5, 0.6);

      const shaft = new THREE.Mesh(spinShaftGeo, new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 0.6 }));
      shaft.position.y = 0.06;
      group.add(shaft);
      const tip = new THREE.Mesh(spinGeo, new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 0.7 }));
      tip.position.y = 0.14;
      group.add(tip);

      // Scale down and place at origin
      group.scale.set(2, 2, 2);
      scene.add(group);

      spins.push({
        group,
        // Current direction (spherical -> will lerp)
        theta,  // polar angle from z-axis
        phi,    // azimuthal angle
        // Target direction
        targetTheta: theta,
        targetPhi: phi,
        // Random jitter
        jitterSpeed: 0.5 + rng() * 1.5,
        jitterPhase: rng() * Math.PI * 2,
      });
    }

    // Net magnetization arrow
    const netGroup = new THREE.Group();
    const netShaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.025, 0.025, 0.8, 12),
      new THREE.MeshLambertMaterial({ color: 0xdd2222 })
    );
    netShaft.position.y = 0.4;
    netGroup.add(netShaft);
    const netTip = new THREE.Mesh(
      new THREE.ConeGeometry(0.06, 0.12, 12),
      new THREE.MeshLambertMaterial({ color: 0xdd2222 })
    );
    netTip.position.y = 0.86;
    netGroup.add(netTip);
    scene.add(netGroup);

    // Info display
    const infoDiv = document.createElement("div");
    infoDiv.style.cssText = "text-align:center;font:13px Arial;color:#555;padding:4px;";
    el.appendChild(infoDiv);

    let lastTime = null;
    let animId;

    function animate(timestamp) {
      animId = requestAnimationFrame(animate);

      if (!lastTime) lastTime = timestamp;
      const dtMs = Math.min(timestamp - lastTime, 50);
      lastTime = timestamp;
      const dt = dtMs / 1000;

      const b0On = model.get("b0_on");
      const t = timestamp / 1000;

      // Update each spin's target and current direction
      let sumX = 0, sumY = 0, sumZ = 0;

      for (const s of spins) {
        if (b0On) {
          // Target: biased toward +z (B0 direction)
          // Each spin has ~60% chance of being "up" (theta < pi/2)
          // Smoothly lerp toward target
          const bias = 0.35 + 0.15 * Math.sin(t * s.jitterSpeed + s.jitterPhase);
          s.targetTheta = bias;  // lean toward z
          s.targetPhi += s.jitterSpeed * 0.3 * dt;  // slow azimuthal wander
        } else {
          // Random directions, wander freely
          s.targetTheta += (rng() - 0.5) * 2.0 * dt;
          s.targetTheta = Math.max(0.1, Math.min(Math.PI - 0.1, s.targetTheta));
          s.targetPhi += s.jitterSpeed * 0.8 * dt;
        }

        // Smooth lerp
        const lerpRate = b0On ? 2.0 : 1.0;
        s.theta += (s.targetTheta - s.theta) * lerpRate * dt;
        s.phi += (s.targetPhi - s.phi) * lerpRate * dt;

        // Add jitter
        const jitter = 0.05 * Math.sin(t * s.jitterSpeed * 3 + s.jitterPhase);

        const theta = s.theta + jitter;
        const phi = s.phi;

        // Convert spherical to cartesian (physics: z=up -> Three.js: y=up)
        const px = Math.sin(theta) * Math.cos(phi);
        const pz = Math.sin(theta) * Math.sin(phi);
        const py = Math.cos(theta);

        // Orient the arrow group
        const dir = new THREE.Vector3(px, py, pz).normalize();
        const up = new THREE.Vector3(0, 1, 0);
        s.group.quaternion.setFromUnitVectors(up, dir);

        sumX += px;
        sumY += py;
        sumZ += pz;
      }

      // Net magnetization
      const n = spins.length;
      const netMx = sumX / n;
      const netMy = sumY / n;
      const netMz = sumZ / n;
      const netMag = Math.sqrt(netMx * netMx + netMy * netMy + netMz * netMz);

      if (netMag > 0.01) {
        const netDir = new THREE.Vector3(netMx, netMy, netMz).normalize();
        const up = new THREE.Vector3(0, 1, 0);
        netGroup.quaternion.setFromUnitVectors(up, netDir);
        netGroup.scale.set(1, netMag * 2, 1);
        netGroup.visible = true;
      } else {
        netGroup.visible = false;
      }

      infoDiv.innerHTML = `<b>Net magnetization:</b> |M\u2080| = ${netMag.toFixed(3)} &nbsp;&nbsp; Mz = ${netMy.toFixed(3)} &nbsp;&nbsp; ` +
        `<span style="color:${b0On ? '#2266cc' : '#999'}">${b0On ? 'B\u2080 ON' : 'B\u2080 OFF'}</span>`;

      controls.update();
      renderer.render(scene, camera);
    }

    animId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animId);
      renderer.dispose();
      controls.dispose();
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) obj.material.dispose();
      });
    };
  },
};
