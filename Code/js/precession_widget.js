// Precession Widget - Three.js 3D magnetization vector animation
// Used in MR Physics Notebooks 1 & 2

const THREE_URL = "https://esm.sh/three@0.170.0";
const ORBIT_URL = "https://esm.sh/three@0.170.0/addons/controls/OrbitControls.js";

export default {
  async render({ model, el }) {
    const THREE = await import(THREE_URL);
    const { OrbitControls } = await import(ORBIT_URL);

    // --- Layout ---
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.gap = "12px";
    container.style.alignItems = "flex-start";
    container.style.justifyContent = "center";
    el.appendChild(container);

    const WIDTH = 500;
    const HEIGHT = 450;
    const SIG_WIDTH = 280;
    const SIG_HEIGHT = 450;

    // --- Three.js Scene ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8f9fa);

    const camera = new THREE.PerspectiveCamera(45, WIDTH / HEIGHT, 0.1, 100);
    camera.position.set(2.2, 1.5, 2.2);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(WIDTH, HEIGHT);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1.5;
    controls.maxDistance = 6;

    // --- Lighting ---
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(3, 5, 3);
    scene.add(dirLight);

    // --- Unit sphere wireframe ---
    const sphereGeo = new THREE.SphereGeometry(1, 24, 16);
    const sphereWire = new THREE.LineSegments(
      new THREE.WireframeGeometry(sphereGeo),
      new THREE.LineBasicMaterial({ color: 0xcccccc, opacity: 0.15, transparent: true })
    );
    scene.add(sphereWire);

    // --- Axis lines & labels ---
    function makeAxis(from, to, color) {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...from),
        new THREE.Vector3(...to),
      ]);
      return new THREE.Line(geo, new THREE.LineBasicMaterial({ color, linewidth: 1 }));
    }
    scene.add(makeAxis([-1.3, 0, 0], [1.3, 0, 0], 0xcc4444));
    scene.add(makeAxis([0, -1.3, 0], [0, 1.3, 0], 0x44aa44));
    scene.add(makeAxis([0, 0, -1.3], [0, 0, 1.3], 0x4444cc));

    function makeLabel(text, position, color) {
      const canvas = document.createElement("canvas");
      canvas.width = 128;
      canvas.height = 64;
      const ctx = canvas.getContext("2d");
      ctx.font = "bold 36px Arial";
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.fillText(text, 64, 44);
      const texture = new THREE.CanvasTexture(canvas);
      const mat = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(mat);
      sprite.position.set(...position);
      sprite.scale.set(0.4, 0.2, 1);
      return sprite;
    }
    scene.add(makeLabel("x", [1.45, 0, 0], "#cc4444"));
    scene.add(makeLabel("y", [0, 1.45, 0], "#44aa44"));
    scene.add(makeLabel("z (B\u2080)", [0, 0, 1.45], "#4444cc"));

    // --- Equator circle ---
    const eqPoints = [];
    for (let i = 0; i <= 64; i++) {
      const a = (i / 64) * Math.PI * 2;
      eqPoints.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0));
    }
    const eqGeo = new THREE.BufferGeometry().setFromPoints(eqPoints);
    scene.add(new THREE.Line(eqGeo, new THREE.LineBasicMaterial({ color: 0xaaaaaa, opacity: 0.3, transparent: true })));

    // --- Magnetization arrow ---
    const arrowGroup = new THREE.Group();
    const shaftGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.85, 12);
    const shaftMat = new THREE.MeshLambertMaterial({ color: 0xdd2222 });
    const shaft = new THREE.Mesh(shaftGeo, shaftMat);
    shaft.position.y = 0.425;
    arrowGroup.add(shaft);

    const tipGeo = new THREE.ConeGeometry(0.08, 0.15, 12);
    const tipMat = new THREE.MeshLambertMaterial({ color: 0xdd2222 });
    const tip = new THREE.Mesh(tipGeo, tipMat);
    tip.position.y = 0.925;
    arrowGroup.add(tip);

    scene.add(arrowGroup);

    // --- Trail ---
    const TRAIL_LEN = 300;
    const trailPositions = new Float32Array(TRAIL_LEN * 3);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
    const trailMat = new THREE.LineBasicMaterial({ color: 0xff8800, opacity: 0.5, transparent: true });
    const trail = new THREE.Line(trailGeo, trailMat);
    scene.add(trail);
    let trailIdx = 0;
    let trailCount = 0;

    // --- Signal panel (Canvas 2D) ---
    const sigCanvas = document.createElement("canvas");
    sigCanvas.width = SIG_WIDTH * 2;
    sigCanvas.height = SIG_HEIGHT * 2;
    sigCanvas.style.width = SIG_WIDTH + "px";
    sigCanvas.style.height = SIG_HEIGHT + "px";
    sigCanvas.style.borderRadius = "6px";
    sigCanvas.style.border = "1px solid #ddd";
    container.appendChild(sigCanvas);
    const sigCtx = sigCanvas.getContext("2d");
    sigCtx.scale(2, 2);

    // Oscilloscope buffer for Mx (coil signal)
    const SCOPE_LEN = 200;
    const mxScope = new Float32Array(SCOPE_LEN);
    let scopeIdx = 0;
    let scopeFrameCount = 0;

    // --- Animation state ---
    let phi = 0;
    let Mxy = 0;
    let Mz = 1;
    let lastTime = null;

    function resetState() {
      const flipDeg = model.get("flip_angle");
      const flipRad = flipDeg * Math.PI / 180;
      Mxy = Math.sin(flipRad);
      Mz = Math.cos(flipRad);
      phi = 0;
      trailIdx = 0;
      trailCount = 0;
      scopeIdx = 0;
      scopeFrameCount = 0;
      mxScope.fill(0);
      lastTime = null;
    }

    resetState();

    model.on("change:flip_angle", () => resetState());
    model.on("change:b0", () => {});
    model.on("change:show_relaxation", () => resetState());
    model.on("change:t1", () => {});
    model.on("change:t2", () => {});

    let animId;

    function animate(timestamp) {
      animId = requestAnimationFrame(animate);

      if (model.get("paused")) {
        lastTime = timestamp;
        controls.update();
        renderer.render(scene, camera);
        return;
      }

      if (!lastTime) lastTime = timestamp;
      const dtMs = Math.min(timestamp - lastTime, 50);
      lastTime = timestamp;
      const dtSec = dtMs / 1000;

      const b0 = model.get("b0");
      const showRelax = model.get("show_relaxation");
      const t1 = model.get("t1");
      const t2 = model.get("t2");

      const visualFreq = b0 * 0.4;
      phi += 2 * Math.PI * visualFreq * dtSec;

      if (showRelax && t1 > 0 && t2 > 0) {
        Mxy *= Math.exp(-dtSec / (t2 / 1000));
        Mz = 1 + (Mz - 1) * Math.exp(-dtSec / (t1 / 1000));
      }

      const Mx = Mxy * Math.cos(phi);
      const My = Mxy * Math.sin(phi);

      // Three.js mapping: physics X->X, physics Z(B0)->Y(up), physics Y->Z
      const tx = Mx;
      const ty = Mz;
      const tz = My;

      const mag = Math.sqrt(tx * tx + ty * ty + tz * tz);
      if (mag > 0.001) {
        const dir = new THREE.Vector3(tx, ty, tz).normalize();
        const up = new THREE.Vector3(0, 1, 0);
        const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
        arrowGroup.quaternion.copy(quat);
        arrowGroup.scale.set(1, mag, 1);
      }

      trailPositions[trailIdx * 3] = tx * mag;
      trailPositions[trailIdx * 3 + 1] = ty * mag;
      trailPositions[trailIdx * 3 + 2] = tz * mag;
      trailIdx = (trailIdx + 1) % TRAIL_LEN;
      trailCount = Math.min(trailCount + 1, TRAIL_LEN);
      trailGeo.setDrawRange(0, trailCount);
      trailGeo.attributes.position.needsUpdate = true;

      // Update scope buffer (subsample to ~30fps for readability)
      scopeFrameCount++;
      if (scopeFrameCount % 2 === 0) {
        mxScope[scopeIdx % SCOPE_LEN] = Mx;
        scopeIdx++;
      }

      drawPanel(sigCtx, SIG_WIDTH, SIG_HEIGHT, Mxy, Mz, Mx, b0, mxScope, scopeIdx, SCOPE_LEN);

      controls.update();
      renderer.render(scene, camera);
    }

    animId = requestAnimationFrame(animate);

    // --- Signal panel drawing: analog meters + oscilloscope ---
    function drawPanel(ctx, w, h, curMxy, curMz, curMx, b0, scope, sIdx, sLen) {
      ctx.clearRect(0, 0, w, h);

      // Background
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, w, h);

      const pad = 16;

      // --- Title ---
      ctx.fillStyle = "#ccc";
      ctx.font = "bold 13px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Signal Readout", w / 2, pad + 2);
      ctx.textAlign = "left";

      // --- Draw a semicircular analog meter ---
      function drawMeter(cx, cy, radius, value, minVal, maxVal, label, valueColor, arcColor) {
        const startAngle = Math.PI * 0.8;  // ~144° (lower left)
        const endAngle = Math.PI * 0.2;    // ~36° (lower right)
        const totalArc = 2 * Math.PI - (startAngle - endAngle);

        // Meter background arc
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, 2 * Math.PI + endAngle, false);
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 12;
        ctx.lineCap = "round";
        ctx.stroke();

        // Filled arc for current value
        const frac = (value - minVal) / (maxVal - minVal);
        const clampedFrac = Math.min(Math.max(frac, 0), 1);
        const valueAngle = startAngle + totalArc * clampedFrac;
        if (clampedFrac > 0.01) {
          ctx.beginPath();
          ctx.arc(cx, cy, radius, startAngle, valueAngle, false);
          ctx.strokeStyle = arcColor;
          ctx.lineWidth = 12;
          ctx.lineCap = "round";
          ctx.stroke();
        }

        // Tick marks
        ctx.lineWidth = 1;
        const nTicks = 5;
        for (let i = 0; i <= nTicks; i++) {
          const tickFrac = i / nTicks;
          const tickAngle = startAngle + totalArc * tickFrac;
          const inner = radius - 18;
          const outer = radius - 8;
          ctx.beginPath();
          ctx.moveTo(cx + inner * Math.cos(tickAngle), cy + inner * Math.sin(tickAngle));
          ctx.lineTo(cx + outer * Math.cos(tickAngle), cy + outer * Math.sin(tickAngle));
          ctx.strokeStyle = "#555";
          ctx.stroke();

          // Tick label
          const tickVal = minVal + (maxVal - minVal) * tickFrac;
          const labelR = radius - 24;
          ctx.fillStyle = "#555";
          ctx.font = "8px Arial";
          ctx.textAlign = "center";
          ctx.fillText(tickVal.toFixed(1), cx + labelR * Math.cos(tickAngle), cy + labelR * Math.sin(tickAngle) + 3);
        }

        // Needle
        const needleLen = radius - 10;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + needleLen * Math.cos(valueAngle), cy + needleLen * Math.sin(valueAngle));
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.lineCap = "round";
        ctx.stroke();
        // Center dot
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();

        // Label below
        ctx.fillStyle = "#aaa";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(label, cx, cy + radius * 0.35);

        // Digital readout
        ctx.fillStyle = valueColor;
        ctx.font = "bold 16px monospace";
        ctx.fillText(value.toFixed(3), cx, cy + radius * 0.55);

        ctx.textAlign = "left";
      }

      // --- |Mxy| meter (left) ---
      const meterRadius = 55;
      const meterY = pad + 22 + meterRadius + 10;
      drawMeter(w * 0.3, meterY, meterRadius, curMxy, 0, 1, "|Mxy| (signal)", "#ff6b6b", "#ff4444");

      // --- Mz meter (right) ---
      drawMeter(w * 0.7, meterY, meterRadius, curMz, -1, 1, "Mz (longitudinal)", "#4ecdc4", "#22aa88");

      // --- Oscilloscope: Mx (coil signal) ---
      const barW = w - pad * 2;
      const scopeTop = meterY + meterRadius * 0.7 + 10;
      const scopeH = h - scopeTop - 35;
      const scopeW = barW;

      ctx.fillStyle = "#aaa";
      ctx.font = "11px Arial";
      ctx.fillText("Mx (coil signal \u2014 oscilloscope)", pad, scopeTop - 4);

      // Scope background
      ctx.fillStyle = "#0d0d1a";
      ctx.fillRect(pad, scopeTop, scopeW, scopeH);
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, scopeTop, scopeW, scopeH);

      // Grid lines
      ctx.strokeStyle = "#1a1a33";
      ctx.lineWidth = 0.5;
      for (let i = 1; i < 4; i++) {
        const gy = scopeTop + (i / 4) * scopeH;
        ctx.beginPath();
        ctx.moveTo(pad, gy);
        ctx.lineTo(pad + scopeW, gy);
        ctx.stroke();
      }
      // Center line (zero)
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, scopeTop + scopeH / 2);
      ctx.lineTo(pad + scopeW, scopeTop + scopeH / 2);
      ctx.stroke();

      // Draw Mx trace (always fixed scale -1 to +1)
      ctx.beginPath();
      ctx.strokeStyle = "#44ff88";
      ctx.lineWidth = 1.5;
      const n = Math.min(sIdx, sLen);
      for (let i = 0; i < n; i++) {
        const dataIdx = (sIdx - n + i + sLen) % sLen;
        const x = pad + (i / sLen) * scopeW;
        const y = scopeTop + scopeH / 2 - (scope[dataIdx] * scopeH / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Scope scale labels
      ctx.fillStyle = "#444";
      ctx.font = "8px Arial";
      ctx.fillText("+1", pad + 2, scopeTop + 10);
      ctx.fillText("-1", pad + 2, scopeTop + scopeH - 3);

      // --- Larmor readout ---
      ctx.fillStyle = "#666";
      ctx.font = "10px Arial";
      const larmor = (42.576 * b0).toFixed(1);
      ctx.fillText(`B\u2080=${b0.toFixed(1)}T   f\u2080=${larmor} MHz`, pad, h - 8);
    }

    // --- Cleanup ---
    return () => {
      cancelAnimationFrame(animId);
      model.off && model.off("change:flip_angle");
      model.off && model.off("change:b0");
      model.off && model.off("change:show_relaxation");
      model.off && model.off("change:t1");
      model.off && model.off("change:t2");
      renderer.dispose();
      controls.dispose();
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (obj.material.map) obj.material.map.dispose();
          obj.material.dispose();
        }
      });
    };
  },
};
