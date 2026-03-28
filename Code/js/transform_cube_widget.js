// Transform Cube Widget - Three.js 3D
// Interactive rigid body / affine transformation demonstration

const THREE_URL = "https://esm.sh/three@0.170.0";
const ORBIT_URL = "https://esm.sh/three@0.170.0/addons/controls/OrbitControls.js";

export default {
  async render({ model, el }) {
    const THREE = await import(THREE_URL);
    const { OrbitControls } = await import(ORBIT_URL);

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.gap = "16px";
    container.style.alignItems = "flex-start";
    container.style.justifyContent = "center";
    el.appendChild(container);

    const WIDTH = 550;
    const HEIGHT = 480;

    // --- Three.js Scene ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f5);

    const camera = new THREE.PerspectiveCamera(40, WIDTH / HEIGHT, 0.1, 200);
    camera.position.set(25, 20, 30);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(WIDTH, HEIGHT);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const dl = new THREE.DirectionalLight(0xffffff, 0.7);
    dl.position.set(20, 30, 20);
    scene.add(dl);

    // --- Grid floor ---
    const grid = new THREE.GridHelper(40, 20, 0xcccccc, 0xe0e0e0);
    grid.position.y = -10;
    scene.add(grid);

    // --- Axes ---
    function addAxis(from, to, color, label) {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...from), new THREE.Vector3(...to)
      ]);
      scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color })));
      // Label
      const c = document.createElement("canvas");
      c.width = 64; c.height = 32;
      const ctx = c.getContext("2d");
      ctx.font = "bold 24px Arial";
      ctx.fillStyle = color instanceof THREE.Color ? `#${color.getHexString()}` :
        `#${new THREE.Color(color).getHexString()}`;
      ctx.textAlign = "center";
      ctx.fillText(label, 32, 24);
      const tex = new THREE.CanvasTexture(c);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
      sprite.position.set(...to);
      sprite.scale.set(2, 1, 1);
      scene.add(sprite);
    }
    addAxis([-18, 0, 0], [18, 0, 0], 0xcc4444, "X");
    addAxis([0, -18, 0], [0, 18, 0], 0x44aa44, "Y");
    addAxis([0, 0, -18], [0, 0, 18], 0x4444cc, "Z");

    // --- Reference cube (ghost) ---
    const cubeSize = 10;
    const refGeo = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
    const refMat = new THREE.MeshBasicMaterial({
      color: 0x6688cc, wireframe: true, transparent: true, opacity: 0.3
    });
    const refCube = new THREE.Mesh(refGeo, refMat);
    scene.add(refCube);

    // --- Transformed cube ---
    const cubeGeo = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
    const cubeMat = new THREE.MeshLambertMaterial({
      color: 0xdd4444, transparent: true, opacity: 0.7
    });
    const cube = new THREE.Mesh(cubeGeo, cubeMat);
    scene.add(cube);

    // Wireframe overlay on transformed cube
    const wireGeo = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
    const wireMat = new THREE.MeshBasicMaterial({ color: 0xaa2222, wireframe: true });
    const wireframe = new THREE.Mesh(wireGeo, wireMat);
    cube.add(wireframe);

    // --- Affine matrix display (Canvas 2D) ---
    const MAT_W = 260;
    const MAT_H = 480;
    const matCanvas = document.createElement("canvas");
    matCanvas.width = MAT_W * 2;
    matCanvas.height = MAT_H * 2;
    matCanvas.style.width = MAT_W + "px";
    matCanvas.style.height = MAT_H + "px";
    matCanvas.style.borderRadius = "6px";
    matCanvas.style.border = "1px solid #ddd";
    container.appendChild(matCanvas);
    const matCtx = matCanvas.getContext("2d");
    matCtx.scale(2, 2);

    let animId;

    function animate() {
      animId = requestAnimationFrame(animate);

      // Read parameters
      const tx = model.get("trans_x");
      const ty = model.get("trans_y");
      const tz = model.get("trans_z");
      const rx = model.get("rot_x") * Math.PI / 180;
      const ry = model.get("rot_y") * Math.PI / 180;
      const rz = model.get("rot_z") * Math.PI / 180;
      const sx = model.get("scale_x");
      const sy = model.get("scale_y");
      const sz = model.get("scale_z");

      // Apply transforms
      cube.position.set(tx, ty, tz);
      cube.rotation.set(rx, ry, rz, "XYZ");
      cube.scale.set(sx, sy, sz);

      // Build affine matrix for display
      cube.updateMatrixWorld();
      const m = cube.matrixWorld.elements;

      drawMatrixPanel(matCtx, MAT_W, MAT_H, m, tx, ty, tz, rx, ry, rz, sx, sy, sz);

      controls.update();
      renderer.render(scene, camera);
    }

    function drawMatrixPanel(ctx, w, h, m, tx, ty, tz, rx, ry, rz, sx, sy, sz) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, w, h);

      const pad = 15;
      let y = pad;

      // Title
      ctx.fillStyle = "#ccc";
      ctx.font = "bold 13px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Affine Matrix (4\u00d74)", w / 2, y + 2);
      y += 25;

      // 4x4 matrix (Three.js stores column-major)
      ctx.font = "13px monospace";
      const cellW = 52;
      const cellH = 22;
      const matLeft = (w - cellW * 4) / 2;

      // Bracket lines
      ctx.strokeStyle = "#888";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(matLeft - 6, y - 4);
      ctx.lineTo(matLeft - 10, y - 4);
      ctx.lineTo(matLeft - 10, y + cellH * 4 + 4);
      ctx.lineTo(matLeft - 6, y + cellH * 4 + 4);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(matLeft + cellW * 4 + 6, y - 4);
      ctx.lineTo(matLeft + cellW * 4 + 10, y - 4);
      ctx.lineTo(matLeft + cellW * 4 + 10, y + cellH * 4 + 4);
      ctx.lineTo(matLeft + cellW * 4 + 6, y + cellH * 4 + 4);
      ctx.stroke();

      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
          // Three.js matrix is column-major: element[col*4+row]
          const val = m[col * 4 + row];
          const isRotScale = row < 3 && col < 3;
          const isTrans = col === 3 && row < 3;
          ctx.fillStyle = isTrans ? "#ff8866" : isRotScale ? "#66bbff" : "#888";
          ctx.textAlign = "center";
          ctx.fillText(val.toFixed(2), matLeft + col * cellW + cellW / 2, y + row * cellH + 16);
        }
      }
      y += cellH * 4 + 15;

      // Legend
      ctx.font = "10px Arial";
      ctx.textAlign = "left";
      ctx.fillStyle = "#66bbff";
      ctx.fillText("\u25a0 Rotation \u00d7 Scale", pad, y);
      y += 15;
      ctx.fillStyle = "#ff8866";
      ctx.fillText("\u25a0 Translation", pad, y);
      y += 25;

      // Parameter readout
      ctx.fillStyle = "#aaa";
      ctx.font = "bold 11px Arial";
      ctx.fillText("Current Parameters:", pad, y);
      y += 18;

      ctx.font = "11px monospace";
      const params = [
        ["Translation", `(${tx.toFixed(1)}, ${ty.toFixed(1)}, ${tz.toFixed(1)})`],
        ["Rotation (\u00b0)", `(${(rx*180/Math.PI).toFixed(1)}, ${(ry*180/Math.PI).toFixed(1)}, ${(rz*180/Math.PI).toFixed(1)})`],
        ["Scale", `(${sx.toFixed(2)}, ${sy.toFixed(2)}, ${sz.toFixed(2)})`],
      ];
      for (const [label, val] of params) {
        ctx.fillStyle = "#888";
        ctx.fillText(label + ":", pad, y);
        ctx.fillStyle = "#ddd";
        ctx.fillText(val, pad + 5, y + 14);
        y += 30;
      }

      // DOF info
      y += 10;
      ctx.fillStyle = "#666";
      ctx.font = "10px Arial";
      const lines = [
        "Rigid body: 6 DOF",
        "  (3 translation + 3 rotation)",
        "",
        "Full affine: 12 DOF",
        "  (+ 3 scale + 3 shear)",
        "",
        "Each transform changes the",
        "affine matrix that maps voxel",
        "coordinates to world space.",
      ];
      for (const line of lines) {
        ctx.fillText(line, pad, y);
        y += 14;
      }

      ctx.textAlign = "left";
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
