// Frequency & Phase Encoding Widget - Canvas 2D
// Animated visualization of gradient encoding in MRI

export default {
  render({ model, el }) {
    const W = 700;
    const H = 420;
    const DPR = Math.min(window.devicePixelRatio, 2);

    const canvas = document.createElement("canvas");
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    canvas.style.borderRadius = "6px";
    canvas.style.border = "1px solid #ddd";
    el.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    ctx.scale(DPR, DPR);

    const GRID = 6; // 6x6 grid of voxels
    const CELL = 48;
    const gridLeft = 40;
    const gridTop = 60;
    const gridW = GRID * CELL;
    const gridH = GRID * CELL;

    // Each voxel has a base "proton density" (brightness)
    const rng = mulberry32(42);
    function mulberry32(a) {
      return function() {
        let t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      };
    }

    const voxels = [];
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        // Simple brain-like pattern
        const cx = (x - GRID / 2 + 0.5) / (GRID / 2);
        const cy = (y - GRID / 2 + 0.5) / (GRID / 2);
        const dist = Math.sqrt(cx * cx + cy * cy);
        let density = 0;
        if (dist < 0.85) density = 0.7;  // brain
        if (dist < 0.5 && Math.abs(cx) < 0.3) density = 1.0; // ventricle-ish
        if (dist > 0.7 && dist < 0.85) density = 0.82; // cortex

        voxels.push({
          x, y, density,
          phase: 0,       // current accumulated phase
          frequency: 0,   // current frequency (from x-gradient)
        });
      }
    }

    let lastTime = null;
    let animId;

    // Animation stages
    const stages = [
      { name: "No gradients", duration: 3, desc: "All voxels precess at the same frequency \u2014 no spatial information" },
      { name: "Frequency encoding (Gx)", duration: 5, desc: "X-gradient makes each column precess at a different frequency" },
      { name: "Phase encoding (Gy)", duration: 4, desc: "Y-gradient pulse gives each row a different phase offset" },
      { name: "Both gradients", duration: 5, desc: "Each voxel has a unique (frequency, phase) signature!" },
    ];
    let stageIdx = 0;
    let stageTime = 0;

    function animate(timestamp) {
      animId = requestAnimationFrame(animate);

      if (!lastTime) lastTime = timestamp;
      const dtMs = Math.min(timestamp - lastTime, 50);
      lastTime = timestamp;
      const dt = dtMs / 1000;
      const speed = model.get("speed");

      stageTime += dt * speed;
      const stage = stages[stageIdx];

      if (stageTime > stage.duration) {
        stageTime -= stage.duration;
        stageIdx = (stageIdx + 1) % stages.length;
        // Reset phases between stages
        if (stageIdx === 0) {
          for (const v of voxels) v.phase = 0;
        }
      }

      const t = timestamp / 1000;
      const currentStage = stages[stageIdx];

      // Update voxel frequencies and phases
      for (const v of voxels) {
        const baseFreq = 2.0; // base precession (visual)
        let freqGx = 0;
        let phaseGy = 0;

        if (currentStage.name === "Frequency encoding (Gx)" || currentStage.name === "Both gradients") {
          freqGx = (v.x - GRID / 2 + 0.5) * 0.8; // frequency varies with x
        }
        if (currentStage.name === "Phase encoding (Gy)" || currentStage.name === "Both gradients") {
          phaseGy = (v.y - GRID / 2 + 0.5) * 0.5 * stageTime; // phase accumulates with y
        }

        v.frequency = baseFreq + freqGx;
        v.phase += v.frequency * dt * Math.PI * 2;
        v.phase += phaseGy * dt;
      }

      draw(ctx, W, H, voxels, GRID, CELL, gridLeft, gridTop, gridW, gridH, currentStage, stageIdx, stages.length, stageTime);
    }

    function draw(ctx, w, h, voxels, grid, cell, gl, gt, gw, gh, stage, sIdx, sTotal, sTime) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fafafa";
      ctx.fillRect(0, 0, w, h);

      // Title
      ctx.fillStyle = "#333";
      ctx.font = "bold 14px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Spatial Encoding: Frequency & Phase", w / 2, 22);

      // Stage indicator
      ctx.fillStyle = "#666";
      ctx.font = "bold 12px Arial";
      ctx.fillText(`Stage ${sIdx + 1}/${sTotal}: ${stage.name}`, w / 2, 42);
      ctx.font = "11px Arial";
      ctx.fillStyle = "#888";
      ctx.fillText(stage.desc, w / 2, 56);
      ctx.textAlign = "left";

      // Draw voxel grid
      for (const v of voxels) {
        const x = gl + v.x * cell;
        const y = gt + v.y * cell;

        // Voxel background (brightness = density)
        const bright = Math.floor(v.density * 180 + 40);
        ctx.fillStyle = `rgb(${bright}, ${bright}, ${bright})`;
        ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);

        // Spin arrow showing current phase
        if (v.density > 0) {
          const cx = x + cell / 2;
          const cy = y + cell / 2;
          const arrowLen = cell * 0.35;
          const ax = cx + Math.cos(v.phase) * arrowLen;
          const ay = cy - Math.sin(v.phase) * arrowLen;

          // Color based on frequency (hue shifts with column)
          const hue = 200 + v.frequency * 30;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(ax, ay);
          ctx.strokeStyle = `hsl(${hue}, 70%, 45%)`;
          ctx.lineWidth = 2;
          ctx.stroke();

          // Arrowhead
          const angle = Math.atan2(cy - ay, ax - cx);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - 6 * Math.cos(angle - 0.4), ay + 6 * Math.sin(angle - 0.4));
          ctx.lineTo(ax - 6 * Math.cos(angle + 0.4), ay + 6 * Math.sin(angle + 0.4));
          ctx.closePath();
          ctx.fillStyle = `hsl(${hue}, 70%, 45%)`;
          ctx.fill();
        }
      }

      // Grid border
      ctx.strokeStyle = "#999";
      ctx.lineWidth = 1;
      ctx.strokeRect(gl, gt, gw, gh);

      // Grid lines
      ctx.strokeStyle = "#ddd";
      ctx.lineWidth = 0.5;
      for (let i = 1; i < grid; i++) {
        ctx.beginPath();
        ctx.moveTo(gl + i * cell, gt);
        ctx.lineTo(gl + i * cell, gt + gh);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(gl, gt + i * cell);
        ctx.lineTo(gl + gw, gt + i * cell);
        ctx.stroke();
      }

      // Axis labels
      ctx.fillStyle = "#cc4444";
      ctx.font = "bold 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText("x (frequency encoding) \u2192", gl + gw / 2, gt + gh + 20);
      ctx.save();
      ctx.translate(gl - 20, gt + gh / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = "#44aa44";
      ctx.fillText("y (phase encoding) \u2192", 0, 0);
      ctx.restore();
      ctx.textAlign = "left";

      // Right panel: explanation
      const rx = gl + gw + 30;
      const ry = gt;
      const rw = w - rx - 15;

      ctx.fillStyle = "#333";
      ctx.font = "bold 12px Arial";
      ctx.fillText("How it works:", rx, ry + 10);

      ctx.fillStyle = "#555";
      ctx.font = "11px Arial";
      const lines = [
        "",
        "Each arrow = a proton spin",
        "in that voxel.",
        "",
        "Frequency encoding (Gx):",
        "  Columns spin at different",
        "  rates during readout.",
        "",
        "Phase encoding (Gy):",
        "  Rows accumulate different",
        "  phases before readout.",
        "",
        "Result: each voxel has a",
        "unique (freq, phase) pair.",
        "",
        "2D FFT decodes this into",
        "an image!"
      ];
      for (let i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], rx, ry + 30 + i * 16);
      }

      // Frequency legend at bottom of grid
      ctx.fillStyle = "#888";
      ctx.font = "9px Arial";
      ctx.textAlign = "center";
      for (let x = 0; x < grid; x++) {
        const f = (2.0 + (x - grid / 2 + 0.5) * 0.8).toFixed(1);
        const showFreq = stage.name.includes("Frequency") || stage.name === "Both gradients";
        if (showFreq) {
          ctx.fillText(`f=${f}`, gl + x * cell + cell / 2, gt + gh + 34);
        }
      }
      ctx.textAlign = "left";
    }

    animId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animId);
    };
  },
};
