// Compass Widget - Canvas 2D animated compass needle
// Shows damped oscillation in a magnetic field

export default {
  render({ model, el }) {
    const SIZE = 380;
    const DPR = Math.min(window.devicePixelRatio, 2);

    // --- Built-in controls (work in both marimo and MyST) ---
    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;align-items:center;gap:12px;margin-bottom:10px;font-family:system-ui,sans-serif;font-size:14px;";
    const sliderLabel = document.createElement("label");
    sliderLabel.textContent = "B\u2080 field strength (mT): ";
    sliderLabel.style.fontWeight = "500";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "5";
    slider.step = "0.1";
    slider.value = String(model.get("b0") || 3.0);
    slider.style.cssText = "flex:1;max-width:300px;";
    const valSpan = document.createElement("span");
    valSpan.textContent = Number(slider.value).toFixed(1);
    valSpan.style.cssText = "min-width:30px;font-variant-numeric:tabular-nums;";
    slider.addEventListener("input", () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = v.toFixed(1);
      model.set("b0", v);
      if (model.save_changes) model.save_changes();
    });
    sliderLabel.appendChild(slider);
    controls.appendChild(sliderLabel);
    controls.appendChild(valSpan);
    el.appendChild(controls);

    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.gap = "12px";
    wrapper.style.alignItems = "flex-start";
    wrapper.style.justifyContent = "center";
    el.appendChild(wrapper);

    // Compass canvas
    const canvas = document.createElement("canvas");
    canvas.width = SIZE * DPR;
    canvas.height = SIZE * DPR;
    canvas.style.width = SIZE + "px";
    canvas.style.height = SIZE + "px";
    wrapper.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    ctx.scale(DPR, DPR);

    // Signal canvas
    const SIG_W = 360;
    const SIG_H = SIZE;
    const sigCanvas = document.createElement("canvas");
    sigCanvas.width = SIG_W * DPR;
    sigCanvas.height = SIG_H * DPR;
    sigCanvas.style.width = SIG_W + "px";
    sigCanvas.style.height = SIG_H + "px";
    sigCanvas.style.borderRadius = "6px";
    sigCanvas.style.border = "1px solid #ddd";
    wrapper.appendChild(sigCanvas);
    const sigCtx = sigCanvas.getContext("2d");
    sigCtx.scale(DPR, DPR);

    const SCOPE_LEN = 300;
    const signalBuf = new Float32Array(SCOPE_LEN);
    let sigIdx = 0;

    // Physics state
    let angle = 1.5;     // start displaced (~86°) so oscillation is visible immediately
    let angVel = 0;      // angular velocity
    let lastTime = null;
    let pushed = false;
    let pushTime = 0;
    let prevB0 = model.get("b0");

    // Re-kick needle whenever B0 changes
    model.on("change:b0", () => {
      const newB0 = model.get("b0");
      if (Math.abs(newB0 - prevB0) > 0.01) {
        angle = 1.2 + Math.random() * 0.5; // kick to ~70-100°
        angVel = 2.0;
        pushTime = 0;
      }
      prevB0 = newB0;
    });

    // Push button
    const btn = document.createElement("button");
    btn.textContent = "\u21bb Push needle";
    btn.style.cssText = "position:absolute;margin-top:8px;padding:6px 16px;border-radius:6px;border:1px solid #888;background:#f0f0f0;cursor:pointer;font-size:13px;";
    // Place it below the compass
    const btnWrap = document.createElement("div");
    btnWrap.style.cssText = "display:flex;justify-content:center;width:" + SIZE + "px;";
    btnWrap.appendChild(btn);

    const leftCol = document.createElement("div");
    leftCol.style.display = "flex";
    leftCol.style.flexDirection = "column";
    leftCol.style.alignItems = "center";
    // Move canvas into leftCol
    wrapper.removeChild(canvas);
    leftCol.appendChild(canvas);
    leftCol.appendChild(btnWrap);
    wrapper.insertBefore(leftCol, sigCanvas);

    btn.addEventListener("click", () => {
      // Give the needle a push (add angular velocity)
      angVel += 4.0 + Math.random() * 2;
      pushed = true;
      pushTime = 0;
    });

    let animId;

    function animate(timestamp) {
      animId = requestAnimationFrame(animate);

      if (!lastTime) lastTime = timestamp;
      const dtMs = Math.min(timestamp - lastTime, 50);
      lastTime = timestamp;
      const dt = dtMs / 1000;

      const b0 = model.get("b0");

      if (b0 > 0) {
        // Damped harmonic oscillator: torque toward 0, damping proportional to b0
        const springK = b0 * 8.0;    // restoring torque
        const damping = b0 * 1.2;    // damping coefficient
        const torque = -springK * angle;
        angVel += torque * dt;
        angVel *= Math.exp(-damping * dt);  // exponential damping
      } else {
        // No field: just friction
        angVel *= Math.exp(-0.3 * dt);
      }

      angle += angVel * dt;
      pushTime += dt;

      // Record coil signal (proportional to angular velocity / rate of change)
      const coilSignal = b0 > 0 ? Math.sin(angle) * Math.exp(-0.5 * b0 * pushTime) : 0;
      signalBuf[sigIdx % SCOPE_LEN] = angle / Math.PI; // normalized angle
      sigIdx++;

      drawCompass(ctx, SIZE, angle, b0);
      drawSignal(sigCtx, SIG_W, SIG_H, signalBuf, sigIdx, SCOPE_LEN, b0, angle);
    }

    function drawCompass(ctx, s, angle, b0) {
      const cx = s / 2;
      const cy = s / 2;
      const r = s * 0.38;
      ctx.clearRect(0, 0, s, s);

      // Background circle
      ctx.beginPath();
      ctx.arc(cx, cy, r + 20, 0, Math.PI * 2);
      ctx.fillStyle = "#f8f8f0";
      ctx.fill();
      ctx.strokeStyle = "#bbb";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Compass rose markings
      ctx.fillStyle = "#999";
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      ctx.fillText("N", cx, cy - r - 6);
      ctx.fillText("S", cx, cy + r + 14);
      ctx.fillText("E", cx + r + 10, cy + 4);
      ctx.fillText("W", cx - r - 10, cy + 4);

      // Tick marks
      for (let i = 0; i < 36; i++) {
        const a = (i / 36) * Math.PI * 2 - Math.PI / 2;
        const inner = i % 9 === 0 ? r - 10 : r - 5;
        const outer = r;
        ctx.beginPath();
        ctx.moveTo(cx + inner * Math.cos(a), cy + inner * Math.sin(a));
        ctx.lineTo(cx + outer * Math.cos(a), cy + outer * Math.sin(a));
        ctx.strokeStyle = i % 9 === 0 ? "#666" : "#ccc";
        ctx.lineWidth = i % 9 === 0 ? 2 : 1;
        ctx.stroke();
      }

      // B0 field indicator (blue arrow at top if active)
      if (b0 > 0) {
        const arrowLen = 25 + b0 * 8;
        ctx.beginPath();
        ctx.moveTo(cx, cy - r - 25);
        ctx.lineTo(cx, cy - r - 25 - arrowLen);
        ctx.strokeStyle = "#2266cc";
        ctx.lineWidth = 3;
        ctx.stroke();
        // Arrowhead
        ctx.beginPath();
        ctx.moveTo(cx, cy - r - 25 - arrowLen);
        ctx.lineTo(cx - 6, cy - r - 25 - arrowLen + 10);
        ctx.lineTo(cx + 6, cy - r - 25 - arrowLen + 10);
        ctx.closePath();
        ctx.fillStyle = "#2266cc";
        ctx.fill();
        ctx.fillStyle = "#2266cc";
        ctx.font = "bold 13px Arial";
        ctx.fillText(`B\u2080 = ${b0.toFixed(1)} mT`, cx, cy - r - 25 - arrowLen - 8);
      } else {
        ctx.fillStyle = "#999";
        ctx.font = "12px Arial";
        ctx.fillText("No B\u2080 field", cx, cy - r - 30);
      }

      // Compass needle (rotates by angle; 0 = pointing up/north)
      const needleLen = r * 0.85;
      const nx = Math.sin(angle);
      const ny = -Math.cos(angle);

      // Red (north) half
      ctx.beginPath();
      ctx.moveTo(cx - ny * 5, cy + nx * 5);
      ctx.lineTo(cx + nx * needleLen, cy + ny * needleLen);
      ctx.lineTo(cx + ny * 5, cy - nx * 5);
      ctx.closePath();
      ctx.fillStyle = "#cc2222";
      ctx.fill();

      // Blue (south) half
      ctx.beginPath();
      ctx.moveTo(cx - ny * 5, cy + nx * 5);
      ctx.lineTo(cx - nx * needleLen * 0.6, cy - ny * needleLen * 0.6);
      ctx.lineTo(cx + ny * 5, cy - nx * 5);
      ctx.closePath();
      ctx.fillStyle = "#2244aa";
      ctx.fill();

      // Center pin
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#888";
      ctx.fill();
    }

    function drawSignal(ctx, w, h, buf, idx, bufLen, b0, angle) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, w, h);

      const pad = 30;

      // Title
      ctx.fillStyle = "#ccc";
      ctx.font = "bold 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Coil Signal (needle oscillation)", w / 2, pad - 10);
      ctx.textAlign = "left";

      // Scope area
      const scopeTop = pad + 5;
      const scopeH = h - scopeTop - 60;
      const scopeW = w - pad * 2;

      ctx.fillStyle = "#0d0d1a";
      ctx.fillRect(pad, scopeTop, scopeW, scopeH);
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, scopeTop, scopeW, scopeH);

      // Center line
      ctx.strokeStyle = "#2a2a3e";
      ctx.beginPath();
      ctx.moveTo(pad, scopeTop + scopeH / 2);
      ctx.lineTo(pad + scopeW, scopeTop + scopeH / 2);
      ctx.stroke();

      // Grid
      ctx.strokeStyle = "#1a1a33";
      ctx.lineWidth = 0.5;
      for (let i = 1; i < 4; i++) {
        const gy = scopeTop + (i / 4) * scopeH;
        ctx.beginPath();
        ctx.moveTo(pad, gy);
        ctx.lineTo(pad + scopeW, gy);
        ctx.stroke();
      }

      // Signal trace (needle angle normalized)
      ctx.beginPath();
      ctx.strokeStyle = "#44aaff";
      ctx.lineWidth = 2;
      const n = Math.min(idx, bufLen);
      for (let i = 0; i < n; i++) {
        const dataIdx = (idx - n + i + bufLen) % bufLen;
        const x = pad + (i / bufLen) * scopeW;
        // Scale: angle/PI maps to full height
        const y = scopeTop + scopeH / 2 - (buf[dataIdx] * scopeH / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Scale labels
      ctx.fillStyle = "#555";
      ctx.font = "9px Arial";
      ctx.fillText("+", pad + 2, scopeTop + 10);
      ctx.fillText("-", pad + 2, scopeTop + scopeH - 3);
      ctx.fillText("0", pad + 2, scopeTop + scopeH / 2 - 3);

      // Current angle readout
      const angleDeg = (angle * 180 / Math.PI) % 360;
      ctx.fillStyle = "#aaa";
      ctx.font = "11px Arial";
      ctx.textAlign = "center";
      ctx.fillText(`Needle angle: ${angleDeg.toFixed(1)}\u00b0`, w / 2, h - 30);

      if (b0 > 0) {
        const freq = (0.3 * Math.sqrt(b0)).toFixed(2);
        ctx.fillStyle = "#44aaff";
        ctx.fillText(`Oscillation freq: ~${freq} Hz`, w / 2, h - 12);
      } else {
        ctx.fillStyle = "#888";
        ctx.fillText("No field \u2014 push the needle!", w / 2, h - 12);
      }
      ctx.textAlign = "left";
    }

    animId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animId);
      btn.removeEventListener("click", () => {});
    };
  },
};
