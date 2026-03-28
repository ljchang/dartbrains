// Spin Ensemble Widget - Canvas 2D animation
// Shows dephasing/rephasing of spin isochromats for spin echo vs gradient echo

export default {
  render({ model, el }) {
    const WIDTH = 950;
    const HEIGHT = 400;
    const DPR = Math.min(window.devicePixelRatio, 2);

    const canvas = document.createElement("canvas");
    canvas.width = WIDTH * DPR;
    canvas.height = HEIGHT * DPR;
    canvas.style.width = WIDTH + "px";
    canvas.style.height = HEIGHT + "px";
    canvas.style.borderRadius = "8px";
    canvas.style.border = "1px solid #ddd";
    el.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    ctx.scale(DPR, DPR);

    // --- Spin configuration ---
    const N_SPINS = 16;
    const spins = [];
    for (let i = 0; i < N_SPINS; i++) {
      spins.push({
        // Frequency offset (determines dephasing rate)
        dFreq: (i - N_SPINS / 2 + 0.5) / (N_SPINS / 2) * 1.0,
        phase: 0,
        color: `hsl(${(i * 360 / N_SPINS)}, 70%, 55%)`,
      });
    }

    // --- Animation state machine ---
    // Phases: coherent -> dephasing -> [refocus pulse] -> rephasing -> echo -> dephasing again
    const PHASE_DURATIONS = {
      coherent: 1.0,
      dephasing1: 2.5,
      refocus: 0.3,
      rephasing: 2.5,
      echo: 1.0,
      dephasing2: 2.5,
      done: 3.0,
    };
    const PHASES = ["coherent", "dephasing1", "refocus", "rephasing", "echo", "dephasing2", "done"];
    const PHASE_LABELS = {
      coherent: "After 90\u00b0 excitation",
      dephasing1: "Dephasing...",
      refocus: "180\u00b0 refocusing pulse",
      rephasing: "Rephasing...",
      echo: "Echo!",
      dephasing2: "Dephasing again...",
      done: "\u2714 Complete \u2014 restarting...",
    };
    const GRE_LABELS = {
      coherent: "After RF excitation",
      dephasing1: "Dephasing (gradient on)...",
      refocus: "Gradient reversal",
      rephasing: "Rephasing...",
      echo: "Gradient echo!",
      dephasing2: "Dephasing (T\u2082* decay)...",
      done: "\u2714 Complete \u2014 restarting...",
    };

    let currentPhaseIdx = 0;
    let phaseTime = 0;
    let totalTime = 0;
    let lastTimestamp = null;

    // Signal history
    const SIG_HISTORY = 500;
    const signalHistory = new Float32Array(SIG_HISTORY);
    let sigIdx = 0;

    // Listen for changes
    model.on("change:sequence_type", () => resetAnimation());
    model.on("change:speed", () => {});

    function resetAnimation() {
      currentPhaseIdx = 0;
      phaseTime = 0;
      totalTime = 0;
      lastTimestamp = null;
      sigIdx = 0;
      signalHistory.fill(0);
      for (const s of spins) s.phase = 0;
    }

    let animId;

    function animate(timestamp) {
      animId = requestAnimationFrame(animate);

      if (model.get("paused")) {
        lastTimestamp = timestamp;
        return;
      }

      if (!lastTimestamp) lastTimestamp = timestamp;
      const dtMs = Math.min(timestamp - lastTimestamp, 50);
      lastTimestamp = timestamp;
      const speed = model.get("speed");
      const dt = (dtMs / 1000) * speed;
      const seqType = model.get("sequence_type");

      phaseTime += dt;
      totalTime += dt;

      const phaseName = PHASES[currentPhaseIdx];
      const phaseDur = PHASE_DURATIONS[phaseName];

      // Advance phase if time exceeded
      if (phaseTime >= phaseDur) {
        phaseTime -= phaseDur;
        currentPhaseIdx = (currentPhaseIdx + 1) % PHASES.length;
        const newPhase = PHASES[currentPhaseIdx];

        // Handle phase transitions
        if (newPhase === "coherent") {
          // Reset all spins to coherent
          for (const s of spins) s.phase = 0;
        } else if (newPhase === "refocus") {
          if (seqType === "spin_echo") {
            // 180° pulse: negate all phases
            for (const s of spins) s.phase = -s.phase;
          } else {
            // Gradient echo: reverse gradient direction (phases will converge)
            // Just mark the reversal point - the dFreq reversal happens in rephasing
          }
        }
      }

      // Update spin phases based on current phase
      const currentPhaseName = PHASES[currentPhaseIdx];
      if (currentPhaseName === "dephasing1" || currentPhaseName === "dephasing2") {
        for (const s of spins) {
          s.phase += s.dFreq * dt * Math.PI * 2;
        }
        // For GRE in dephasing2, add extra random-ish dephasing (T2' effects)
        if (seqType === "gradient_echo" && currentPhaseName === "dephasing2") {
          for (const s of spins) {
            s.phase += s.dFreq * dt * Math.PI * 0.5; // extra dephasing
          }
        }
      } else if (currentPhaseName === "rephasing") {
        // Phases converge
        for (const s of spins) {
          s.phase -= s.dFreq * dt * Math.PI * 2;
        }
        if (seqType === "gradient_echo") {
          // GRE: some residual dephasing from T2' effects doesn't refocus
          for (const s of spins) {
            s.phase += s.dFreq * dt * Math.PI * 0.3;
          }
        }
      }
      // coherent, refocus, echo: phases stay as they are

      // Compute net magnetization
      let netX = 0, netY = 0;
      for (const s of spins) {
        netX += Math.cos(s.phase);
        netY += Math.sin(s.phase);
      }
      netX /= N_SPINS;
      netY /= N_SPINS;
      const netMag = Math.sqrt(netX * netX + netY * netY);

      // Record signal
      signalHistory[sigIdx % SIG_HISTORY] = netMag;
      sigIdx++;

      // --- Drawing ---
      draw(ctx, WIDTH, HEIGHT, spins, netX, netY, netMag,
           currentPhaseName, seqType, signalHistory, sigIdx, SIG_HISTORY);
    }

    function draw(ctx, w, h, spins, netX, netY, netMag, phaseName, seqType, sigHist, sIdx, sigLen) {
      ctx.clearRect(0, 0, w, h);

      // Background
      ctx.fillStyle = "#fafafa";
      ctx.fillRect(0, 0, w, h);

      // --- Left: Spin diagram (circular) ---
      const cx = 150;
      const cy = h / 2;
      const radius = 110;

      // Circle border
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 5, 0, Math.PI * 2);
      ctx.strokeStyle = "#ddd";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw individual spins
      for (const s of spins) {
        const endX = cx + Math.cos(s.phase) * radius * 0.85;
        const endY = cy - Math.sin(s.phase) * radius * 0.85;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(cy - endY, endX - cx);
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - 8 * Math.cos(angle - 0.4), endY + 8 * Math.sin(angle - 0.4));
        ctx.lineTo(endX - 8 * Math.cos(angle + 0.4), endY + 8 * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fillStyle = s.color;
        ctx.fill();
      }

      // Draw net magnetization (bold red)
      const netEndX = cx + netX * radius;
      const netEndY = cy - netY * radius;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(netEndX, netEndY);
      ctx.strokeStyle = "#dd2222";
      ctx.lineWidth = 4;
      ctx.stroke();
      // Arrowhead
      const netAngle = Math.atan2(cy - netEndY, netEndX - cx);
      ctx.beginPath();
      ctx.moveTo(netEndX, netEndY);
      ctx.lineTo(netEndX - 14 * Math.cos(netAngle - 0.35), netEndY + 14 * Math.sin(netAngle - 0.35));
      ctx.lineTo(netEndX - 14 * Math.cos(netAngle + 0.35), netEndY + 14 * Math.sin(netAngle + 0.35));
      ctx.closePath();
      ctx.fillStyle = "#dd2222";
      ctx.fill();

      // |Mxy| bar
      const barX = 20;
      const barY = h - 35;
      const barW = 300;
      const barH = 14;
      ctx.fillStyle = "#eee";
      ctx.fillRect(barX, barY, barW, barH);
      ctx.fillStyle = netMag > 0.5 ? "#22aa66" : netMag > 0.2 ? "#ddaa22" : "#cc4444";
      ctx.fillRect(barX, barY, barW * netMag, barH);
      ctx.strokeStyle = "#999";
      ctx.lineWidth = 1;
      ctx.strokeRect(barX, barY, barW, barH);
      ctx.fillStyle = "#333";
      ctx.font = "bold 11px Arial";
      ctx.fillText(`|Mxy| = ${netMag.toFixed(2)}`, barX + barW + 8, barY + 12);

      // Phase label
      const labels = seqType === "spin_echo" ? PHASE_LABELS : GRE_LABELS;
      ctx.fillStyle = "#333";
      ctx.font = "bold 14px Arial";
      ctx.textAlign = "center";
      ctx.fillText(labels[phaseName] || phaseName, cx, 22);
      ctx.textAlign = "left";

      // Sequence type label
      ctx.fillStyle = "#666";
      ctx.font = "11px Arial";
      ctx.fillText(seqType === "spin_echo" ? "Spin Echo (90\u00b0-180\u00b0)" : "Gradient Echo", barX, barY - 8);

      // --- Right: Signal trace ---
      const sigLeft = 330;
      const sigTop = 30;
      const sigW = w - sigLeft - 15;
      const sigH = h - 70;

      // Border
      ctx.strokeStyle = "#ccc";
      ctx.lineWidth = 1;
      ctx.strokeRect(sigLeft, sigTop, sigW, sigH);

      // Title
      ctx.fillStyle = "#333";
      ctx.font = "bold 12px Arial";
      ctx.fillText("|Mxy| Signal", sigLeft + 4, sigTop - 8);

      // Grid lines
      ctx.strokeStyle = "#eee";
      ctx.lineWidth = 0.5;
      for (let i = 1; i < 4; i++) {
        const gy = sigTop + (i / 4) * sigH;
        ctx.beginPath();
        ctx.moveTo(sigLeft, gy);
        ctx.lineTo(sigLeft + sigW, gy);
        ctx.stroke();
      }

      // Phase boundary markers on the signal trace
      const phaseNames = ["90\u00b0", "dephase", seqType === "spin_echo" ? "180\u00b0" : "Grev", "rephase", "echo", "decay", ""];
      const phaseDurs = [1.0, 2.5, 0.3, 2.5, 1.0, 2.5, 3.0];
      let cumTime = 0;
      const totalSeqTime = phaseDurs.reduce((a, b) => a + b, 0);
      ctx.font = "8px Arial";
      ctx.textAlign = "center";
      for (let p = 0; p < phaseDurs.length - 1; p++) {
        cumTime += phaseDurs[p];
        const frac = cumTime / totalSeqTime;
        const mx = sigLeft + frac * sigW;
        // Vertical dashed line
        ctx.strokeStyle = "#bbb";
        ctx.lineWidth = 0.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(mx, sigTop);
        ctx.lineTo(mx, sigTop + sigH);
        ctx.stroke();
        ctx.setLineDash([]);
        // Label
        if (p < phaseNames.length - 1 && phaseNames[p + 1]) {
          ctx.fillStyle = "#888";
          const labelX = sigLeft + (cumTime - phaseDurs[p] / 2) / totalSeqTime * sigW;
          ctx.fillText(phaseNames[p], labelX, sigTop + sigH + 12);
        }
      }
      // First label
      ctx.fillText(phaseNames[0], sigLeft + (phaseDurs[0] / 2) / totalSeqTime * sigW, sigTop + sigH + 12);
      ctx.textAlign = "left";

      // Signal trace (don't wrap -- show linear from 0 to buffer fill)
      ctx.beginPath();
      ctx.strokeStyle = "#dd2222";
      ctx.lineWidth = 2;
      const n = Math.min(sIdx, sigLen);
      for (let i = 0; i < n; i++) {
        const x = sigLeft + (i / sigLen) * sigW;
        const y = sigTop + sigH - sigHist[i % sigLen] * sigH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Playhead
      if (n > 0 && n < sigLen) {
        const px = sigLeft + (n / sigLen) * sigW;
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(px, sigTop);
        ctx.lineTo(px, sigTop + sigH);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Axis labels
      ctx.fillStyle = "#999";
      ctx.font = "9px Arial";
      ctx.fillText("1.0", sigLeft - 22, sigTop + 10);
      ctx.fillText("0", sigLeft - 10, sigTop + sigH + 2);
      ctx.textAlign = "right";
      ctx.fillText("time \u2192", sigLeft + sigW, sigTop + sigH + 12);
      ctx.textAlign = "left";
    }

    animId = requestAnimationFrame(animate);

    // --- Cleanup ---
    return () => {
      cancelAnimationFrame(animId);
      model.off("change:sequence_type");
      model.off("change:speed");
    };
  },
};
