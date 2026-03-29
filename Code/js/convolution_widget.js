// Convolution Widget - Canvas 2D
// Animated convolution of stimulus events with HRF

export default {
  render({ model, el }) {
    const W = 750;
    const H = 400;
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

    // HRF function (double gamma)
    function gammaFunc(t, shape, scale) {
      if (t <= 0) return 0;
      const x = t / scale;
      return Math.pow(x, shape - 1) * Math.exp(-x) / (Math.pow(scale, shape) * gammaFn(shape));
    }
    // Stirling approximation for gamma function
    function gammaFn(n) {
      if (n === 1) return 1;
      if (n === 0.5) return Math.sqrt(Math.PI);
      return (n - 1) * gammaFn(n - 1);
    }
    // Simpler HRF using explicit formula
    function hrf(t, peakTime, undershootTime, ratio) {
      if (t <= 0) return 0;
      const peak = Math.pow(t / peakTime, peakTime) * Math.exp(-t + peakTime);
      const undershoot = Math.pow(t / undershootTime, undershootTime) * Math.exp(-t + undershootTime);
      return peak - ratio * undershoot;
    }

    // Pre-compute HRF
    const dt = 0.1; // seconds
    const hrfLen = 300; // 30 seconds
    const hrfArr = new Float32Array(hrfLen);
    let hrfMax = 0;
    for (let i = 0; i < hrfLen; i++) {
      hrfArr[i] = hrf(i * dt, 6, 16, 0.167);
      if (Math.abs(hrfArr[i]) > hrfMax) hrfMax = Math.abs(hrfArr[i]);
    }
    if (hrfMax > 0) for (let i = 0; i < hrfLen; i++) hrfArr[i] /= hrfMax;

    // Generate stimulus events based on pattern
    const totalDur = 600; // 60 seconds in samples
    function makeStimulus(pattern) {
      const stim = new Float32Array(totalDur);
      if (pattern === "single") {
        stim[50] = 1; // 5s
      } else if (pattern === "spaced") {
        stim[50] = 1; stim[200] = 1; stim[400] = 1;
      } else if (pattern === "block") {
        for (let s = 50; s < 550; s += 200) {
          for (let i = s; i < s + 100 && i < totalDur; i++) stim[i] = 1;
        }
      } else if (pattern === "rapid") {
        const onsets = [30, 80, 120, 180, 230, 290, 340, 400, 450, 520];
        for (const o of onsets) if (o < totalDur) stim[o] = 1;
      }
      return stim;
    }

    // Convolve (partial, up to current time)
    function convolve(stim, hrf, hrfLen, upTo) {
      const result = new Float32Array(upTo);
      for (let t = 0; t < upTo; t++) {
        let sum = 0;
        for (let k = 0; k < hrfLen && k <= t; k++) {
          sum += stim[t - k] * hrf[k];
        }
        result[t] = sum;
      }
      return result;
    }

    let currentSample = 0;
    let lastTime = null;
    let animId;

    function animate(timestamp) {
      animId = requestAnimationFrame(animate);

      if (!lastTime) lastTime = timestamp;
      const dtMs = Math.min(timestamp - lastTime, 50);
      lastTime = timestamp;

      const speed = model.get("speed");
      const pattern = model.get("pattern");

      // Advance time
      const samplesToAdd = Math.ceil(speed * 15 * dtMs / 1000);
      currentSample = Math.min(currentSample + samplesToAdd, totalDur);

      // Reset when done
      if (currentSample >= totalDur) {
        currentSample = 0;
      }

      const stim = makeStimulus(pattern);
      const bold = convolve(stim, hrfArr, hrfLen, currentSample);

      draw(ctx, W, H, stim, hrfArr, hrfLen, bold, currentSample, totalDur, dt);
    }

    function draw(ctx, w, h, stim, hrf, hrfLen, bold, current, total, dt) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fafafa";
      ctx.fillRect(0, 0, w, h);

      const pad = 50;
      const plotW = w - pad - 20;
      const gapY = 15;

      // Three rows: HRF (small), Stimulus, BOLD
      const hrfH = 70;
      const stimH = 80;
      const boldH = h - pad - hrfH - stimH - gapY * 3 - 20;

      const hrfTop = 25;
      const stimTop = hrfTop + hrfH + gapY;
      const boldTop = stimTop + stimH + gapY;

      // --- HRF (top, small) ---
      ctx.fillStyle = "#333";
      ctx.font = "bold 11px Arial";
      ctx.fillText("HRF (hemodynamic response function)", pad, hrfTop - 5);

      ctx.strokeStyle = "#eee";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, hrfTop, plotW * 0.4, hrfH);

      ctx.beginPath();
      ctx.strokeStyle = "#8855cc";
      ctx.lineWidth = 2;
      const hrfPlotW = plotW * 0.4;
      for (let i = 0; i < hrfLen; i++) {
        const x = pad + (i / hrfLen) * hrfPlotW;
        const y = hrfTop + hrfH / 2 - (hrf[i] * hrfH * 0.4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // HRF zero line
      ctx.strokeStyle = "#ddd";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(pad, hrfTop + hrfH / 2);
      ctx.lineTo(pad + hrfPlotW, hrfTop + hrfH / 2);
      ctx.stroke();

      ctx.fillStyle = "#999";
      ctx.font = "9px Arial";
      ctx.fillText("0", pad - 10, hrfTop + hrfH / 2 + 3);
      ctx.fillText("30s", pad + hrfPlotW - 15, hrfTop + hrfH + 12);

      // --- Stimulus (middle) ---
      ctx.fillStyle = "#333";
      ctx.font = "bold 11px Arial";
      ctx.fillText("Stimulus timing", pad, stimTop - 5);

      ctx.strokeStyle = "#eee";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, stimTop, plotW, stimH);

      // Stimulus bars
      for (let i = 0; i < total; i++) {
        if (stim[i] > 0) {
          const x = pad + (i / total) * plotW;
          const barW = Math.max(1, plotW / total);
          const opacity = i < current ? 1.0 : 0.15;
          ctx.fillStyle = `rgba(239, 85, 59, ${opacity})`;
          ctx.fillRect(x, stimTop + stimH - stim[i] * stimH * 0.8, barW, stim[i] * stimH * 0.8);
        }
      }

      // Playhead
      const playX = pad + (current / total) * plotW;
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(playX, stimTop);
      ctx.lineTo(playX, stimTop + stimH);
      ctx.stroke();
      ctx.setLineDash([]);

      // Individual HRF copies (ghost overlay) for events that have occurred
      for (let i = 0; i < current; i++) {
        if (stim[i] > 0) {
          ctx.beginPath();
          ctx.strokeStyle = "rgba(136, 85, 204, 0.25)";
          ctx.lineWidth = 1;
          for (let k = 0; k < hrfLen && (i + k) < total; k++) {
            const x = pad + ((i + k) / total) * plotW;
            const y = stimTop + stimH - (hrf[k] * stimH * 0.6) - stimH * 0.1;
            if (k === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }

      // Time labels
      ctx.fillStyle = "#999";
      ctx.font = "9px Arial";
      for (let s = 0; s <= 60; s += 10) {
        const x = pad + (s / (total * dt)) * plotW;
        ctx.fillText(`${s}s`, x - 6, stimTop + stimH + 12);
      }

      // --- BOLD signal (bottom) ---
      ctx.fillStyle = "#333";
      ctx.font = "bold 11px Arial";
      ctx.fillText("Predicted BOLD signal (stimulus \u2217 HRF)", pad, boldTop - 5);

      ctx.strokeStyle = "#eee";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, boldTop, plotW, boldH);

      // Zero line
      ctx.strokeStyle = "#ddd";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(pad, boldTop + boldH * 0.6);
      ctx.lineTo(pad + plotW, boldTop + boldH * 0.6);
      ctx.stroke();

      if (bold.length > 1) {
        // Find max for scaling (fixed at reasonable range)
        let maxBold = 0;
        for (let i = 0; i < bold.length; i++) {
          if (Math.abs(bold[i]) > maxBold) maxBold = Math.abs(bold[i]);
        }
        const scale = maxBold > 0 ? 1 / maxBold : 1;

        // Fill area
        ctx.beginPath();
        ctx.moveTo(pad, boldTop + boldH * 0.6);
        for (let i = 0; i < bold.length; i++) {
          const x = pad + (i / total) * plotW;
          const y = boldTop + boldH * 0.6 - (bold[i] * scale * boldH * 0.45);
          ctx.lineTo(x, y);
        }
        ctx.lineTo(pad + ((bold.length - 1) / total) * plotW, boldTop + boldH * 0.6);
        ctx.closePath();
        ctx.fillStyle = "rgba(54, 162, 235, 0.2)";
        ctx.fill();

        // Line
        ctx.beginPath();
        ctx.strokeStyle = "#2288dd";
        ctx.lineWidth = 2.5;
        for (let i = 0; i < bold.length; i++) {
          const x = pad + (i / total) * plotW;
          const y = boldTop + boldH * 0.6 - (bold[i] * scale * boldH * 0.45);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Playhead on BOLD
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(playX, boldTop);
      ctx.lineTo(playX, boldTop + boldH);
      ctx.stroke();
      ctx.setLineDash([]);

      // Time indicator
      const currentTime = (current * dt).toFixed(1);
      ctx.fillStyle = "#333";
      ctx.font = "bold 11px Arial";
      ctx.textAlign = "right";
      ctx.fillText(`t = ${currentTime}s`, w - 15, boldTop + boldH + 15);
      ctx.textAlign = "left";
    }

    animId = requestAnimationFrame(animate);

    model.on("change:pattern", () => { currentSample = 0; });

    return () => {
      cancelAnimationFrame(animId);
      model.off && model.off("change:pattern");
    };
  },
};
