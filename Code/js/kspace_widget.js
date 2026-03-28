// K-Space Widget - Canvas 2D
// Progressive k-space filling with real-time image reconstruction via FFT

export default {
  render({ model, el }) {
    const W = 750;
    const H = 380;
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

    // --- Generate phantom and its k-space ---
    const N = 128;
    const phantom = new Float32Array(N * N);
    // Simple brain-like phantom
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const cx = N / 2, cy = N / 2;
        const dx = (x - cx) / 45, dy = (y - cy) / 55;
        if (dx * dx + dy * dy < 1) phantom[y * N + x] = 0.3; // skull
        const bx = (x - cx) / 38, by = (y - cy) / 48;
        if (bx * bx + by * by < 1) phantom[y * N + x] = 0.7; // brain
        // Ventricles
        const vl = ((x - cx + 10) / 8) ** 2 + ((y - cy) / 18) ** 2;
        const vr = ((x - cx - 10) / 8) ** 2 + ((y - cy) / 18) ** 2;
        if (vl < 1 || vr < 1) phantom[y * N + x] = 1.0;
        // Cortex ring
        const co = ((x - cx) / 36) ** 2 + ((y - cy) / 46) ** 2;
        const ci = ((x - cx) / 28) ** 2 + ((y - cy) / 38) ** 2;
        if (co < 1 && ci >= 1 && bx * bx + by * by < 1) phantom[y * N + x] = 0.82;
      }
    }

    // Simple 2D FFT (real-valued input, returns complex k-space)
    // Using DFT row-column decomposition
    function fft1d(re, im, n, inverse) {
      // Bit-reversal permutation
      for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
          [re[i], re[j]] = [re[j], re[i]];
          [im[i], im[j]] = [im[j], im[i]];
        }
      }
      for (let len = 2; len <= n; len *= 2) {
        const ang = (inverse ? -1 : 1) * 2 * Math.PI / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
          let curRe = 1, curIm = 0;
          for (let j = 0; j < len / 2; j++) {
            const a = i + j, b = i + j + len / 2;
            const tRe = re[b] * curRe - im[b] * curIm;
            const tIm = re[b] * curIm + im[b] * curRe;
            re[b] = re[a] - tRe; im[b] = im[a] - tIm;
            re[a] += tRe; im[a] += tIm;
            const newCurRe = curRe * wRe - curIm * wIm;
            curIm = curRe * wIm + curIm * wRe;
            curRe = newCurRe;
          }
        }
      }
      if (inverse) {
        for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
      }
    }

    function fft2d(reArr, imArr, n, inverse) {
      const rowRe = new Float32Array(n), rowIm = new Float32Array(n);
      // Rows
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) { rowRe[x] = reArr[y * n + x]; rowIm[x] = imArr[y * n + x]; }
        fft1d(rowRe, rowIm, n, inverse);
        for (let x = 0; x < n; x++) { reArr[y * n + x] = rowRe[x]; imArr[y * n + x] = rowIm[x]; }
      }
      // Columns
      const colRe = new Float32Array(n), colIm = new Float32Array(n);
      for (let x = 0; x < n; x++) {
        for (let y = 0; y < n; y++) { colRe[y] = reArr[y * n + x]; colIm[y] = imArr[y * n + x]; }
        fft1d(colRe, colIm, n, inverse);
        for (let y = 0; y < n; y++) { reArr[y * n + x] = colRe[y]; imArr[y * n + x] = colIm[y]; }
      }
    }

    function fftshift(arr, n) {
      const half = n / 2;
      const tmp = new Float32Array(n * n);
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const sy = (y + half) % n, sx = (x + half) % n;
          tmp[sy * n + sx] = arr[y * n + x];
        }
      }
      arr.set(tmp);
    }

    // Compute full k-space
    const ksRe = new Float32Array(phantom);
    const ksIm = new Float32Array(N * N);
    fft2d(ksRe, ksIm, N, false);
    fftshift(ksRe, N);
    fftshift(ksIm, N);

    // Animation state
    let currentLine = 0;
    let lastTime = null;
    const linesPerSec = 40;

    // Masked k-space for progressive fill
    const maskRe = new Float32Array(N * N);
    const maskIm = new Float32Array(N * N);

    // ImageData buffers
    const kspaceImg = ctx.createImageData(N, N);
    const reconImg = ctx.createImageData(N, N);
    const phantomImg = ctx.createImageData(N, N);

    // Pre-render phantom image
    for (let i = 0; i < N * N; i++) {
      const v = Math.min(255, phantom[i] * 255);
      phantomImg.data[i * 4] = v;
      phantomImg.data[i * 4 + 1] = v;
      phantomImg.data[i * 4 + 2] = v;
      phantomImg.data[i * 4 + 3] = 255;
    }

    let animId;

    function animate(timestamp) {
      animId = requestAnimationFrame(animate);

      if (!lastTime) lastTime = timestamp;
      const dtMs = Math.min(timestamp - lastTime, 100);
      lastTime = timestamp;
      const dt = dtMs / 1000;

      const speed = model.get("speed");
      const maskType = model.get("mask_type");

      // Progressive fill: add lines over time
      const linesToAdd = Math.ceil(linesPerSec * speed * dt);
      for (let l = 0; l < linesToAdd && currentLine < N; l++) {
        // Fill line in k-space (EPI-style: zigzag)
        const y = currentLine;
        for (let x = 0; x < N; x++) {
          maskRe[y * N + x] = ksRe[y * N + x];
          maskIm[y * N + x] = ksIm[y * N + x];
        }
        currentLine++;
      }

      // If mask_type is not "progressive", apply a static mask
      let dispRe, dispIm;
      if (maskType === "progressive") {
        dispRe = new Float32Array(maskRe);
        dispIm = new Float32Array(maskIm);
      } else {
        dispRe = new Float32Array(N * N);
        dispIm = new Float32Array(N * N);
        const half = N / 2;
        const rFrac = model.get("radius_fraction");
        const maxR = Math.sqrt(half * half + half * half);
        const r = rFrac * maxR;
        for (let y = 0; y < N; y++) {
          for (let x = 0; x < N; x++) {
            const dist = Math.sqrt((x - half) ** 2 + (y - half) ** 2);
            let keep = false;
            if (maskType === "center") keep = dist <= r;
            else if (maskType === "periphery") keep = dist > r;
            else if (maskType === "undersampled") keep = y % 4 === 0;
            else keep = true; // full
            if (keep) {
              dispRe[y * N + x] = ksRe[y * N + x];
              dispIm[y * N + x] = ksIm[y * N + x];
            }
          }
        }
      }

      // Reconstruct image from masked k-space
      const recRe = new Float32Array(dispRe);
      const recIm = new Float32Array(dispIm);
      fftshift(recRe, N);
      fftshift(recIm, N);
      fft2d(recRe, recIm, N, true);

      // Render k-space display (log magnitude)
      let maxKs = 0;
      for (let i = 0; i < N * N; i++) {
        const m = Math.log1p(Math.sqrt(dispRe[i] ** 2 + dispIm[i] ** 2));
        if (m > maxKs) maxKs = m;
      }
      for (let i = 0; i < N * N; i++) {
        const m = Math.log1p(Math.sqrt(dispRe[i] ** 2 + dispIm[i] ** 2));
        const v = maxKs > 0 ? Math.min(255, (m / maxKs) * 255) : 0;
        // Viridis-ish colormap
        kspaceImg.data[i * 4] = Math.min(255, v * 0.3);
        kspaceImg.data[i * 4 + 1] = Math.min(255, v * 0.7);
        kspaceImg.data[i * 4 + 2] = Math.min(255, v * 1.2);
        kspaceImg.data[i * 4 + 3] = 255;
      }

      // Render reconstructed image
      let maxRec = 0;
      for (let i = 0; i < N * N; i++) {
        const m = Math.sqrt(recRe[i] ** 2 + recIm[i] ** 2);
        if (m > maxRec) maxRec = m;
      }
      for (let i = 0; i < N * N; i++) {
        const m = Math.sqrt(recRe[i] ** 2 + recIm[i] ** 2);
        const v = maxRec > 0 ? Math.min(255, (m / maxRec) * 255) : 0;
        reconImg.data[i * 4] = v;
        reconImg.data[i * 4 + 1] = v;
        reconImg.data[i * 4 + 2] = v;
        reconImg.data[i * 4 + 3] = 255;
      }

      // Reset if filled
      if (maskType === "progressive" && currentLine >= N) {
        currentLine = 0;
        maskRe.fill(0);
        maskIm.fill(0);
      }

      draw(ctx, W, H, phantomImg, kspaceImg, reconImg, N, currentLine, maskType);
    }

    function draw(ctx, w, h, phImg, ksImg, rcImg, n, line, maskType) {
      ctx.fillStyle = "#fafafa";
      ctx.fillRect(0, 0, w, h);

      const imgSize = Math.min(h - 60, (w - 60) / 3);
      const pad = (w - imgSize * 3 - 20) / 2;
      const top = 40;

      const titles = ["Original Image", "K-Space", "Reconstructed Image"];
      const xPositions = [pad, pad + imgSize + 10, pad + imgSize * 2 + 20];

      // Create temp canvases for scaling
      for (let i = 0; i < 3; i++) {
        const img = [phImg, ksImg, rcImg][i];
        const tc = document.createElement("canvas");
        tc.width = n; tc.height = n;
        tc.getContext("2d").putImageData(img, 0, 0);
        ctx.drawImage(tc, xPositions[i], top, imgSize, imgSize);

        // Border
        ctx.strokeStyle = "#ccc";
        ctx.lineWidth = 1;
        ctx.strokeRect(xPositions[i], top, imgSize, imgSize);

        // Title
        ctx.fillStyle = "#333";
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "center";
        ctx.fillText(titles[i], xPositions[i] + imgSize / 2, top - 8);
      }

      // Progress indicator for progressive mode
      if (maskType === "progressive") {
        const prog = line / n;
        const barX = xPositions[1];
        const barY = top + imgSize + 8;
        ctx.fillStyle = "#ddd";
        ctx.fillRect(barX, barY, imgSize, 6);
        ctx.fillStyle = "#4488cc";
        ctx.fillRect(barX, barY, imgSize * prog, 6);

        // Scanning line indicator on k-space image
        const lineY = top + (line / n) * imgSize;
        ctx.strokeStyle = "#ff4444";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xPositions[1], lineY);
        ctx.lineTo(xPositions[1] + imgSize, lineY);
        ctx.stroke();

        ctx.fillStyle = "#666";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`Acquiring line ${line}/${n}`, xPositions[1] + imgSize / 2, barY + 20);
      }

      ctx.textAlign = "left";
      ctx.fillStyle = "#888";
      ctx.font = "10px Arial";
      ctx.fillText(`Mode: ${maskType}`, 10, h - 8);
    }

    animId = requestAnimationFrame(animate);

    model.on("change:mask_type", () => {
      currentLine = 0;
      maskRe.fill(0);
      maskIm.fill(0);
    });

    return () => {
      cancelAnimationFrame(animId);
      model.off("change:mask_type");
    };
  },
};
