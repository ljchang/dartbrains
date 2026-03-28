// Smoothing Widget - Canvas 2D
// Interactive Gaussian spatial smoothing demonstration

export default {
  render({ model, el }) {
    const W = 800;
    const H = 320;
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

    // Create synthetic brain phantom (128x128)
    const N = 128;
    const phantom = new Float32Array(N * N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const cx = N / 2, cy = N / 2;
        const dx = (x - cx) / 45, dy = (y - cy) / 55;
        if (dx * dx + dy * dy < 1) phantom[y * N + x] = 0.3;
        const bx = (x - cx) / 38, by = (y - cy) / 48;
        if (bx * bx + by * by < 1) phantom[y * N + x] = 0.7;
        const vl = ((x - cx + 10) / 8) ** 2 + ((y - cy) / 18) ** 2;
        const vr = ((x - cx - 10) / 8) ** 2 + ((y - cy) / 18) ** 2;
        if (vl < 1 || vr < 1) phantom[y * N + x] = 1.0;
        const co = ((x - cx) / 36) ** 2 + ((y - cy) / 46) ** 2;
        const ci = ((x - cx) / 28) ** 2 + ((y - cy) / 38) ** 2;
        if (co < 1 && ci >= 1 && bx * bx + by * by < 1) phantom[y * N + x] = 0.85;
        // Add some noise-like texture
        if (phantom[y * N + x] > 0) {
          phantom[y * N + x] += (Math.sin(x * 1.7 + y * 2.3) * 0.05 +
                                  Math.sin(x * 3.1 - y * 1.9) * 0.03);
        }
      }
    }

    // Gaussian smoothing (separable 2D convolution)
    function gaussianSmooth(input, n, sigma) {
      if (sigma < 0.5) return new Float32Array(input);

      // Build 1D kernel
      const kRadius = Math.ceil(sigma * 3);
      const kSize = kRadius * 2 + 1;
      const kernel = new Float32Array(kSize);
      let sum = 0;
      for (let i = 0; i < kSize; i++) {
        const x = i - kRadius;
        kernel[i] = Math.exp(-x * x / (2 * sigma * sigma));
        sum += kernel[i];
      }
      for (let i = 0; i < kSize; i++) kernel[i] /= sum;

      // Horizontal pass
      const tmp = new Float32Array(n * n);
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          let val = 0;
          for (let k = 0; k < kSize; k++) {
            const sx = Math.min(Math.max(x + k - kRadius, 0), n - 1);
            val += input[y * n + sx] * kernel[k];
          }
          tmp[y * n + x] = val;
        }
      }

      // Vertical pass
      const output = new Float32Array(n * n);
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          let val = 0;
          for (let k = 0; k < kSize; k++) {
            const sy = Math.min(Math.max(y + k - kRadius, 0), n - 1);
            val += tmp[sy * n + x] * kernel[k];
          }
          output[y * n + x] = val;
        }
      }
      return output;
    }

    let animId;
    let prevFwhm = -1;
    let smoothedData = null;
    const phantomImgData = ctx.createImageData(N, N);
    const smoothImgData = ctx.createImageData(N, N);

    // Pre-render phantom
    for (let i = 0; i < N * N; i++) {
      const v = Math.min(255, Math.max(0, phantom[i] * 255));
      phantomImgData.data[i * 4] = v;
      phantomImgData.data[i * 4 + 1] = v;
      phantomImgData.data[i * 4 + 2] = v;
      phantomImgData.data[i * 4 + 3] = 255;
    }

    function animate() {
      animId = requestAnimationFrame(animate);

      const fwhm = model.get("fwhm");
      const sigma = fwhm / 2.355; // FWHM = 2.355 * sigma

      // Only recompute if FWHM changed
      if (fwhm !== prevFwhm) {
        prevFwhm = fwhm;
        smoothedData = gaussianSmooth(phantom, N, sigma);

        for (let i = 0; i < N * N; i++) {
          const v = Math.min(255, Math.max(0, smoothedData[i] * 255));
          smoothImgData.data[i * 4] = v;
          smoothImgData.data[i * 4 + 1] = v;
          smoothImgData.data[i * 4 + 2] = v;
          smoothImgData.data[i * 4 + 3] = 255;
        }
      }

      draw(ctx, W, H, phantomImgData, smoothImgData, N, fwhm, sigma);
    }

    function draw(ctx, w, h, origImg, smoothImg, n, fwhm, sigma) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fafafa";
      ctx.fillRect(0, 0, w, h);

      const imgSize = h - 60;
      const gap = 20;
      const startX = 20;

      // Original image
      const tc1 = document.createElement("canvas");
      tc1.width = n; tc1.height = n;
      tc1.getContext("2d").putImageData(origImg, 0, 0);
      ctx.drawImage(tc1, startX, 40, imgSize, imgSize);
      ctx.strokeStyle = "#ccc";
      ctx.strokeRect(startX, 40, imgSize, imgSize);
      ctx.fillStyle = "#333";
      ctx.font = "bold 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Original", startX + imgSize / 2, 30);

      // Kernel visualization (2D Gaussian)
      const kernX = startX + imgSize + gap;
      const kernSize = 100;
      const kernCanvas = document.createElement("canvas");
      kernCanvas.width = kernSize; kernCanvas.height = kernSize;
      const kCtx = kernCanvas.getContext("2d");
      const kImg = kCtx.createImageData(kernSize, kernSize);
      for (let y = 0; y < kernSize; y++) {
        for (let x = 0; x < kernSize; x++) {
          const dx = (x - kernSize / 2) / (kernSize / 2);
          const dy = (y - kernSize / 2) / (kernSize / 2);
          const sigNorm = Math.max(sigma / 15, 0.05);
          const v = Math.exp(-(dx * dx + dy * dy) / (2 * sigNorm * sigNorm));
          const c = Math.min(255, v * 255);
          const idx = (y * kernSize + x) * 4;
          kImg.data[idx] = 255;
          kImg.data[idx + 1] = Math.floor(255 - c * 0.7);
          kImg.data[idx + 2] = Math.floor(255 - c);
          kImg.data[idx + 3] = 255;
        }
      }
      kCtx.putImageData(kImg, 0, 0);
      const kernY = 40 + (imgSize - kernSize) / 2;
      ctx.drawImage(kernCanvas, kernX, kernY, kernSize, kernSize);
      ctx.strokeStyle = "#ccc";
      ctx.strokeRect(kernX, kernY, kernSize, kernSize);
      ctx.fillStyle = "#333";
      ctx.font = "bold 11px Arial";
      ctx.fillText("Kernel", kernX + kernSize / 2, kernY - 8);
      ctx.font = "10px Arial";
      ctx.fillStyle = "#666";
      ctx.fillText(`FWHM=${fwhm.toFixed(1)}mm`, kernX + kernSize / 2, kernY + kernSize + 14);
      ctx.fillText(`\u03c3=${sigma.toFixed(1)}mm`, kernX + kernSize / 2, kernY + kernSize + 28);

      // Arrow
      ctx.fillStyle = "#999";
      ctx.font = "24px Arial";
      ctx.fillText("\u2192", kernX + kernSize + 8, 40 + imgSize / 2 + 8);

      // Smoothed image
      const smoothX = kernX + kernSize + gap + 25;
      const tc2 = document.createElement("canvas");
      tc2.width = n; tc2.height = n;
      tc2.getContext("2d").putImageData(smoothImg, 0, 0);
      ctx.drawImage(tc2, smoothX, 40, imgSize, imgSize);
      ctx.strokeStyle = "#ccc";
      ctx.strokeRect(smoothX, 40, imgSize, imgSize);
      ctx.fillStyle = "#333";
      ctx.font = "bold 12px Arial";
      ctx.fillText(`Smoothed (FWHM=${fwhm.toFixed(1)})`, smoothX + imgSize / 2, 30);

      ctx.textAlign = "left";
    }

    animId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animId);
    };
  },
};
