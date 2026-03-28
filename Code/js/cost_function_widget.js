// Cost Function Widget - Canvas 2D
// Interactive image registration with SSE cost function

export default {
  render({ model, el }) {
    const W = 800;
    const H = 300;
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

    // Grid size for the images
    const N = 30;
    const SQ = 10; // square size (same for both)
    const REF_X = 10; // reference square top-left position
    const REF_Y = 10;

    // Reference image: square at (REF_X, REF_Y)
    const refImg = new Float32Array(N * N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (x >= REF_X && x < REF_X + SQ && y >= REF_Y && y < REF_Y + SQ) {
          refImg[y * N + x] = 1;
        }
      }
    }

    // SSE history for cost landscape
    const SSE_HISTORY = 100;
    const sseHistory = [];

    let animId;
    let lastTime = null;

    function animate(timestamp) {
      animId = requestAnimationFrame(animate);

      const tx = model.get("trans_x");
      const ty = model.get("trans_y");

      // Create target image: same-size square shifted by (tx, ty) from origin
      const targetImg = new Float32Array(N * N);
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const ox = x - Math.round(tx); // original coordinate
          const oy = y - Math.round(ty);
          if (ox >= 0 && ox < SQ && oy >= 0 && oy < SQ) {
            targetImg[y * N + x] = 1;
          }
        }
      }

      // Compute SSE
      let sse = 0;
      for (let i = 0; i < N * N; i++) {
        const d = targetImg[i] - refImg[i];
        sse += d * d;
      }

      // Record history
      sseHistory.push({ tx, ty, sse });
      if (sseHistory.length > SSE_HISTORY) sseHistory.shift();

      draw(ctx, W, H, targetImg, refImg, N, tx, ty, sse, sseHistory);
    }

    function drawImage(ctx, img, n, x0, y0, size, title) {
      const cellSize = size / n;
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const v = img[y * n + x];
          if (v > 0) {
            ctx.fillStyle = `rgba(70, 130, 200, ${v})`;
            ctx.fillRect(x0 + x * cellSize, y0 + y * cellSize, cellSize + 0.5, cellSize + 0.5);
          }
        }
      }
      // Grid border
      ctx.strokeStyle = "#ccc";
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, size, size);
      // Title
      ctx.fillStyle = "#333";
      ctx.font = "bold 11px Arial";
      ctx.textAlign = "center";
      ctx.fillText(title, x0 + size / 2, y0 - 8);
      ctx.textAlign = "left";
    }

    function draw(ctx, w, h, target, ref, n, tx, ty, sse, history) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fafafa";
      ctx.fillRect(0, 0, w, h);

      const imgSize = h - 60;
      const gap = 20;
      const startX = 15;

      // Target image
      drawImage(ctx, target, n, startX, 35, imgSize, "Target (moveable)");

      // Difference overlay
      const diffX = startX + imgSize + gap;
      // Draw difference image
      const cellSize = imgSize / n;
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const d = Math.abs(target[y * n + x] - ref[y * n + x]);
          if (d > 0) {
            ctx.fillStyle = `rgba(220, 60, 60, ${d})`;
            ctx.fillRect(diffX + x * cellSize, 35 + y * cellSize, cellSize + 0.5, cellSize + 0.5);
          }
          // Show reference in green where it's alone
          if (ref[y * n + x] > 0 && target[y * n + x] === 0) {
            ctx.fillStyle = `rgba(60, 180, 60, 0.4)`;
            ctx.fillRect(diffX + x * cellSize, 35 + y * cellSize, cellSize + 0.5, cellSize + 0.5);
          }
          // Overlap in yellow
          if (ref[y * n + x] > 0 && target[y * n + x] > 0) {
            ctx.fillStyle = `rgba(60, 200, 60, 0.8)`;
            ctx.fillRect(diffX + x * cellSize, 35 + y * cellSize, cellSize + 0.5, cellSize + 0.5);
          }
        }
      }
      ctx.strokeStyle = "#ccc";
      ctx.lineWidth = 1;
      ctx.strokeRect(diffX, 35, imgSize, imgSize);
      ctx.fillStyle = "#333";
      ctx.font = "bold 11px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Overlap (green=match, red=mismatch)", diffX + imgSize / 2, 27);
      ctx.textAlign = "left";

      // SSE display
      const sseX = diffX + imgSize + gap + 10;
      const sseW = w - sseX - 15;
      const sseH = imgSize;

      ctx.fillStyle = "#333";
      ctx.font = "bold 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Cost Function (SSE)", sseX + sseW / 2, 27);
      ctx.textAlign = "left";

      // SSE bar
      const maxSSE = 350;
      const barH = Math.min(sseH, (sse / maxSSE) * sseH);
      ctx.fillStyle = "#eee";
      ctx.fillRect(sseX, 35, sseW * 0.4, sseH);
      const barColor = sse < 1 ? "#22cc66" : sse < 100 ? "#ddaa22" : "#dd4444";
      ctx.fillStyle = barColor;
      ctx.fillRect(sseX, 35 + sseH - barH, sseW * 0.4, barH);
      ctx.strokeStyle = "#999";
      ctx.lineWidth = 1;
      ctx.strokeRect(sseX, 35, sseW * 0.4, sseH);

      // SSE value
      ctx.fillStyle = barColor;
      ctx.font = "bold 20px monospace";
      ctx.textAlign = "center";
      ctx.fillText(sse.toFixed(0), sseX + sseW * 0.7, 35 + sseH / 2 - 10);
      ctx.font = "11px Arial";
      ctx.fillStyle = "#888";
      ctx.fillText("SSE", sseX + sseW * 0.7, 35 + sseH / 2 + 10);
      ctx.textAlign = "left";

      // Parameters
      ctx.fillStyle = "#666";
      ctx.font = "10px monospace";
      ctx.fillText(`tx=${tx.toFixed(0)}, ty=${ty.toFixed(0)}`, sseX, 35 + sseH + 15);

      // Goal hint
      if (sse < 1) {
        ctx.fillStyle = "#22cc66";
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "center";
        ctx.fillText("\u2714 Perfect alignment!", sseX + sseW / 2, h - 5);
        ctx.textAlign = "left";
      }
    }

    animId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animId);
    };
  },
};
