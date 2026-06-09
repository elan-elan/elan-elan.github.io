(() => {
  const root = document.getElementById("spiral-diffusion-demo");
  if (!root) return;

  const modes = [
    {
      id: "noise",
      label: "Noising",
      caption: "Forward corruption: a clean spiral is gradually mixed with a Gaussian cloud."
    },
    {
      id: "flow",
      label: "Flow",
      caption: "Few-step transport: particles move from noise toward the spiral along a learned-map style path."
    },
    {
      id: "tokens",
      label: "Tokens",
      caption: "Discrete denoising: coordinate tokens hop between bins until the spiral code is recovered."
    }
  ];

  root.innerHTML = `
    <div class="diffusion-toy-header">
      <div class="diffusion-toy-copy">
        <strong>Spiral Toy Dataset</strong>
        <span data-demo-caption>${modes[0].caption}</span>
      </div>
      <div class="diffusion-toy-controls" role="group" aria-label="Spiral demo mode">
        ${modes.map((mode, index) => `<button type="button" data-demo-mode="${mode.id}" class="${index === 0 ? "is-active" : ""}">${mode.label}</button>`).join("")}
        <button type="button" data-demo-pause aria-pressed="false">Pause</button>
      </div>
    </div>
    <canvas class="diffusion-toy-canvas" width="960" height="540" aria-label="Animated spiral point cloud"></canvas>
    <div class="diffusion-toy-legend" aria-hidden="true">
      <span><i class="legend-target"></i>target spiral</span>
      <span><i class="legend-state"></i>current state</span>
      <span><i class="legend-guide"></i>path hint</span>
    </div>
  `;

  const canvas = root.querySelector("canvas");
  const caption = root.querySelector("[data-demo-caption]");
  const buttons = Array.from(root.querySelectorAll("[data-demo-mode]"));
  const pauseButton = root.querySelector("[data-demo-pause]");
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let mode = "noise";
  let paused = reduceMotion;
  let frameId = null;
  let lastTime = 0;

  const palette = ["#2f6f7e", "#b13f5b", "#7254a8", "#b8732d", "#2f7d4f"];
  const points = makeSpiralPoints(360);

  pauseButton.textContent = paused ? "Play" : "Pause";
  pauseButton.setAttribute("aria-pressed", String(paused));

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      mode = button.dataset.demoMode;
      const nextMode = modes.find((item) => item.id === mode);
      caption.textContent = nextMode.caption;
      buttons.forEach((item) => item.classList.toggle("is-active", item === button));
      draw(lastTime || performance.now());
    });
  });

  pauseButton.addEventListener("click", () => {
    paused = !paused;
    pauseButton.textContent = paused ? "Play" : "Pause";
    pauseButton.setAttribute("aria-pressed", String(paused));
    if (!paused) frameId = requestAnimationFrame(loop);
  });

  const resizeObserver = new ResizeObserver(() => draw(lastTime || performance.now()));
  resizeObserver.observe(root);

  draw(performance.now());
  if (!paused) frameId = requestAnimationFrame(loop);

  function loop(time) {
    lastTime = time;
    draw(time);
    if (!paused) frameId = requestAnimationFrame(loop);
  }

  window.addEventListener("pagehide", () => {
    if (frameId) cancelAnimationFrame(frameId);
  });

  function makeSpiralPoints(count) {
    const random = mulberry32(202606);
    const values = [];
    const grid = 18;
    const gridMin = -1.15;
    const gridSpan = 2.3;
    const step = gridSpan / (grid - 1);

    for (let i = 0; i < count; i += 1) {
      const u = i / Math.max(count - 1, 1);
      const theta = 0.45 + u * Math.PI * 5.25 + gaussian(random) * 0.045;
      const radius = 0.13 + 0.86 * u;
      const jitter = 0.026 + 0.016 * u;
      const target = {
        x: radius * Math.cos(theta) + gaussian(random) * jitter,
        y: radius * Math.sin(theta) + gaussian(random) * jitter
      };
      const noise = {
        x: clamp(gaussian(random) * 0.62, -1.22, 1.22),
        y: clamp(gaussian(random) * 0.62, -1.22, 1.22)
      };
      const tokenStart = {
        x: gridMin + Math.floor(random() * grid) * step,
        y: gridMin + Math.floor(random() * grid) * step
      };
      const tokenEnd = {
        x: gridMin + Math.round((target.x - gridMin) / step) * step,
        y: gridMin + Math.round((target.y - gridMin) / step) * step
      };
      values.push({
        target,
        noise,
        tokenStart,
        tokenEnd,
        theta,
        color: palette[Math.floor((theta / (Math.PI * 2)) * palette.length) % palette.length]
      });
    }

    return values;
  }

  function draw(timeMs) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(320, Math.floor(rect.width * dpr));
    const height = Math.max(220, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.save();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fbfcfe";
    ctx.fillRect(0, 0, width, height);
    drawGrid(width, height);
    drawTargetGhost(width, height);

    const time = timeMs / 1000;
    if (mode === "noise") drawNoising(width, height, time);
    if (mode === "flow") drawFlow(width, height, time);
    if (mode === "tokens") drawTokens(width, height, time);
    ctx.restore();
  }

  function drawGrid(width, height) {
    ctx.save();
    ctx.strokeStyle = "rgba(72, 85, 99, 0.12)";
    ctx.lineWidth = Math.max(1, width / 900);
    for (let v = -1; v <= 1.001; v += 0.5) {
      const horizontalA = project({ x: -1.25, y: v }, width, height);
      const horizontalB = project({ x: 1.25, y: v }, width, height);
      const verticalA = project({ x: v, y: -1.25 }, width, height);
      const verticalB = project({ x: v, y: 1.25 }, width, height);
      ctx.beginPath();
      ctx.moveTo(horizontalA.x, horizontalA.y);
      ctx.lineTo(horizontalB.x, horizontalB.y);
      ctx.moveTo(verticalA.x, verticalA.y);
      ctx.lineTo(verticalB.x, verticalB.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawTargetGhost(width, height) {
    ctx.save();
    ctx.globalAlpha = 0.18;
    points.forEach((point) => {
      const p = project(point.target, width, height);
      ctx.fillStyle = "#26364a";
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(1.15, width / 560), 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  function drawNoising(width, height, time) {
    const progress = 0.5 - 0.5 * Math.cos(time * 0.85);
    const eased = smooth(progress);
    points.forEach((point, index) => {
      const local = clamp(eased + Math.sin(index * 0.11 + time * 1.2) * 0.035, 0, 1);
      const current = mix(point.target, point.noise, local);
      drawDot(project(current, width, height), point.color, 2.2 + local * 1.5, 0.9 - local * 0.2);
    });
    drawMeter(width, height, progress, "clean", "noise");
  }

  function drawFlow(width, height, time) {
    const progress = (time * 0.16) % 1;
    const eased = smooth(progress);

    points.forEach((point, index) => {
      if (index % 18 === 0) drawFlowPath(point, width, height, eased);
    });

    points.forEach((point, index) => {
      const current = flowPosition(point, clamp(eased + Math.sin(index * 0.07) * 0.018, 0, 1));
      drawDot(project(current, width, height), point.color, 2.45, 0.9);
    });
    drawMeter(width, height, progress, "noise", "spiral");
  }

  function drawFlowPath(point, width, height, progress) {
    ctx.save();
    ctx.strokeStyle = "rgba(47, 111, 126, 0.22)";
    ctx.lineWidth = Math.max(1.2, width / 620);
    ctx.beginPath();
    for (let i = 0; i <= 26; i += 1) {
      const t = progress * (i / 26);
      const p = project(flowPosition(point, t), width, height);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function flowPosition(point, progress) {
    const base = mix(point.noise, point.target, smooth(progress));
    const dx = point.target.x - point.noise.x;
    const dy = point.target.y - point.noise.y;
    const norm = Math.hypot(dx, dy) || 1;
    const bend = Math.sin(progress * Math.PI) * (1 - progress * 0.35) * 0.18;
    return {
      x: base.x + (-dy / norm) * bend,
      y: base.y + (dx / norm) * bend
    };
  }

  function drawTokens(width, height, time) {
    const progress = (time * 0.14) % 1;
    const discrete = Math.floor(smooth(progress) * 9) / 8;
    drawTokenGrid(width, height);
    points.forEach((point, index) => {
      const jitter = index % 5 === 0 ? 0.02 * Math.sin(time * 2 + index) : 0;
      const current = mix(point.tokenStart, point.tokenEnd, clamp(discrete + jitter, 0, 1));
      drawSquare(project(current, width, height), point.color, 4.2, 0.78);
    });
    drawMeter(width, height, progress, "random bins", "spiral bins");
  }

  function drawTokenGrid(width, height) {
    ctx.save();
    ctx.strokeStyle = "rgba(177, 63, 91, 0.09)";
    ctx.lineWidth = Math.max(0.8, width / 1200);
    for (let v = -1.15; v <= 1.151; v += 2.3 / 17) {
      const horizontalA = project({ x: -1.15, y: v }, width, height);
      const horizontalB = project({ x: 1.15, y: v }, width, height);
      const verticalA = project({ x: v, y: -1.15 }, width, height);
      const verticalB = project({ x: v, y: 1.15 }, width, height);
      ctx.beginPath();
      ctx.moveTo(horizontalA.x, horizontalA.y);
      ctx.lineTo(horizontalB.x, horizontalB.y);
      ctx.moveTo(verticalA.x, verticalA.y);
      ctx.lineTo(verticalB.x, verticalB.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawMeter(width, height, progress, leftLabel, rightLabel) {
    const meterWidth = Math.min(width * 0.42, 320 * (window.devicePixelRatio || 1));
    const meterHeight = Math.max(8, height * 0.018);
    const x = width / 2 - meterWidth / 2;
    const y = height - Math.max(30, height * 0.08);
    ctx.save();
    ctx.fillStyle = "rgba(38, 54, 74, 0.1)";
    roundedRect(ctx, x, y, meterWidth, meterHeight, meterHeight / 2);
    ctx.fill();
    ctx.fillStyle = "#2f6f7e";
    if (progress > 0.002) {
      roundedRect(ctx, x, y, meterWidth * progress, meterHeight, meterHeight / 2);
      ctx.fill();
    }
    ctx.fillStyle = "#5d6878";
    ctx.font = `${Math.max(10, width / 78)}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
    ctx.textAlign = "right";
    ctx.fillText(leftLabel, x - 10, y + meterHeight * 1.1);
    ctx.textAlign = "left";
    ctx.fillText(rightLabel, x + meterWidth + 10, y + meterHeight * 1.1);
    ctx.restore();
  }

  function drawDot(p, color, radius, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawSquare(p, color, size, alpha) {
    const dpr = window.devicePixelRatio || 1;
    const side = size * dpr;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.fillRect(p.x - side / 2, p.y - side / 2, side, side);
    ctx.restore();
  }

  function project(point, width, height) {
    const scale = Math.min(width, height) * 0.36;
    return {
      x: width / 2 + point.x * scale,
      y: height / 2 - point.y * scale
    };
  }

  function mix(a, b, t) {
    return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
  }

  function smooth(t) {
    return t * t * (3 - 2 * t);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function roundedRect(context, x, y, width, height, radius) {
    const safeWidth = Math.max(width, radius * 2);
    context.beginPath();
    context.moveTo(x + radius, y);
    context.arcTo(x + safeWidth, y, x + safeWidth, y + height, radius);
    context.arcTo(x + safeWidth, y + height, x, y + height, radius);
    context.arcTo(x, y + height, x, y, radius);
    context.arcTo(x, y, x + safeWidth, y, radius);
    context.closePath();
  }

  function gaussian(random) {
    const u = 1 - random();
    const v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function mulberry32(seed) {
    return function random() {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
})();