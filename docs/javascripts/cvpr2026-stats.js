(function () {
  const data = {
    sourceUrl: "https://paperswithcode.co/conferences/cvpr-2026",
    apiUrl: "https://paperswithcode.co/api/v1/conferences/cvpr-2026/areas-with-tasks",
    parsedDate: "2026-06-08",
    totalPapers: 2676,
    oralPapers: 93,
    spotlightPapers: 380,
    bestPaperFinalists: 15,
    areas: [
      {
        name: "Vision",
        count: 1761,
        color: "#275f5a",
        tasks: [
          { name: "Image Understanding", count: 630 },
          { name: "3D generation", count: 220 },
          { name: "Image generation", count: 151 },
          { name: "3D understanding", count: 136 },
          { name: "Image Classification", count: 132 },
          { name: "Image segmentation", count: 81 },
          { name: "Object Detection", count: 80 },
          { name: "Pose Estimation", count: 72 },
          { name: "Image editing", count: 66 },
          { name: "Depth estimation", count: 58 },
          { name: "Medical Imaging", count: 38 },
          { name: "Image Restoration", count: 34 },
          { name: "Image Matching", count: 22 },
          { name: "Image super-resolution", count: 18 },
          { name: "SLAM and Localization", count: 9 },
          { name: "Stereo Matching", count: 7 },
          { name: "3D semantic segmentation", count: 4 },
          { name: "3D instance segmentation", count: 2 },
          { name: "Optical Flow", count: 1 },
        ],
      },
      {
        name: "General",
        count: 1236,
        color: "#4f7770",
        tasks: [
          { name: "Language Modeling", count: 700 },
          { name: "Reinforcement Learning", count: 178 },
          { name: "Reasoning", count: 105 },
          { name: "Robotics", count: 65 },
          { name: "Remote Sensing", count: 38 },
          { name: "Anomaly Detection", count: 26 },
          { name: "Autonomous Driving", count: 23 },
          { name: "Embedding models", count: 22 },
          { name: "OCR", count: 19 },
          { name: "Deepfake and Forensics", count: 15 },
          { name: "Computer Use Agents", count: 11 },
          { name: "Agents", count: 10 },
          { name: "Omni models", count: 9 },
          { name: "Document Understanding", count: 8 },
          { name: "World Models", count: 4 },
          { name: "Scene Text Recognition", count: 3 },
        ],
      },
      {
        name: "Video",
        count: 304,
        color: "#b66f2d",
        tasks: [
          { name: "Video generation", count: 167 },
          { name: "Video classification", count: 91 },
          { name: "Video Understanding", count: 15 },
          { name: "Video segmentation", count: 13 },
          { name: "Object Tracking", count: 12 },
          { name: "Video super-resolution", count: 6 },
        ],
      },
      {
        name: "Language",
        count: 102,
        color: "#6d6f8c",
        tasks: [
          { name: "Question Answering", count: 98 },
          { name: "Summarization", count: 3 },
          { name: "Machine Translation", count: 1 },
        ],
      },
      {
        name: "Audio",
        count: 12,
        color: "#60785c",
        tasks: [
          { name: "Automatic Speech Recognition", count: 8 },
          { name: "Audio understanding", count: 2 },
          { name: "Audio Classification", count: 1 },
          { name: "Text-to-speech", count: 1 },
        ],
      },
      {
        name: "Other",
        count: 1,
        color: "#8b5f61",
        tasks: [{ name: "Tabular Learning", count: 1 }],
      },
    ],
  };
  function formatNumber(value) {
    return value.toLocaleString("en-US");
  }

  function formatLabelCount(value) {
    return `${formatNumber(value)} ${value === 1 ? "paper label" : "paper labels"}`;
  }

  function createElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) {
      element.className = className;
    }
    if (text !== undefined) {
      element.textContent = text;
    }
    return element;
  }

  function appendText(parent, tagName, className, text) {
    const element = createElement(tagName, className, text);
    parent.appendChild(element);
    return element;
  }

  function renderSummary(root, assignmentTotal, taskTotal) {
    const summary = createElement("div", "cvpr-stats-summary");
    const items = [
      ["Accepted papers", formatNumber(data.totalPapers)],
      ["Area-task labels", formatNumber(assignmentTotal)],
      ["Tasks", formatNumber(taskTotal)],
      ["Oral / Spotlight / Finalist", `${data.oralPapers} / ${data.spotlightPapers} / ${data.bestPaperFinalists}`],
    ];

    items.forEach(([label, value]) => {
      const card = createElement("div", "cvpr-stats-metric");
      appendText(card, "strong", null, value);
      appendText(card, "span", null, label);
      summary.appendChild(card);
    });

    root.appendChild(summary);
  }

  function renderBarChart(options) {
    const chart = createElement("figure", `cvpr-stat-chart ${options.className || ""}`.trim());
    const header = createElement("div", "cvpr-stat-chart-header");
    appendText(header, "h3", null, options.title);
    if (options.description) {
      appendText(header, "p", null, options.description);
    }
    chart.appendChild(header);

    const bars = createElement("div", "cvpr-stat-bars");
    if (options.selectable) {
      bars.setAttribute("role", "group");
      bars.setAttribute("aria-label", options.selectionLabel || options.title);
    } else {
      bars.setAttribute("role", "list");
    }

    options.bars.forEach((bar) => {
      const percentOfMax = options.max > 0 ? (bar.count / options.max) * 100 : 0;
      const percentOfTotal = options.total > 0 ? (bar.count / options.total) * 100 : 0;
      const row = createElement(options.selectable ? "button" : "div", "cvpr-stat-bar-row");
      if (options.selectable) {
        row.type = "button";
        row.classList.add("is-selectable");
        row.dataset.category = bar.name;
        row.setAttribute("aria-pressed", String(bar.name === options.selectedName));
        if (bar.name === options.selectedName) {
          row.classList.add("is-selected");
        }
        row.addEventListener("click", () => options.onSelect(bar));
      } else {
        row.setAttribute("role", "listitem");
      }
      row.setAttribute(
        "aria-label",
        options.showShare
          ? `${bar.name}: ${formatLabelCount(bar.count)}, ${percentOfTotal.toFixed(1)} percent of labels`
          : `${bar.name}: ${formatLabelCount(bar.count)}`,
      );

      appendText(row, "span", "cvpr-stat-label", bar.name);
      const track = createElement("span", "cvpr-stat-track");
      const fill = createElement("span", "cvpr-stat-fill");
  fill.style.width = `${Math.max(percentOfMax, 0.8)}%`;
      fill.style.backgroundColor = bar.color || options.color || "#275f5a";
      track.appendChild(fill);
      row.appendChild(track);

      const valueText = options.showShare
        ? `${formatNumber(bar.count)} · ${percentOfTotal.toFixed(1)}%`
        : formatNumber(bar.count);
      appendText(row, "span", "cvpr-stat-value", valueText);
      bars.appendChild(row);
    });

    chart.appendChild(bars);
    return chart;
  }

  function renderInteractiveCharts(root, assignmentTotal) {
    let selectedArea = null;
    let detailHasEnteredViewport = false;
    let scrollCheckQueued = false;
    let selectedAtScrollY = 0;
    const wrapper = createElement("div", "cvpr-stat-interactive");
    const areaHost = createElement("div", "cvpr-stat-area-host");
    const detailHost = createElement("div", "cvpr-stat-detail-host");
    detailHost.hidden = true;
    detailHost.setAttribute("aria-live", "polite");

    function updateAreaButtons() {
      areaHost.querySelectorAll(".cvpr-stat-bar-row.is-selectable").forEach((button) => {
        const isSelected = selectedArea !== null && button.dataset.category === selectedArea.name;
        button.classList.toggle("is-selected", isSelected);
        button.setAttribute("aria-pressed", String(isSelected));
      });
    }

    function detailIsInViewport() {
      const rect = detailHost.getBoundingClientRect();
      return rect.bottom > 0 && rect.top < window.innerHeight;
    }

    function clearDetail() {
      selectedArea = null;
      detailHasEnteredViewport = false;
      selectedAtScrollY = 0;
      detailHost.textContent = "";
      detailHost.hidden = true;
      updateAreaButtons();
    }

    function checkScrollAway() {
      if (selectedArea === null || detailHost.hidden || scrollCheckQueued) {
        return;
      }
      scrollCheckQueued = true;
      window.requestAnimationFrame(() => {
        scrollCheckQueued = false;
        if (selectedArea === null || detailHost.hidden) {
          return;
        }
        const isVisible = detailIsInViewport();
        const scrolledSinceSelection = Math.abs(window.scrollY - selectedAtScrollY) > 40;
        detailHasEnteredViewport = detailHasEnteredViewport || isVisible;
        if (detailHasEnteredViewport && scrolledSinceSelection && !isVisible) {
          clearDetail();
        }
      });
    }

    function updateDetail() {
      if (selectedArea === null) {
        clearDetail();
        return;
      }
      detailHost.textContent = "";
      detailHost.hidden = false;
      detailHost.appendChild(
        renderBarChart({
          title: `${selectedArea.name} Task Breakdown`,
          description: `${formatLabelCount(selectedArea.count)} across ${selectedArea.tasks.length} ${selectedArea.tasks.length === 1 ? "task" : "tasks"}.`,
          bars: selectedArea.tasks.map((task) => ({ ...task, color: selectedArea.color })),
          max: Math.max(...selectedArea.tasks.map((task) => task.count)),
          total: selectedArea.count,
          showShare: false,
          color: selectedArea.color,
        }),
      );
      detailHasEnteredViewport = detailIsInViewport();
    }

    function selectArea(area) {
      selectedArea = data.areas.find((candidate) => candidate.name === area.name) || selectedArea;
      updateAreaButtons();
      updateDetail();
      selectedAtScrollY = window.scrollY;
    }

    areaHost.appendChild(
      renderBarChart({
        title: "Overall Area Distribution",
        description: "Paper counts by Papers with Code area label.",
        bars: data.areas,
        max: Math.max(...data.areas.map((area) => area.count)),
        total: assignmentTotal,
        showShare: true,
        selectable: true,
        selectedName: selectedArea?.name,
        selectionLabel: "CVPR 2026 research areas",
        onSelect: selectArea,
        className: "cvpr-stat-chart-wide",
      }),
    );

    wrapper.append(areaHost, detailHost);
    root.appendChild(wrapper);
    window.addEventListener("scroll", checkScrollAway, { passive: true });
    window.addEventListener("resize", checkScrollAway);
  }

  function renderSource(root) {
    const source = createElement("p", "cvpr-stats-source");
    source.append("Data parsed from ");
    const link = createElement("a", null, "Papers with Code CVPR 2026");
    link.href = data.sourceUrl;
    link.rel = "noopener";
    source.appendChild(link);
    source.append(` on ${data.parsedDate}. Area and task counts are Papers with Code labels, not a mutually exclusive partition; a paper can appear under more than one task.`);
    root.appendChild(source);
  }

  function render() {
    const root = document.getElementById("cvpr2026-statistics");
    if (!root || root.dataset.rendered === "true") {
      return;
    }
    root.dataset.rendered = "true";
    root.textContent = "";

    const assignmentTotal = data.areas.reduce((sum, area) => sum + area.count, 0);
    const taskTotal = data.areas.reduce((sum, area) => sum + area.tasks.length, 0);
    renderSummary(root, assignmentTotal, taskTotal);

    renderInteractiveCharts(root, assignmentTotal);
    renderSource(root);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render, { once: true });
  } else {
    render();
  }
})();
