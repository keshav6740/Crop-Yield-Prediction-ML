document.addEventListener("DOMContentLoaded", () => {
  const canAnimate = typeof window.anime === "function";
  const canChart = typeof window.Chart === "function";

  const runAnimation = (config) => {
    if (canAnimate) {
      window.anime(config);
    }
  };

  const form = document.getElementById("prediction-form");
  if (!form) {
    console.error("prediction-form not found; aborting UI script init.");
    return;
  }
  const loadingIndicator = document.getElementById("loading-indicator");
  const predictionResultSection = document.getElementById("prediction-result-section");
  const yieldValue = document.getElementById("yield-value");
  const confidenceIntervalDisplay = document.getElementById("confidence-interval");
  const yieldRange = document.getElementById("yield-range");
  const modelNameDisplay = document.getElementById("model-name-display");
  const featureContributionsSection = document.getElementById("feature-contributions-section");
  const featureContributionsList = document.getElementById("feature-contributions-list");
  const historicalDataSection = document.getElementById("historical-data-section");
  const historicalMessage = document.getElementById("historical-message");
  const historicalYieldChartEl = document.getElementById("historicalYieldChart");
  const historicalYieldChartCtx = historicalYieldChartEl.getContext("2d");
  const weatherIcon = document.getElementById("weather-icon");
  const weatherLabel = document.getElementById("weather-label");
  const cropEmoji = document.getElementById("crop-emoji");
  const cropPillLabel = document.getElementById("crop-pill-label");
  const climatePill = document.getElementById("climate-pill");
  const cropPill = document.getElementById("crop-pill");
  const cropSelect = document.getElementById("crop-type");
  const seasonSelect = document.getElementById("season");
  const stateSelect = document.getElementById("state");
  const areaInput = document.getElementById("area");
  const yearInput = document.getElementById("crop-year");
  const rainfallInput = document.getElementById("rainfall");
  const fertilizerInput = document.getElementById("fertilizer");
  const pesticideInput = document.getElementById("pesticide");
  const motifLayer = document.getElementById("motif-layer");
  const particleField = document.getElementById("particle-field");
  const liveLocation = document.getElementById("live-location");
  const liveCrop = document.getElementById("live-crop");
  const liveNumbers = document.getElementById("live-numbers");
  const healthFill = document.getElementById("health-fill");
  const healthLabel = document.getElementById("health-label");
  const adviceList = document.getElementById("advice-list");
  const copySummaryBtn = document.getElementById("copy-summary-btn");
  const shareSummaryBtn = document.getElementById("share-summary-btn");
  const formControls = Array.from(form.querySelectorAll("select, input, button[type='submit']"));
  let historicalChartInstance = null;
  let lastPrediction = null;

  const API_BASE_URL =
    window.location.protocol === "file:"
      ? "http://localhost:5000"
      : window.location.origin;

  const formatNumber = (value, digits = 2) =>
    Number.parseFloat(value).toFixed(digits);

  const cropThemeProfiles = [
    { key: "rice", className: "theme-rice", emoji: "🌾", label: "Paddy Theme" },
    { key: "maize", className: "theme-maize", emoji: "🌽", label: "Maize Theme" },
    { key: "cotton", className: "theme-cotton", emoji: "🧶", label: "Cotton Theme" },
    { key: "sugarcane", className: "theme-sugar", emoji: "🎋", label: "Sugarcane Theme" },
    { key: "gram", className: "theme-pulse", emoji: "🫘", label: "Pulse Theme" },
    { key: "lentil", className: "theme-pulse", emoji: "🫘", label: "Pulse Theme" },
    { key: "chickpea", className: "theme-pulse", emoji: "🫘", label: "Pulse Theme" },
    { key: "wheat", className: "theme-grain", emoji: "🌾", label: "Grain Theme" }
  ];

  const weatherProfiles = {
    kharif: { icon: "🌧", label: "Monsoon Active" },
    rabi: { icon: "🌤", label: "Cool Season Window" },
    summer: { icon: "☀", label: "Hot Dry Window" },
    winter: { icon: "❄", label: "Winter Growth Window" },
    autumn: { icon: "🍂", label: "Autumn Transition" },
    wholeyear: { icon: "🌀", label: "Year-round Cycle" },
    annual: { icon: "🌀", label: "Annual Cycle" }
  };

  const getCurrentThemeColors = () => {
    const styles = getComputedStyle(document.body);
    return {
      main: styles.getPropertyValue("--theme-a").trim() || "#1f5c2f",
      accent: styles.getPropertyValue("--theme-b").trim() || "#499a4b",
      soft: styles.getPropertyValue("--theme-soft").trim() || "rgba(73, 154, 75, 0.16)"
    };
  };

  function setCropTheme(cropName) {
    const normalized = (cropName || "").toLowerCase();
    const theme =
      cropThemeProfiles.find((profile) => normalized.includes(profile.key)) ||
      { className: "theme-grain", emoji: "🌾", label: "Crop Theme: Grain" };

    document.body.classList.remove(
      "theme-rice",
      "theme-maize",
      "theme-cotton",
      "theme-sugar",
      "theme-pulse",
      "theme-grain"
    );
    document.body.classList.add(theme.className);
    if (cropEmoji) cropEmoji.textContent = theme.emoji;
    if (cropPillLabel) {
      cropPillLabel.textContent = theme.label.startsWith("Crop Theme")
        ? theme.label
        : `Crop Theme: ${theme.label.replace(" Theme", "")}`;
    }
    if (motifLayer) {
      motifLayer.className = `motif-layer active motif-${theme.className.replace("theme-", "")}`;
    }
    syncParticles(theme.className);
  }

  function syncParticles(themeClassName) {
    if (!particleField) return;
    if (themeClassName !== "theme-cotton") {
      particleField.innerHTML = "";
      return;
    }

    if (particleField.childElementCount > 0) {
      return;
    }

    const bloomCount = 16;
    for (let i = 0; i < bloomCount; i += 1) {
      const bloom = document.createElement("span");
      bloom.className = "cotton-bloom";
      const size = 10 + Math.random() * 18;
      bloom.style.width = `${size}px`;
      bloom.style.height = `${size}px`;
      bloom.style.left = `${Math.random() * 100}%`;
      bloom.style.animationDuration = `${9 + Math.random() * 7}s`;
      bloom.style.animationDelay = `${Math.random() * 5}s`;
      particleField.appendChild(bloom);
    }
  }

  function setWeatherTheme(seasonName) {
    const normalized = (seasonName || "").toLowerCase().replace(/\s+/g, "");
    const weather = weatherProfiles[normalized] || { icon: "☀", label: "Sunny Growth Window" };
    if (weatherIcon) weatherIcon.textContent = weather.icon;
    if (weatherLabel) weatherLabel.textContent = weather.label;
    if (canAnimate && weatherIcon) {
      window.anime.remove(weatherIcon);
      runAnimation({
        targets: weatherIcon,
        scale: [0.85, 1.06, 1],
        duration: 600,
        easing: "easeOutBack"
      });
    }
  }

  function syncThemeFromInputs() {
    setCropTheme(cropSelect.value);
    setWeatherTheme(seasonSelect.value);
    const colors = getCurrentThemeColors();
    if (climatePill) climatePill.style.borderColor = colors.accent;
    if (cropPill) cropPill.style.borderColor = colors.accent;
  }

  function toNum(value) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function renderLiveDashboard() {
    const state = stateSelect.value || "--";
    const season = seasonSelect.value || "--";
    const crop = cropSelect.value || "--";
    const area = toNum(areaInput.value);
    const rain = toNum(rainfallInput.value);
    const fert = toNum(fertilizerInput.value);
    const pest = toNum(pesticideInput.value);
    const year = toNum(yearInput.value);

    if (liveLocation) liveLocation.textContent = `State: ${state} | Season: ${season}`;
    if (liveCrop) liveCrop.textContent = `Crop: ${crop} | Year: ${year ? Math.trunc(year) : "--"}`;
    if (liveNumbers) liveNumbers.textContent = `Area: ${area ?? "--"} ha | Rainfall: ${rain ?? "--"} mm`;

    const checks = [
      area !== null && area > 0,
      rain !== null && rain > 0,
      fert !== null && fert > 0,
      pest !== null && pest >= 0,
      !!stateSelect.value,
      !!cropSelect.value,
      !!seasonSelect.value
    ];
    const score = Math.round((checks.filter(Boolean).length / checks.length) * 100);
    if (healthFill) healthFill.style.width = `${score}%`;
    if (healthLabel) {
      healthLabel.textContent =
        score >= 85 ? "Excellent input coverage" :
        score >= 60 ? "Good coverage, refine a few fields" :
        "Low coverage, add more field context";
    }

    const hints = [];
    if (rain !== null && rain < 600) hints.push("Rainfall appears low; consider irrigation support planning.");
    if (rain !== null && rain > 1800) hints.push("Heavy rainfall profile; monitor drainage and fungal risk.");
    if (fert !== null && area !== null && area > 0 && fert / area > 1200) hints.push("High fertilizer intensity detected; check efficiency and soil testing.");
    if (pest !== null && pest > 6000) hints.push("Pesticide usage is high; validate threshold-based application.");
    if (!stateSelect.value || !cropSelect.value || !seasonSelect.value) hints.push("Select state, crop, and season for stronger model context.");
    if (hints.length === 0) hints.push("Inputs look balanced. Run forecast to view model drivers and historical trend.");

    if (adviceList) {
      adviceList.innerHTML = "";
      hints.slice(0, 3).forEach((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        adviceList.appendChild(li);
      });
    }
  }

  const setFieldStatus = (text) => {
    const status = document.getElementById("form-status");
    if (status) {
      status.textContent = text || "";
    }
  };

  function buildSummaryText() {
    if (!lastPrediction) return "";
    const topFactors = (lastPrediction.feature_contributions || [])
      .slice(0, 3)
      .map((f) => `${f.feature} (${f.contribution >= 0 ? "+" : "-"}${Math.abs(f.contribution).toFixed(2)})`)
      .join(", ");

    const ci =
      lastPrediction.predicted_yield_lower !== undefined && lastPrediction.predicted_yield_upper !== undefined
        ? `${formatNumber(lastPrediction.predicted_yield_lower)} - ${formatNumber(lastPrediction.predicted_yield_upper)}`
        : "N/A";

    return [
      "Crop Yield Forecast Summary",
      `State: ${stateSelect.value || "--"}`,
      `Crop: ${cropSelect.value || "--"}`,
      `Season: ${seasonSelect.value || "--"}`,
      `Predicted Yield: ${formatNumber(lastPrediction.predicted_yield)} tons/hectare`,
      `90% CI: ${ci}`,
      `Top Drivers: ${topFactors || "N/A"}`,
      `Model: ${lastPrediction.model_used || "N/A"}`,
      `Source: ${window.location.href}`
    ].join("\n");
  }

  async function copySummaryToClipboard() {
    if (!lastPrediction) return;
    const text = buildSummaryText();
    await navigator.clipboard.writeText(text);
    setFieldStatus("Summary copied to clipboard.");
  }

  async function shareSummary() {
    if (!lastPrediction) return;
    const text = buildSummaryText();
    if (navigator.share) {
      await navigator.share({
        title: "Crop Yield Forecast",
        text
      });
      setFieldStatus("Summary shared.");
      return;
    }
    await copySummaryToClipboard();
  }

  async function populateDropdowns() {
    try {
      setFieldStatus("Loading states, crops, and seasons...");
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      const response = await fetch(`${API_BASE_URL}/get-categories`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      const data = await response.json();

      if (!data.success || !data.categories) {
        throw new Error(data.error || "Failed to load categories.");
      }

      const populateSelect = (selectId, options, label) => {
        const selectElement = document.getElementById(selectId);
        selectElement.innerHTML = `<option value="">Choose ${label}</option>`;
        [...options].sort().forEach((option) => {
          selectElement.innerHTML += `<option value="${option}">${option}</option>`;
        });
      };

      populateSelect("state", data.categories.State || [], "a state");
      populateSelect("crop-type", data.categories.Crop || [], "a crop");
      populateSelect("season", data.categories.Season || [], "a season");
      setFieldStatus("Ready");
    } catch (error) {
      console.error("Dropdown load failed:", error);
      setFieldStatus("Form options failed to load. You can still submit if values are selected.");
      alert(`Error loading form options: ${error.message}`);
    }
  }

  function resetHistoryPanel() {
    historicalYieldChartEl.style.display = "block";
    historicalMessage.classList.add("hidden");
    historicalMessage.textContent = "";
  }

  function buildHistoricalChart(labels, historicalValues, predictedValues, predictedYear) {
    if (!canChart) {
      historicalYieldChartEl.style.display = "none";
      historicalMessage.textContent = "Chart library unavailable. Prediction still works.";
      historicalMessage.classList.remove("hidden");
      return;
    }

    if (historicalChartInstance) {
      historicalChartInstance.destroy();
    }

    const colors = getCurrentThemeColors();
    const compactMode = window.matchMedia("(min-width: 1100px) and (min-height: 700px)").matches;
    const tickSize = compactMode ? 10 : 12;
    const legendSize = compactMode ? 10 : 11;

    historicalChartInstance = new Chart(historicalYieldChartCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Historical Avg Yield",
            data: historicalValues,
            borderColor: colors.main,
            backgroundColor: colors.soft,
            borderWidth: 3,
            tension: 0.3,
            spanGaps: true
          },
          {
            label: `Predicted Yield (${predictedYear})`,
            data: predictedValues,
            borderColor: colors.accent,
            backgroundColor: colors.accent,
            pointRadius: 6,
            pointHoverRadius: 8,
            type: "scatter"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: "#3f3f46",
              usePointStyle: true,
              pointStyle: "line",
              boxWidth: compactMode ? 10 : 14,
              padding: compactMode ? 8 : 12,
              font: { size: legendSize, weight: "600" }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: "Yield (tons/hectare)",
              color: "#3f3f46",
              font: { size: tickSize + 1, weight: "700" }
            },
            ticks: {
              color: "#52525b",
              font: { size: tickSize, weight: "600" },
              maxTicksLimit: compactMode ? 5 : 7
            },
            grid: {
              color: "rgba(63, 63, 70, 0.12)"
            }
          },
          x: {
            ticks: {
              color: "#52525b",
              font: { size: tickSize, weight: "600" },
              maxTicksLimit: compactMode ? 6 : 10
            },
            grid: {
              color: "rgba(63, 63, 70, 0.08)"
            }
          }
        }
      }
    });
  }

  async function fetchHistoricalData(state, cropType, predictedYield, predictedYear) {
    try {
      resetHistoryPanel();
      const response = await fetch(
        `${API_BASE_URL}/get-historical-yield?state=${encodeURIComponent(state)}&cropType=${encodeURIComponent(cropType)}`
      );
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Historical data request failed." }));
        throw new Error(errorData.error || `HTTP error ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || "Historical data unavailable.");
      }

      if (!Array.isArray(data.years) || data.years.length === 0) {
        historicalYieldChartEl.style.display = "none";
        historicalMessage.textContent = data.message || "No historical data found for this state/crop.";
        historicalMessage.classList.remove("hidden");
        historicalDataSection.classList.remove("hidden");
        return;
      }

      const chartLabels = [...data.years];
      const chartYields = [...data.yields];

      if (!chartLabels.includes(predictedYear)) {
        chartLabels.push(predictedYear);
        chartYields.push(null);
      }

      const combined = chartLabels
        .map((year, idx) => ({ year, histYield: chartYields[idx] }))
        .sort((a, b) => a.year - b.year);

      const sortedLabels = combined.map((d) => d.year);
      const sortedHistYields = combined.map((d) => d.histYield);
      const predictedData = sortedLabels.map((year) => (year === predictedYear ? predictedYield : null));

      buildHistoricalChart(sortedLabels, sortedHistYields, predictedData, predictedYear);
      historicalDataSection.classList.remove("hidden");
    } catch (error) {
      console.error("Error fetching historical data:", error);
      historicalDataSection.classList.add("hidden");
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFieldStatus("Submitting forecast request...");
    loadingIndicator.classList.remove("hidden");
    featureContributionsSection.classList.add("hidden");
    historicalDataSection.classList.add("hidden");
    resetHistoryPanel();

    const formData = {
      state: document.getElementById("state").value,
      cropType: document.getElementById("crop-type").value,
      season: document.getElementById("season").value,
      Area: Number.parseFloat(document.getElementById("area").value),
      Crop_Year: Number.parseInt(document.getElementById("crop-year").value, 10),
      Annual_Rainfall: Number.parseFloat(document.getElementById("rainfall").value),
      Fertilizer: Number.parseFloat(document.getElementById("fertilizer").value),
      Pesticide: Number.parseFloat(document.getElementById("pesticide").value)
    };

    try {
      setFieldStatus("Contacting model API...");
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });

      const data = await response.json().catch(() => ({ success: false, error: "Unexpected server response." }));
      if (!response.ok || !data.success) {
        throw new Error(data.error || `Prediction failed with HTTP ${response.status}`);
      }

      const predictedValue = formatNumber(data.predicted_yield);
      setFieldStatus(`Forecast received: ${predictedValue} tons/hectare`);
      lastPrediction = data;
      if (copySummaryBtn) copySummaryBtn.disabled = false;
      if (shareSummaryBtn) shareSummaryBtn.disabled = false;

      if (predictionResultSection) {
        predictionResultSection.classList.remove("hidden");
      }

      if (yieldValue) {
        yieldValue.textContent = predictedValue;
      }
      if (modelNameDisplay) {
        modelNameDisplay.textContent = data.model_used || "N/A";
      }

      if (data.predicted_yield_lower !== undefined && data.predicted_yield_upper !== undefined) {
        if (yieldRange) {
          yieldRange.textContent = `${formatNumber(data.predicted_yield_lower)} - ${formatNumber(data.predicted_yield_upper)}`;
        }
        if (confidenceIntervalDisplay) {
          confidenceIntervalDisplay.classList.remove("hidden");
        }
      } else {
        if (confidenceIntervalDisplay) {
          confidenceIntervalDisplay.classList.add("hidden");
        }
      }

      featureContributionsList.innerHTML = "";
      if (Array.isArray(data.feature_contributions) && data.feature_contributions.length > 0) {
        const colors = getCurrentThemeColors();
        data.feature_contributions.slice(0, 4).forEach((item) => {
          const li = document.createElement("li");
          const direction = item.contribution >= 0 ? "up" : "down";
          const sign = item.contribution >= 0 ? "+" : "-";
          li.className = `factor ${direction}`;
          li.textContent = `${item.feature}: ${sign}${Math.abs(item.contribution).toFixed(2)}`;
          if (direction === "up") {
            li.style.borderLeftColor = colors.accent;
          }
          featureContributionsList.appendChild(li);
        });
        featureContributionsSection.classList.remove("hidden");
      }

      runAnimation({
        targets: "#prediction-result-section",
        opacity: [0, 1],
        scale: [0.98, 1],
        duration: 500,
        easing: "easeOutQuad"
      });

      await fetchHistoricalData(formData.state, formData.cropType, data.predicted_yield, formData.Crop_Year);

      if (predictionResultSection) {
        predictionResultSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (error) {
      setFieldStatus(`Forecast failed: ${error.message}`);
      alert(error.message);
    } finally {
      loadingIndicator.classList.add("hidden");
    }
  };

  form.addEventListener("submit", handleSubmit);
  const predictButton = document.getElementById("predict-button");
  predictButton.addEventListener("click", () => setFieldStatus("Validating inputs..."));

  if (copySummaryBtn) {
    copySummaryBtn.addEventListener("click", async () => {
      try {
        await copySummaryToClipboard();
      } catch (err) {
        setFieldStatus(`Copy failed: ${err.message}`);
      }
    });
  }

  if (shareSummaryBtn) {
    shareSummaryBtn.addEventListener("click", async () => {
      try {
        await shareSummary();
      } catch (err) {
        setFieldStatus(`Share failed: ${err.message}`);
      }
    });
  }

  formControls.forEach((control, idx) => {
    control.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        form.requestSubmit();
        return;
      }
      if (control.tagName === "TEXTAREA") return;
      event.preventDefault();
      const next = formControls[idx + 1];
      if (next) {
        next.focus();
      } else {
        form.requestSubmit();
      }
    });

    const group = control.closest(".form-group");
    if (group) {
      control.addEventListener("focus", () => group.classList.add("focused"));
      control.addEventListener("blur", () => group.classList.remove("focused"));
    }
  });

  cropSelect.addEventListener("change", () => {
    syncThemeFromInputs();
    renderLiveDashboard();
  });
  seasonSelect.addEventListener("change", () => {
    syncThemeFromInputs();
    renderLiveDashboard();
  });
  stateSelect.addEventListener("change", renderLiveDashboard);
  [areaInput, yearInput, rainfallInput, fertilizerInput, pesticideInput].forEach((el) => {
    el.addEventListener("input", renderLiveDashboard);
  });

  runAnimation({
    targets: ".reveal",
    translateY: [24, 0],
    opacity: [0, 1],
    delay: canAnimate ? window.anime.stagger(90) : 0,
    duration: 700,
    easing: "easeOutCubic"
  });

  populateDropdowns().finally(() => {
    syncThemeFromInputs();
    renderLiveDashboard();
  });
});
