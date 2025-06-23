document.addEventListener('DOMContentLoaded', async function() {
    const form = document.getElementById('prediction-form');
    const loadingIndicator = document.getElementById('loading-indicator');
    const predictionResultSection = document.getElementById('prediction-result-section');
    const yieldValue = document.getElementById('yield-value');
    const confidenceIntervalDisplay = document.getElementById('confidence-interval');
    const yieldRange = document.getElementById('yield-range');
    const modelNameDisplay = document.getElementById('model-name-display');
    const featureContributionsSection = document.getElementById('feature-contributions-section');
    const featureContributionsList = document.getElementById('feature-contributions-list');
    const historicalDataSection = document.getElementById('historical-data-section');
    const historicalYieldChartCtx = document.getElementById('historicalYieldChart').getContext('2d');
    let historicalChartInstance = null;

    const API_BASE_URL = 'http://localhost:5000'; // Or your deployed API URL

    async function populateDropdowns() {
        try {
            const response = await fetch(`${API_BASE_URL}/get-categories`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            if (data.success && data.categories) {
                const populateSelect = (selectId, options) => {
                    const selectElement = document.getElementById(selectId);
                    selectElement.innerHTML = `<option value="">Select a ${selectId.replace('-', ' ')}</option>`;
                    options.sort().forEach(option => {
                        selectElement.innerHTML += `<option value="${option}">${option}</option>`;
                    });
                };
                populateSelect('state', data.categories.State || []);
                populateSelect('crop-type', data.categories.Crop || []);
                populateSelect('season', data.categories.Season || []);
            } else {
                console.error('Failed to load categories:', data.error);
                alert('Error loading form options. Please check server logs and refresh.');
            }
        } catch (error) {
            console.error('Error fetching categories:', error);
            alert(`Error loading form options: ${error.message}. Please ensure the backend is running and model.pkl is loaded correctly.`);
        }
    }

    await populateDropdowns();

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        loadingIndicator.classList.remove('hidden');
        predictionResultSection.classList.add('hidden');
        featureContributionsSection.classList.add('hidden');
        historicalDataSection.classList.add('hidden');
        
        const formData = {
            state: document.getElementById('state').value,
            cropType: document.getElementById('crop-type').value,
            season: document.getElementById('season').value,
            Area: parseFloat(document.getElementById('area').value),
            Crop_Year: parseInt(document.getElementById('crop-year').value),
            Annual_Rainfall: parseFloat(document.getElementById('rainfall').value),
            Fertilizer: parseFloat(document.getElementById('fertilizer').value),
            Pesticide: parseFloat(document.getElementById('pesticide').value)
        };

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({error: "Unknown server error"}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            if (data.success) {
                yieldValue.textContent = data.predicted_yield;
                modelNameDisplay.textContent = data.model_used || 'N/A';

                if (data.predicted_yield_lower !== undefined && data.predicted_yield_upper !== undefined) {
                    yieldRange.textContent = `${data.predicted_yield_lower} - ${data.predicted_yield_upper}`;
                    confidenceIntervalDisplay.classList.remove('hidden');
                } else {
                    confidenceIntervalDisplay.classList.add('hidden');
                }

                if (data.feature_contributions && data.feature_contributions.length > 0) {
                    featureContributionsList.innerHTML = ''; // Clear previous
                    data.feature_contributions.forEach(item => {
                        const li = document.createElement('li');
                        const changeType = item.contribution > 0 ? 'increased' : 'decreased';
                        const absContribution = Math.abs(item.contribution);
                        li.textContent = `${item.feature}: ${changeType} prediction by ~${absContribution.toFixed(2)}`;
                        li.style.color = item.contribution > 0 ? '#4ade80' : '#f87171'; // Green for positive, Red for negative
                        featureContributionsList.appendChild(li);
                    });
                    featureContributionsSection.classList.remove('hidden');
                }


                predictionResultSection.classList.remove('hidden');
                anime({
                    targets: '#prediction-result-section',
                    opacity: [0, 1],
                    translateY: [20, 0],
                    duration: 800,
                    easing: 'easeOutElastic'
                });

                // Fetch and display historical data
                await fetchHistoricalData(formData.state, formData.cropType, data.predicted_yield, formData.Crop_Year);

            } else {
                alert('Error making prediction: ' + data.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            loadingIndicator.classList.add('hidden');
        }
    });

    async function fetchHistoricalData(state, cropType, predictedYield, predictedYear) {
        try {
            const response = await fetch(`${API_BASE_URL}/get-historical-yield?state=${encodeURIComponent(state)}&cropType=${encodeURIComponent(cropType)}`);
            if (!response.ok) {
                 const errorData = await response.json().catch(() => ({error: "Unknown server error fetching historical data"}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            if (data.success && data.years && data.years.length > 0) {
                historicalDataSection.classList.remove('hidden');
                if (historicalChartInstance) {
                    historicalChartInstance.destroy();
                }
                
                // Prepare data for chart, adding the current prediction
                const chartLabels = [...data.years];
                const chartYields = [...data.yields];
                
                // Find if predictedYear already exists in historical data
                const predictedYearIndex = chartLabels.indexOf(predictedYear);
                if (predictedYearIndex > -1) {
                    // If it exists, we might want to show it differently or add a separate point
                    // For now, let's add a separate dataset for prediction
                     historicalChartInstance = new Chart(historicalYieldChartCtx, {
                        type: 'line',
                        data: {
                            labels: chartLabels,
                            datasets: [{
                                label: 'Historical Average Yield',
                                data: chartYields,
                                borderColor: 'rgba(75, 192, 192, 1)',
                                tension: 0.1
                            }, {
                                label: `Predicted Yield (${predictedYear})`,
                                data: chartLabels.map(yr => yr === predictedYear ? predictedYield : null), // Only plot prediction for its year
                                borderColor: 'rgba(255, 99, 132, 1)',
                                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                pointRadius: 6,
                                pointHoverRadius: 8,
                                type: 'scatter' // Show prediction as a distinct point
                            }]
                        },
                        options: { scales: { y: { beginAtZero: true, title: { display: true, text: 'Yield (tons/hectare)' } } } }
                    });

                } else {
                     // Add predicted year and yield if not already present, and sort
                    chartLabels.push(predictedYear);
                    chartYields.push(null); // Placeholder for historical
                    
                    const combined = chartLabels.map((year, index) => ({ year, histYield: chartYields[index] }))
                                             .sort((a, b) => a.year - b.year);
                    
                    const sortedLabels = combined.map(d => d.year);
                    const sortedHistYields = combined.map(d => d.histYield);
                    
                    const predictedData = sortedLabels.map(yr => yr === predictedYear ? predictedYield : null);

                    historicalChartInstance = new Chart(historicalYieldChartCtx, {
                        type: 'line',
                        data: {
                            labels: sortedLabels,
                            datasets: [{
                                label: 'Historical Average Yield',
                                data: sortedHistYields,
                                borderColor: 'rgba(75, 192, 192, 1)',
                                tension: 0.1,
                                spanGaps: true // Connect lines over nulls for historical data
                            }, {
                                label: `Predicted Yield (${predictedYear})`,
                                data: predictedData,
                                borderColor: 'rgba(255, 99, 132, 1)',
                                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                pointRadius: 6,
                                pointHoverRadius: 8,
                                type: 'scatter' 
                            }]
                        },
                        options: { scales: { y: { beginAtZero: true, title: { display: true, text: 'Yield (tons/hectare)' } } } }
                    });
                }


            } else if (data.message) {
                 historicalDataSection.classList.remove('hidden');
                 document.getElementById('historicalYieldChart').style.display = 'none'; // Hide canvas
                 const p = document.createElement('p');
                 p.textContent = data.message;
                 if(historicalDataSection.childElementCount > 1) historicalDataSection.removeChild(historicalDataSection.lastChild); // remove old p if exists
                 historicalDataSection.appendChild(p);

            } else {
                historicalDataSection.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error fetching historical yield data:', error);
            historicalDataSection.classList.add('hidden');
            // Optionally, display a small error message for historical data
        }
    }

});