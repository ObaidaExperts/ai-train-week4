document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const stressTestBtn = document.getElementById('stress-test-btn');
    const promptArea = document.getElementById('prompt');
    const modelSelect = document.getElementById('model');
    const experimentTypeSelect = document.getElementById('experiment-type');
    const tokenDisplay = document.getElementById('token-count');
    const costDisplay = document.getElementById('total-cost');
    const inputCostDisplay = document.getElementById('input-cost');
    const outputCostDisplay = document.getElementById('output-cost');
    const responseDisplay = document.getElementById('ai-response');
    const logsBody = document.getElementById('logs-body');
    const progress = document.getElementById('context-progress');
    const contextInfo = document.getElementById('context-info');
    const apiIndicator = document.getElementById('api-status');
    const apiText = document.getElementById('api-text');

    const compareMode = document.getElementById('compare-mode');
    const temperatureSlider = document.getElementById('temperature');
    const topP_Slider = document.getElementById('top_p');
    const topK_Input = document.getElementById('top_k');
    const tempValLabel = document.getElementById('temp-val');
    const topP_ValLabel = document.getElementById('topp-val');
    const topK_ValLabel = document.getElementById('topk-val');
    const responseCompareDisplay = document.getElementById('ai-response-compare');
    const responseContainer = document.getElementById('response-container');
    const comparisonLabel = document.getElementById('comparison-label');

    // Update Slider Labels
    temperatureSlider.oninput = () => tempValLabel.textContent = temperatureSlider.value;
    topP_Slider.oninput = () => topP_ValLabel.textContent = topP_Slider.value;
    topK_Input.oninput = () => topK_ValLabel.textContent = topK_Input.value || 'None';

    // Fetch and populate metadata
    async function fetchMetadata() {
        try {
            const res = await fetch('/metadata');
            const data = await res.json();
            modelSelect.innerHTML = data.models.map(m => `<option value="${m}">${m}</option>`).join('');
            experimentTypeSelect.innerHTML = data.experiment_types.map(t => `<option value="${t}">${t}</option>`).join('');
            // Set Default if Decoding Strategy exists
            if (data.experiment_types.includes("Decoding Strategy")) {
                experimentTypeSelect.value = "Decoding Strategy";
            }
        } catch (e) { console.error('Failed to fetch metadata'); }
    }

    // Check API Health
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) { apiIndicator.classList.add('online'); apiText.textContent = 'API Online'; }
        } catch (e) { console.error('API health check failed'); }
    }

    // Refresh logs
    async function refreshLogs() {
        try {
            const res = await fetch('/results');
            const data = await res.json();
            logsBody.innerHTML = '';
            
            // Log Header Update (handled in HTML, but we'll ensure data matches)
            data.reverse().forEach(log => {
                const row = `
                    <tr class="fade-in">
                        <td>${new Date(log.Timestamp).toLocaleTimeString()}</td>
                        <td><small>${log.Model}</small></td>
                        <td>${log.Experiment_Type}</td>
                        <td class="params-cell">
                            ${log.Temperature !== "" ? `T: ${log.Temperature}` : '-'} 
                            ${log.Top_P !== "" ? `| P: ${log.Top_P}` : ''}
                        </td>
                        <td>In: ${log.Input_Tokens} | Out: ${log.Output_Tokens}</td>
                        <td class="text-accent">$${parseFloat(log.Cost_USD).toFixed(6)}</td>
                        <td><span class="status-badge">${log.Status}</span></td>
                    </tr>
                `;
                logsBody.insertAdjacentHTML('beforeend', row);
            });
        } catch (e) { console.error('Failed to load logs'); }
    }

    // Call API helper
    async function callApi(payload) {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'API request failed');
        }
        return await res.json();
    }

    // Run Experiment
    async function runExperiment(text = null) {
        const prompt = text || promptArea.value;
        const model = modelSelect.value;
        const experiment_type = experimentTypeSelect.value;
        const temperature = parseFloat(temperatureSlider.value);
        const top_p = parseFloat(topP_Slider.value);
        const top_k = parseInt(topK_Input.value) || null;
        
        if (!prompt) return alert('Please enter a prompt');

        runBtn.disabled = true;
        runBtn.textContent = 'Analyzing...';
        responseDisplay.innerHTML = '<p class="placeholder">Analyzing tokens and generating response...</p>';
        responseCompareDisplay.innerHTML = '<p class="placeholder">Comparison view...</p>';
        
        // Handle Layout
        if (compareMode.checked) {
            responseContainer.classList.add('comparing');
            responseCompareDisplay.classList.remove('hidden');
            comparisonLabel.classList.remove('hidden');
        } else {
            responseContainer.classList.remove('comparing');
            responseCompareDisplay.classList.add('hidden');
            comparisonLabel.classList.add('hidden');
        }

        try {
            // 1st Run: Primary (Deterministic if Comparing, else Slider)
            const payloadPrimary = { 
                prompt, model, experiment_type,
                temperature: compareMode.checked ? 0 : temperature,
                top_p: compareMode.checked ? 1 : top_p,
                top_k: compareMode.checked ? null : top_k
            };
            
            const dataPrimary = await callApi(payloadPrimary);
            const analysisP = dataPrimary.log_analysis;
            
            tokenDisplay.textContent = analysisP.input_tokens + analysisP.output_tokens;
            costDisplay.textContent = `$${parseFloat(analysisP.cost_usd).toFixed(6)}`;
            inputCostDisplay.textContent = `Experiment: ${analysisP.experiment_type} ${compareMode.checked ? '(T=0)' : ''}`;
            responseDisplay.innerHTML = `<p>${dataPrimary.response}</p>`;

            // Update Window UI (based on primary)
            const percent = analysisP.usage_percentage;
            progress.style.width = `${percent}%`;
            contextInfo.textContent = `${percent}% of context limit (${analysisP.input_tokens} / ${analysisP.context_limit})`;
            
            // 2nd Run: Comparison (Creative/Slider settings)
            if (compareMode.checked) {
                const payloadSecondary = { prompt, model, experiment_type, temperature, top_p, top_k };
                const dataSec = await callApi(payloadSecondary);
                const analysisS = dataSec.log_analysis;
                
                responseCompareDisplay.innerHTML = `<p>${dataSec.response}</p>`;
                outputCostDisplay.textContent = `Creative: T=${temperature}`; // Reuse output cost for meta
            } else {
                outputCostDisplay.textContent = `Out: ${analysisP.output_tokens} tokens`;
            }

            refreshLogs();
        } catch (e) {
            responseDisplay.innerHTML = `<p style="color:var(--danger)">Error: ${e.message}</p>`;
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Analyze & Run';
        }
    }

    stressTestBtn.addEventListener('click', () => {
        const largeText = "Context limit test. ".repeat(4000);
        promptArea.value = largeText;
        experimentTypeSelect.value = "Stress Test";
        runExperiment(largeText);
    });

    runBtn.addEventListener('click', () => runExperiment());

    fetchMetadata();
    checkHealth();
    refreshLogs();
});
