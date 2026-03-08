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
    const logprobsMode = document.getElementById('logprobs-mode');
    const temperatureSlider = document.getElementById('temperature');
    const topP_Slider = document.getElementById('top_p');
    const topK_Input = document.getElementById('top_k');
    const tempValLabel = document.getElementById('temp-val');
    const topP_ValLabel = document.getElementById('topp-val');
    const topK_ValLabel = document.getElementById('topk-val');
    const responseCompareDisplay = document.getElementById('ai-response-compare');
    const responseContainer = document.getElementById('response-container');
    const comparisonLabel = document.getElementById('comparison-label');
    const logprobsLegend = document.getElementById('logprobs-legend');

    // Token probability panel
    const tokenProbSection = document.getElementById('token-prob-section');
    const tokenProbBody = document.getElementById('token-prob-body');
    const avgConfEl = document.getElementById('avg-confidence');
    const minConfEl = document.getElementById('min-confidence');
    const highConfCountEl = document.getElementById('high-conf-count');
    const lowConfCountEl = document.getElementById('low-conf-count');

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
                            ${log.Temperature !== "" && log.Temperature !== null ? `T: ${log.Temperature}` : '-'} 
                            ${log.Top_P !== "" && log.Top_P !== null ? `| P: ${log.Top_P}` : ''}
                            ${log.Logprobs === "True" || log.Logprobs === true ? `| LP: Yes` : ''}
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

    // Helper: Build Logprobs HTML (colored spans in response)
    function buildLogprobsHtml(logprobsArray) {
        let html = '';
        logprobsArray.forEach(lp => {
            const prob = lp.probability_pct;
            let cls = 'high';
            if (prob < 50) cls = 'low';
            else if (prob < 90) cls = 'medium';
            // Replace newlines with <br> for proper rendering in sequence
            const safeToken = lp.token.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
            html += `<span class="token-logprob ${cls}" title="Logprob: ${lp.logprob.toFixed(2)} (${prob}%)">${safeToken}</span>`;
        });
        return html;
    }

    // Helper: Render Token Probability Panel
    function renderTokenProbabilities(logprobsArray) {
        if (!logprobsArray || logprobsArray.length === 0) {
            tokenProbSection.classList.add('hidden');
            return;
        }

        // Stats calculations
        const probs = logprobsArray.map(lp => lp.probability_pct);
        const avg = (probs.reduce((a, b) => a + b, 0) / probs.length).toFixed(1);
        const min = Math.min(...probs).toFixed(1);
        const highCount = probs.filter(p => p >= 90).length;
        const lowCount = probs.filter(p => p < 50).length;

        avgConfEl.textContent = `${avg}%`;
        minConfEl.textContent = `${min}%`;
        highConfCountEl.textContent = highCount;
        lowConfCountEl.textContent = lowCount;

        // Build table rows
        tokenProbBody.innerHTML = '';
        logprobsArray.forEach((lp, i) => {
            const prob = lp.probability_pct;
            let level = 'high';
            if (prob < 50) level = 'low';
            else if (prob < 90) level = 'medium';

            // Display whitespace/newlines visibly
            const displayToken = lp.token
                .replace(/\n/g, '\\n')
                .replace(/\t/g, '\\t')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            const probColor = level === 'high' ? 'var(--success)' : level === 'medium' ? 'var(--warning)' : 'var(--danger)';

            const row = `<tr class="fade-in">
                <td>${i + 1}</td>
                <td><span class="token-cell">${displayToken}</span></td>
                <td class="logprob-cell">${lp.logprob.toFixed(4)}</td>
                <td class="prob-cell" style="color:${probColor}">${prob}%</td>
                <td class="conf-bar-cell">
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill ${level}" style="width:${Math.min(prob, 100)}%"></div>
                    </div>
                </td>
                <td><span class="level-badge ${level}">${level}</span></td>
            </tr>`;
            tokenProbBody.insertAdjacentHTML('beforeend', row);
        });

        tokenProbSection.classList.remove('hidden');
    }

    // Run Experiment
    async function runExperiment(text = null) {
        const prompt = text || promptArea.value;
        const model = modelSelect.value;
        const experiment_type = experimentTypeSelect.value;
        const temperature = parseFloat(temperatureSlider.value);
        const top_p = parseFloat(topP_Slider.value);
        const top_k = parseInt(topK_Input.value) || null;
        const return_logprobs = logprobsMode.checked;
        
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
                top_k: compareMode.checked ? null : top_k,
                return_logprobs: return_logprobs
            };
            
            const dataPrimary = await callApi(payloadPrimary);
            const analysisP = dataPrimary.log_analysis;
            
            tokenDisplay.textContent = analysisP.input_tokens + analysisP.output_tokens;
            costDisplay.textContent = `$${parseFloat(analysisP.cost_usd).toFixed(6)}`;
            inputCostDisplay.textContent = `Experiment: ${analysisP.experiment_type} ${compareMode.checked ? '(T=0)' : ''}`;
            
            if (dataPrimary.logprobs) {
                responseDisplay.innerHTML = `<p style="line-height:2;">${buildLogprobsHtml(dataPrimary.logprobs)}</p>`;
                logprobsLegend.classList.remove('hidden');
                renderTokenProbabilities(dataPrimary.logprobs);
            } else {
                responseDisplay.innerHTML = `<p>${dataPrimary.response}</p>`;
                if (!compareMode.checked) logprobsLegend.classList.add('hidden');
                tokenProbSection.classList.add('hidden');
            }

            // Update Window UI (based on primary)
            const percent = analysisP.usage_percentage;
            progress.style.width = `${percent}%`;
            contextInfo.textContent = `${percent}% of context limit (${analysisP.input_tokens} / ${analysisP.context_limit})`;
            
            // 2nd Run: Comparison (Creative/Slider settings)
            if (compareMode.checked) {
                const payloadSecondary = { prompt, model, experiment_type, temperature, top_p, top_k, return_logprobs };
                const dataSec = await callApi(payloadSecondary);
                const analysisS = dataSec.log_analysis;
                
                if (dataSec.logprobs) {
                    responseCompareDisplay.innerHTML = `<p style="line-height:2;">${buildLogprobsHtml(dataSec.logprobs)}</p>`;
                    logprobsLegend.classList.remove('hidden');
                } else {
                    responseCompareDisplay.innerHTML = `<p>${dataSec.response}</p>`;
                }
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
