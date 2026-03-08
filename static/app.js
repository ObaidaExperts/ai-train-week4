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

    // Fetch and populate metadata
    async function fetchMetadata() {
        try {
            const res = await fetch('/metadata');
            const data = await res.json();
            
            // Populate models
            modelSelect.innerHTML = data.models.map(m => `<option value="${m}">${m}</option>`).join('');
            
            // Populate experiment types
            experimentTypeSelect.innerHTML = data.experiment_types.map(t => `<option value="${t}">${t}</option>`).join('');
        } catch (e) {
            console.error('Failed to fetch metadata');
        }
    }

    // Check API Health
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                apiIndicator.classList.add('online');
                apiText.textContent = 'API Online';
            }
        } catch (e) {
            console.error('API health check failed');
        }
    }

    // Refresh logs
    async function refreshLogs() {
        try {
            const res = await fetch('/results');
            const data = await res.json();
            logsBody.innerHTML = '';
            data.reverse().slice(0, 5).forEach(log => {
                const row = `
                    <tr>
                        <td>${new Date(log.Timestamp).toLocaleTimeString()}</td>
                        <td>${log.Model}</td>
                        <td>In: ${log.Input_Tokens} | Out: ${log.Output_Tokens}</td>
                        <td>$${parseFloat(log.Cost_USD).toFixed(6)}</td>
                        <td><span class="status-badge">${log.Status}</span></td>
                    </tr>
                `;
                logsBody.insertAdjacentHTML('beforeend', row);
            });
        } catch (e) {
            console.error('Failed to load logs');
        }
    }

    // Run Experiment
    async function runExperiment(text = null) {
        const prompt = text || promptArea.value;
        const model = modelSelect.value;
        const experiment_type = experimentTypeSelect.value;
        
        if (!prompt) return alert('Please enter a prompt');

        runBtn.disabled = true;
        runBtn.textContent = 'Analyzing...';
        responseDisplay.innerHTML = '<p class="placeholder">Analyzing tokens and generating response...</p>';

        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, model, experiment_type })
            });
            const data = await res.json();

            if (res.ok) {
                const analysis = data.log_analysis;
                tokenDisplay.textContent = analysis.input_tokens + analysis.output_tokens;
                costDisplay.textContent = `$${parseFloat(analysis.cost_usd).toFixed(6)}`;
                
                // Use actual rates from models or approximate for UI
                inputCostDisplay.textContent = `Type: ${analysis.experiment_type}`; 
                
                responseDisplay.innerHTML = `<p>${data.response}</p>`;
                
                const percent = analysis.usage_percentage;
                progress.style.width = `${percent}%`;
                contextInfo.textContent = `${percent}% of context limit (${analysis.input_tokens} / ${analysis.context_limit})`;
                
                if (percent > 80) progress.style.background = 'var(--danger)';
                else if (percent > 50) progress.style.background = 'var(--warning)';
                else progress.style.background = 'var(--primary)';

                refreshLogs();
            } else {
                responseDisplay.innerHTML = `<p style="color:var(--danger)">Error: ${data.detail || 'Failed to run experiment'}</p>`;
            }
        } catch (e) {
            responseDisplay.innerHTML = `<p style="color:var(--danger)">Network Error: ${e.message}</p>`;
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Analyze & Run';
        }
    }

    // Stress Test
    stressTestBtn.addEventListener('click', () => {
        const largeText = "Context limit test. ".repeat(4000); // Generate large prompt
        promptArea.value = largeText;
        experimentTypeSelect.value = "Stress Test";
        runExperiment(largeText);
    });

    runBtn.addEventListener('click', () => runExperiment());

    fetchMetadata();
    checkHealth();
    refreshLogs();
});
