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

    // Fetch and populate metadata (agenticModel defined later in Single vs Agentic section)
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
            // Populate agentic model selector from same source
            const agenticEl = document.getElementById('agentic-model');
            if (agenticEl) {
                agenticEl.innerHTML = data.models.map(m => `<option value="${m}">${m}</option>`).join('');
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
                // OpenAI: real token-level probabilities returned
                responseDisplay.innerHTML = `<p style="line-height:2;">${buildLogprobsHtml(dataPrimary.logprobs)}</p>`;
                logprobsLegend.classList.remove('hidden');
                renderTokenProbabilities(dataPrimary.logprobs);
            } else if (dataPrimary.logprobs_supported === false) {
                // Non-OpenAI model: show response + a warning banner
                responseDisplay.innerHTML = `<p>${dataPrimary.response}</p>`;
                logprobsLegend.classList.add('hidden');
                tokenProbSection.classList.remove('hidden');
                tokenProbBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:2rem; color:var(--warning);">
                    ⚠️ ${dataPrimary.logprobs_note}
                </td></tr>`;
                avgConfEl.textContent = 'N/A';
                minConfEl.textContent = 'N/A';
                highConfCountEl.textContent = 'N/A';
                lowConfCountEl.textContent = 'N/A';
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

    // ═══════════════════════════════════════════
    // Tab switching
    // ═══════════════════════════════════════════
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => { p.classList.remove('active'); p.classList.add('hidden'); });
            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.tab);
            target.classList.remove('hidden');
            target.classList.add('active');
        });
    });

    // ═══════════════════════════════════════════
    // Tool Calling Tab
    // ═══════════════════════════════════════════
    const toolRunBtn    = document.getElementById('tool-run-btn');
    const toolErrorBtn  = document.getElementById('tool-error-btn');
    const toolPromptArea = document.getElementById('tool-prompt');
    const toolTraceCard  = document.getElementById('tool-trace-card');
    const toolTraceSteps = document.getElementById('tool-trace-steps');
    const toolStatusBadge = document.getElementById('tool-status-badge');
    const toolSchemasDisplay = document.getElementById('tool-schemas-display');

    // Example prompt pills (only tool-call pills with data-prompt, not agentic pills)
    document.querySelectorAll('.example-pill[data-prompt]').forEach(pill => {
        pill.addEventListener('click', () => {
            toolPromptArea.value = pill.dataset.prompt;
            toolPromptArea.focus();
        });
    });

    // Fetch and display schemas
    async function loadToolSchemas() {
        try {
            const res = await fetch('/tools/schemas');
            const data = await res.json();
            toolSchemasDisplay.textContent = JSON.stringify(data.tools, null, 2);
        } catch (e) {
            toolSchemasDisplay.textContent = 'Failed to load tool schemas.';
        }
    }
    loadToolSchemas();

    // Render step helper
    function renderStep(step) {
        const el = document.createElement('div');
        el.className = 'trace-step';

        if (step.step === 'user_prompt') {
            el.classList.add('step-user');
            el.innerHTML = `<div class="trace-step-title">👤 User Prompt</div>
                <div class="trace-step-body">${step.content}</div>`;

        } else if (step.step === 'tool_call_requested') {
            el.classList.add('step-tool-call');
            let argsHtml = '';
            try {
                const parsed = JSON.parse(step.args_raw);
                argsHtml = Object.entries(parsed)
                    .map(([k, v]) => `<span class="arg-chip">${k}: <strong>${JSON.stringify(v)}</strong></span>`)
                    .join(' ');
            } catch { argsHtml = `<code>${step.args_raw}</code>`; }
            el.innerHTML = `<div class="trace-step-title">🔧 Tool Called: ${step.tool}</div>
                <div class="trace-step-body">Arguments: ${argsHtml}</div>`;

        } else if (step.step === 'tool_result') {
            el.classList.add('step-result');
            el.innerHTML = `<div class="trace-step-title">✅ Tool Result: ${step.tool}</div>
                <div class="trace-step-body"><pre>${JSON.stringify(step.result, null, 2)}</pre></div>`;

        } else if (step.step === 'tool_error') {
            el.classList.add('step-error');
            el.innerHTML = `<div class="trace-step-title">❌ Tool Error: ${step.tool}</div>
                <div class="trace-step-body" style="color:var(--danger)">${step.error}</div>`;

        } else if (step.step === 'final_answer') {
            el.classList.add('step-answer');
            el.innerHTML = `<div class="trace-step-title">🤖 Final AI Answer</div>
                <div class="trace-step-body">${step.content}</div>`;

        } else if (step.step === 'model_direct_answer') {
            el.classList.add('step-direct');
            el.innerHTML = `<div class="trace-step-title">💬 Direct Answer (no tool triggered)</div>
                <div class="trace-step-body">${step.content}</div>`;
        }

        toolTraceSteps.appendChild(el);
    }

    async function runToolCall(forceError = false) {
        const prompt = toolPromptArea.value.trim();
        if (!prompt && !forceError) return alert('Please enter a prompt');

        const enabledTools = [...document.querySelectorAll('.tool-checkbox:checked')].map(cb => cb.value);

        toolRunBtn.disabled = true;
        toolErrorBtn.disabled = true;
        toolRunBtn.textContent = 'Running...';
        toolTraceSteps.innerHTML = '';
        toolTraceCard.classList.remove('hidden');
        toolStatusBadge.className = 'tool-status-badge';
        toolStatusBadge.textContent = '';

        try {
            const payload = {
                prompt: prompt || 'What is the weather in Paris?',
                model: 'gpt-4o',
                enabled_tools: enabledTools.length ? enabledTools : null,
                force_error: forceError
            };
            const res = await fetch('/tool-call', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error((await res.json()).detail || 'Tool call failed');
            const data = await res.json();

            // Render trace steps with staggered animation
            for (const step of data.steps) {
                await new Promise(r => setTimeout(r, 150));
                renderStep(step);
                toolTraceCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            if (data.tool_error) {
                toolStatusBadge.className = 'tool-status-badge error';
                toolStatusBadge.textContent = '❌ Tool Error Handled';
            } else if (!data.tool_called) {
                toolStatusBadge.className = 'tool-status-badge direct';
                toolStatusBadge.textContent = '💬 Direct Response';
            } else {
                toolStatusBadge.className = 'tool-status-badge success';
                toolStatusBadge.textContent = `✅ Tool Executed: ${data.tool_called}`;
            }
        } catch (e) {
            toolTraceSteps.innerHTML = `<div class="trace-step step-error"><div class="trace-step-title">Error</div><div class="trace-step-body" style="color:var(--danger)">${e.message}</div></div>`;
        } finally {
            toolRunBtn.disabled = false;
            toolErrorBtn.disabled = false;
            toolRunBtn.textContent = '▶ Run Tool Call';
        }
    }

    toolRunBtn.addEventListener('click', () => runToolCall(false));
    toolErrorBtn.addEventListener('click', () => runToolCall(true));

    // ═══════════════════════════════════════════
    // Single vs Agentic Flow
    // ═══════════════════════════════════════════
    const agenticTab = document.getElementById('agentic-flow-tab');
    const agenticRequest = document.getElementById('agentic-request');
    const agenticModel = document.getElementById('agentic-model');
    const agenticSingleBtn = document.getElementById('agentic-single-btn');
    const agenticAgenticBtn = document.getElementById('agentic-agentic-btn');
    const agenticBothBtn = document.getElementById('agentic-both-btn');
    const agenticSingleResponse = document.getElementById('agentic-single-response');
    const agenticSingleMeta = document.getElementById('agentic-single-meta');
    const agenticAgenticResponse = document.getElementById('agentic-agentic-response');
    const agenticAgenticMeta = document.getElementById('agentic-agentic-meta');
    const agenticAgenticPlan = document.getElementById('agentic-agentic-plan');
    const agenticAgenticTrace = document.getElementById('agentic-agentic-trace');
    const agenticPlanDetails = document.getElementById('agentic-plan-details');
    const agenticTraceDetails = document.getElementById('agentic-trace-details');
    const agenticComparisonSummary = document.getElementById('agentic-comparison-summary');

    // Example pills: use delegation on the agentic tab container
    if (agenticTab) {
        agenticTab.addEventListener('click', (e) => {
            const pill = e.target.closest('button.agentic-example');
            if (pill && pill.dataset.request && agenticRequest) {
                e.preventDefault();
                e.stopPropagation();
                agenticRequest.value = pill.dataset.request;
                agenticRequest.focus();
            }
        });
    }

    function setButtonLoading(btn, loading) {
        const text = btn.querySelector('.btn-text');
        const spinner = btn.querySelector('.btn-spinner');
        if (text && spinner) {
            if (loading) {
                btn.disabled = true;
                spinner.classList.remove('hidden');
            } else {
                btn.disabled = false;
                spinner.classList.add('hidden');
            }
        }
    }

    function parseApiError(body) {
        if (!body || typeof body !== 'object') return 'Request failed';
        const d = body.detail;
        if (typeof d === 'string') return d;
        if (Array.isArray(d) && d.length > 0) {
            const first = d[0];
            return first.msg || first.message || JSON.stringify(first);
        }
        return body.message || 'Request failed';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function renderAgenticSingle(data) {
        agenticSingleMeta.textContent = `${data.input_tokens || 0} in / ${data.output_tokens || 0} out tokens`;
        agenticSingleResponse.innerHTML = `<div class="response-text">${escapeHtml(data.response || '').replace(/\n/g, '<br>')}</div>`;
    }

    function renderAgenticFlow(data) {
        agenticAgenticMeta.textContent = `${data.input_tokens || 0} in / ${data.output_tokens || 0} out · ${data.iterations || 0} iterations`;
        agenticAgenticResponse.innerHTML = `<div class="response-text">${escapeHtml(data.response || '').replace(/\n/g, '<br>')}</div>`;

        if (data.plan && data.plan.steps) {
            agenticPlanDetails.classList.remove('hidden');
            agenticPlanDetails.open = false;
            agenticAgenticPlan.innerHTML = `<pre>${escapeHtml(JSON.stringify(data.plan, null, 2))}</pre>`;
        } else {
            agenticPlanDetails.classList.add('hidden');
        }

        if (data.steps && data.steps.length > 0) {
            agenticTraceDetails.classList.remove('hidden');
            agenticTraceDetails.open = false;
            let traceHtml = '';
            data.steps.forEach((s) => {
                if (s.phase === 'planning') {
                    traceHtml += `<div class="trace-step step-user"><strong>Phase 1: Planning</strong></div>`;
                } else if (s.phase === 'tool_call') {
                    const resultStr = s.result ? escapeHtml(JSON.stringify(s.result)) : '';
                    traceHtml += `<div class="trace-step step-tool-call"><strong>Tool:</strong> ${escapeHtml(s.tool)} → <code>${resultStr}</code></div>`;
                } else if (s.phase === 'tool_error') {
                    traceHtml += `<div class="trace-step step-error"><strong>Tool Error:</strong> ${escapeHtml(s.tool)} — ${escapeHtml(s.error || '')}</div>`;
                } else if (s.phase === 'complete') {
                    traceHtml += `<div class="trace-step step-answer"><strong>Phase 2: Complete</strong></div>`;
                }
            });
            agenticAgenticTrace.innerHTML = traceHtml;
        } else {
            agenticTraceDetails.classList.add('hidden');
        }
    }

    async function runAgenticSingle() {
        if (!agenticRequest || !agenticModel) return;
        const user_request = agenticRequest.value.trim();
        if (!user_request) return alert('Please enter a trip request');

        setButtonLoading(agenticSingleBtn, true);
        agenticSingleResponse.innerHTML = '<p class="placeholder agentic-loading">Running single prompt...</p>';
        agenticSingleMeta.textContent = '';
        agenticComparisonSummary.classList.add('hidden');

        try {
            const res = await fetch('/agentic-flow/single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_request, model: agenticModel.value })
            });
            const body = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(parseApiError(body));
            renderAgenticSingle(body);
        } catch (e) {
            agenticSingleResponse.innerHTML = `<p class="agentic-error">Error: ${escapeHtml(e.message)}</p>`;
        } finally {
            setButtonLoading(agenticSingleBtn, false);
        }
    }

    async function runAgenticFlow() {
        if (!agenticRequest || !agenticModel) return;
        const user_request = agenticRequest.value.trim();
        if (!user_request) return alert('Please enter a trip request');

        setButtonLoading(agenticAgenticBtn, true);
        agenticAgenticResponse.innerHTML = '<p class="placeholder agentic-loading">Running agentic flow (planning + tools)...</p>';
        agenticAgenticMeta.textContent = '';
        agenticPlanDetails.classList.add('hidden');
        agenticTraceDetails.classList.add('hidden');
        agenticComparisonSummary.classList.add('hidden');

        try {
            const res = await fetch('/agentic-flow/agentic', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_request, model: agenticModel.value })
            });
            const body = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(parseApiError(body));
            renderAgenticFlow(body);
        } catch (e) {
            agenticAgenticResponse.innerHTML = `<p class="agentic-error">Error: ${escapeHtml(e.message)}</p>`;
        } finally {
            setButtonLoading(agenticAgenticBtn, false);
        }
    }

    async function runAgenticBoth() {
        if (!agenticRequest || !agenticModel) return;
        const user_request = agenticRequest.value.trim();
        if (!user_request) return alert('Please enter a trip request');

        setButtonLoading(agenticSingleBtn, true);
        setButtonLoading(agenticAgenticBtn, true);
        setButtonLoading(agenticBothBtn, true);
        agenticSingleResponse.innerHTML = '<p class="placeholder agentic-loading">Running...</p>';
        agenticAgenticResponse.innerHTML = '<p class="placeholder agentic-loading">Running...</p>';
        agenticSingleMeta.textContent = '';
        agenticAgenticMeta.textContent = '';
        agenticPlanDetails.classList.add('hidden');
        agenticTraceDetails.classList.add('hidden');
        agenticComparisonSummary.classList.add('hidden');

        try {
            const [singleRes, agenticRes] = await Promise.all([
                fetch('/agentic-flow/single', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_request, model: agenticModel.value })
                }),
                fetch('/agentic-flow/agentic', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_request, model: agenticModel.value })
                })
            ]);

            const singleBody = await singleRes.json().catch(() => ({}));
            const agenticBody = await agenticRes.json().catch(() => ({}));

            const singleData = singleRes.ok ? singleBody : null;
            const agenticData = agenticRes.ok ? agenticBody : null;

            if (singleData) renderAgenticSingle(singleData);
            else {
                agenticSingleResponse.innerHTML = `<p class="agentic-error">Error: ${escapeHtml(parseApiError(singleBody))}</p>`;
            }

            if (agenticData) renderAgenticFlow(agenticData);
            else {
                agenticAgenticResponse.innerHTML = `<p class="agentic-error">Error: ${escapeHtml(parseApiError(agenticBody))}</p>`;
            }

            if (singleData && agenticData) {
                const singleTokens = (singleData.input_tokens || 0) + (singleData.output_tokens || 0);
                const agenticTokens = (agenticData.input_tokens || 0) + (agenticData.output_tokens || 0);
                const diff = agenticTokens - singleTokens;
                agenticComparisonSummary.innerHTML = `
                    <span>Single: ${singleTokens} tokens</span>
                    <span>Agentic: ${agenticTokens} tokens (${agenticData.iterations || 0} iterations)</span>
                    <span class="agentic-diff">${diff > 0 ? '+' : ''}${diff} tokens for agentic</span>
                `;
                agenticComparisonSummary.classList.remove('hidden');
            }
        } catch (e) {
            agenticSingleResponse.innerHTML = `<p class="agentic-error">Error: ${escapeHtml(e.message)}</p>`;
            agenticAgenticResponse.innerHTML = `<p class="agentic-error">Error: ${escapeHtml(e.message)}</p>`;
        } finally {
            setButtonLoading(agenticSingleBtn, false);
            setButtonLoading(agenticAgenticBtn, false);
            setButtonLoading(agenticBothBtn, false);
        }
    }

    if (agenticSingleBtn) agenticSingleBtn.addEventListener('click', runAgenticSingle);
    if (agenticAgenticBtn) agenticAgenticBtn.addEventListener('click', runAgenticFlow);
    if (agenticBothBtn) agenticBothBtn.addEventListener('click', runAgenticBoth);
});
