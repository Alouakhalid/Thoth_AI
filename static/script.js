document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const tempInput = document.getElementById('temperature');
    const tempVal = document.getElementById('temp-val');
    const newChatBtn = document.getElementById('new-chat-btn');
    const toggleLogsBtn = document.getElementById('toggle-logs-btn');
    const logsPanel = document.getElementById('logs-panel');
    const logsOverlay = document.getElementById('logs-overlay');
    const closeLogsBtn = document.getElementById('close-logs-btn');
    const logsContent = document.getElementById('logs-content');
    const welcomeScreen = document.getElementById('welcome-screen');
    const sidebar = document.getElementById('sidebar');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');

    // Bulk train elements
    const bulkTrainBtn = document.getElementById('bulk-train-btn');
    const bulkModal = document.getElementById('bulk-modal');
    const bulkModalOverlay = document.getElementById('bulk-modal-overlay');
    const closeBulkModal = document.getElementById('close-bulk-modal');
    const bulkText = document.getElementById('bulk-text');
    const submitBulkTrain = document.getElementById('submit-bulk-train');
    const trainStatus = document.getElementById('train-status');

    let hasMessages = false;

    // Auto-resize textarea
    chatInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        sendBtn.disabled = this.value.trim() === '';
    });

    // Settings sliders
    tempInput.addEventListener('input', (e) => tempVal.textContent = e.target.value);

    // Mobile menu
    mobileMenuBtn.addEventListener('click', () => sidebar.classList.toggle('open'));

    // New chat
    newChatBtn.addEventListener('click', () => {
        chatContainer.innerHTML = '';
        if (welcomeScreen) {
            chatContainer.appendChild(createWelcomeScreen());
        }
        hasMessages = false;
        sidebar.classList.remove('open');
    });

    function createWelcomeScreen() {
        const div = document.createElement('div');
        div.className = 'welcome-screen';
        div.id = 'welcome-screen';
        div.innerHTML = `
            <img src="/static/logo.png" alt="Thoth" class="welcome-logo">
            <h2>Thoth AI</h2>
            <p class="welcome-subtitle">Intelligent · Adaptive · Continuously Learning</p>
            <p class="welcome-desc">I am a continuously-learning language model. Every message you send trains me to become smarter. Ask me anything or teach me something new.</p>
            <div class="welcome-chips">
                <button class="chip" data-prompt="Tell me about ancient Egypt">🏛️ Ancient Egypt</button>
                <button class="chip" data-prompt="What is artificial intelligence?">🤖 About AI</button>
                <button class="chip" data-prompt="How does the sun work?">☀️ How the Sun Works</button>
                <button class="chip" data-prompt="Tell me about space exploration">🚀 Space Exploration</button>
            </div>
        `;
        // Attach chip handlers
        div.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                chatInput.value = chip.dataset.prompt;
                chatInput.dispatchEvent(new Event('input'));
                sendMessage();
            });
        });
        return div;
    }

    // Attach initial chip handlers
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            chatInput.value = chip.dataset.prompt;
            chatInput.dispatchEvent(new Event('input'));
            sendMessage();
        });
    });

    // Send message
    const sendMessage = async () => {
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.disabled = true;

        // Remove welcome screen
        if (!hasMessages) {
            const ws = document.getElementById('welcome-screen');
            if (ws) ws.remove();
            hasMessages = true;
        }

        addUserMessage(text);
        const loadingId = addAiLoading();

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: text,
                    max_new_tokens: 500,
                    temperature: parseFloat(tempInput.value)
                })
            });

            if (!response.ok) throw new Error('Failed to generate');

            const data = await response.json();
            updateAiMessage(loadingId, data.response);
            setTimeout(fetchLogs, 2000);
        } catch (error) {
            console.error(error);
            updateAiMessage(loadingId, 'Sorry, I encountered an error. Please try again.');
        }
    };

    sendBtn.addEventListener('click', sendMessage);

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    function addUserMessage(text) {
        const msg = document.createElement('div');
        msg.className = 'message user-message';
        msg.innerHTML = `
            <div class="message-inner">
                <div class="avatar user-avatar">
                    <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 20 20" height="18" width="18">
                        <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"/>
                    </svg>
                </div>
                <div class="message-body"><p>${escapeHtml(text)}</p></div>
            </div>
        `;
        chatContainer.appendChild(msg);
        scrollToBottom();
    }

    function addAiLoading() {
        const id = 'msg-' + Date.now();
        const msg = document.createElement('div');
        msg.className = 'message ai-message';
        msg.id = id;
        msg.innerHTML = `
            <div class="message-inner">
                <div class="avatar ai-avatar">
                    <img src="/static/logo.png" alt="Thoth">
                </div>
                <div class="message-body">
                    <div class="typing-indicator"><span></span><span></span><span></span></div>
                </div>
            </div>
        `;
        chatContainer.appendChild(msg);
        scrollToBottom();
        return id;
    }

    function updateAiMessage(id, text) {
        const msg = document.getElementById(id);
        if (!msg) return;
        const body = msg.querySelector('.message-body');
        body.innerHTML = '';

        const safeText = escapeHtml(text);
        const p = document.createElement('p');
        body.appendChild(p);

        let i = 0;
        function typeWriter() {
            if (i < safeText.length) {
                p.innerHTML += safeText.charAt(i);
                i++;
                scrollToBottom();
                setTimeout(typeWriter, Math.random() * 15 + 8);
            }
        }
        typeWriter();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // =====================
    //   TRAINING LOGS
    // =====================

    let logPollInterval;

    function openLogs() {
        logsPanel.classList.add('active');
        logsOverlay.classList.add('active');
        fetchLogs();
        logPollInterval = setInterval(fetchLogs, 3000);
    }

    function closeLogs() {
        logsPanel.classList.remove('active');
        logsOverlay.classList.remove('active');
        clearInterval(logPollInterval);
    }

    toggleLogsBtn.addEventListener('click', openLogs);
    closeLogsBtn.addEventListener('click', closeLogs);
    logsOverlay.addEventListener('click', closeLogs);

    async function fetchLogs() {
        try {
            const res = await fetch('/logs');
            const data = await res.json();
            if (data.logs.length === 0) {
                logsContent.innerHTML = '<p class="logs-empty">No training logs yet.</p>';
                return;
            }
            logsContent.innerHTML = '';
            data.logs.forEach(line => {
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.innerHTML = formatLogLine(line);
                logsContent.appendChild(entry);
            });
            logsContent.scrollTop = logsContent.scrollHeight;
        } catch (err) {
            console.error('Failed to fetch logs:', err);
        }
    }

    function formatLogLine(line) {
        const timeMatch = line.match(/\[(.*?)\]/);
        const time = timeMatch ? timeMatch[1] : '';
        const isTrained = line.includes('TRAINED');
        const statusClass = isTrained ? 'log-status-trained' : 'log-status-skipped';
        const statusText = isTrained ? '✓ TRAINED' : '⊘ SKIPPED';

        const textMatch = line.match(/Text: "(.*?)"/);
        const text = textMatch ? textMatch[1] : '';

        const tokensMatch = line.match(/Tokens: (\d+)/);
        const tokens = tokensMatch ? tokensMatch[1] : '?';

        const chunksMatch = line.match(/Chunks: (\d+)/);
        const chunks = chunksMatch ? chunksMatch[1] : '?';

        const lossMatch = line.match(/Loss: ([\d.]+)/);
        const loss = lossMatch ? lossMatch[1] : null;

        let html = `<span class="log-time">${escapeHtml(time)}</span> `;
        html += `<span class="${statusClass}">${statusText}</span><br>`;
        html += `<span class="log-text">"${escapeHtml(text.substring(0, 50))}${text.length > 50 ? '...' : ''}"</span><br>`;
        html += `Tokens: ${tokens} · Chunks: ${chunks}`;
        if (loss) html += ` · Loss: <span class="log-loss">${loss}</span>`;
        return html;
    }

    // =====================
    //   BULK TRAIN MODAL
    // =====================

    function openBulkModal() {
        bulkModal.classList.add('active');
        bulkModalOverlay.classList.add('active');
    }

    function closeBulkModalFn() {
        bulkModal.classList.remove('active');
        bulkModalOverlay.classList.remove('active');
        trainStatus.textContent = '';
    }

    bulkTrainBtn.addEventListener('click', openBulkModal);
    closeBulkModal.addEventListener('click', closeBulkModalFn);
    bulkModalOverlay.addEventListener('click', closeBulkModalFn);

    submitBulkTrain.addEventListener('click', async () => {
        const text = bulkText.value.trim();
        if (!text) {
            trainStatus.textContent = '⚠️ Please enter some text to train on.';
            trainStatus.style.color = 'var(--danger)';
            return;
        }

        submitBulkTrain.disabled = true;
        trainStatus.textContent = '⏳ Training in progress (30 epochs)... This may take a moment.';
        trainStatus.style.color = 'var(--accent)';

        try {
            const response = await fetch('/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (response.ok) {
                trainStatus.textContent = '✅ Training complete! Thoth has learned from your text.';
                trainStatus.style.color = 'var(--success)';
                bulkText.value = '';
            } else {
                const err = await response.json();
                trainStatus.textContent = `❌ Error: ${err.detail}`;
                trainStatus.style.color = 'var(--danger)';
            }
        } catch (e) {
            trainStatus.textContent = `❌ Connection failed: ${e.message}`;
            trainStatus.style.color = 'var(--danger)';
        }

        submitBulkTrain.disabled = false;
    });

    // =====================
    //   MODEL DASHBOARD
    // =====================

    const toggleStatsBtn = document.getElementById('toggle-stats-btn');
    const statsModal = document.getElementById('stats-modal');
    const statsModalOverlay = document.getElementById('stats-modal-overlay');
    const closeStatsModal = document.getElementById('close-stats-modal');
    const statsGrid = document.getElementById('stats-grid');

    function openStatsModal() {
        statsModal.classList.add('active');
        statsModalOverlay.classList.add('active');
        fetchStats();
    }

    function closeStatsModalFn() {
        statsModal.classList.remove('active');
        statsModalOverlay.classList.remove('active');
    }

    toggleStatsBtn.addEventListener('click', openStatsModal);
    closeStatsModal.addEventListener('click', closeStatsModalFn);
    statsModalOverlay.addEventListener('click', closeStatsModalFn);

    async function fetchStats() {
        try {
            const res = await fetch('/stats');
            const data = await res.json();
            statsGrid.innerHTML = `
                <div class="stat-card highlight">
                    <span class="stat-value">${data.version}</span>
                    <span class="stat-label">Model Version</span>
                </div>
                <div class="stat-card highlight">
                    <span class="stat-value">${data.parameters_human}</span>
                    <span class="stat-label">Parameters</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${data.vocab_size.toLocaleString()}</span>
                    <span class="stat-label">Vocabulary Size</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${data.block_size}</span>
                    <span class="stat-label">Context Window</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${data.layers}L / ${data.attention_heads}H</span>
                    <span class="stat-label">Layers / Heads</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${data.train_sessions}</span>
                    <span class="stat-label">Training Sessions</span>
                </div>
            `;
        } catch (e) {
            statsGrid.innerHTML = '<div class="stat-card"><span class="stat-label">Failed to load stats</span></div>';
        }
    }

    // =====================
    //    EXPORT CHAT
    // =====================

    const exportChatBtn = document.getElementById('export-chat-btn');

    exportChatBtn.addEventListener('click', () => {
        const messages = chatContainer.querySelectorAll('.message');
        if (messages.length === 0) {
            alert('No chat messages to export.');
            return;
        }

        let text = '=== Thoth AI Chat Export ===\n';
        text += `Date: ${new Date().toLocaleString()}\n\n`;

        messages.forEach(msg => {
            const isUser = msg.classList.contains('user-message');
            const body = msg.querySelector('.message-body p, .message-content p');
            if (body) {
                text += `${isUser ? 'You' : 'Thoth'}: ${body.textContent}\n\n`;
            }
        });

        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `thoth_chat_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // =====================
    //  KEYBOARD SHORTCUTS
    // =====================

    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Shift + N = New Chat
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'N') {
            e.preventDefault();
            newChatBtn.click();
        }
        // Escape = close any open panel/modal
        if (e.key === 'Escape') {
            closeLogs();
            closeBulkModalFn();
            closeStatsModalFn();
        }
    });
});
