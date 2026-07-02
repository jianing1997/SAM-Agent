// SAM-Agent - Frontend JavaScript

let ws = null;
let isConnected = false;
let currentThinkingSteps = [];

// Session Management
let currentSessionId = null;
let sessions = {}; // { sessionId: { id, title, messages: [], timestamp } }
let sessionOrder = []; // Array of session IDs in order

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadTools();
    loadSessions();  // Load sessions
    initCurrentSession();  // Initialize current session
    adjustTextareaHeight();
});

// WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/chat`;
    
    updateStatus('connecting', 'Connecting...');
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        isConnected = true;
        updateStatus('connected', 'Connected');
        console.log('✅ WebSocket connection successful');
        console.log(`📌 Current session ID: ${currentSessionId}`);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('error', 'Connection error');
    };
    
    ws.onclose = () => {
        isConnected = false;
        updateStatus('disconnected', 'Disconnected');
        console.log('WebSocket connection closed');
        // Attempt to reconnect
        setTimeout(initWebSocket, 3000);
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'user_message':
            // User message already displayed
            break;
            
        case 'thinking_start':
            showTypingIndicator();
            currentThinkingSteps = [];
            break;
            
        case 'step':
            removeTypingIndicator();
            addThinkingStep(data);
            break;

        case 'tool_trace':
            removeTypingIndicator();
            addToolTrace(data.steps || [], data.thoughts || []);
            break;
            
        case 'assistant_message':
            removeTypingIndicator();
            addAssistantMessage(data);
            // Session is auto-saved in addAssistantMessage
            saveSessions();  // Persist to localStorage
            updateSessionList();  // Update sidebar
            break;
            
        case 'error':
            removeTypingIndicator();
            addErrorMessage(data.message);
            break;
    }
}

// Update connection status
function updateStatus(status, text) {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    
    statusDot.className = 'status-dot';
    if (status === 'connected') {
        statusDot.classList.add('connected');
    } else if (status === 'error') {
        statusDot.classList.add('error');
    }
    
    statusText.textContent = text;
}

// Load tools list
async function loadTools() {
    try {
        const response = await fetch('/api/tools');
        const data = await response.json();
        
        const toolsList = document.getElementById('tools-list');
        const toolCount = document.getElementById('tool-count');
        
        if (toolsList) {
            toolsList.innerHTML = '';
            
            data.tools.forEach(tool => {
                const toolItem = document.createElement('div');
                toolItem.className = 'tool-item';
                toolItem.innerHTML = `
                    <span class="tool-icon">${tool.icon}</span>
                    <div class="tool-info">
                        <div class="tool-name">${tool.name}</div>
                        <div class="tool-description">${tool.description}</div>
                    </div>
                `;
                toolsList.appendChild(toolItem);
            });
        }
        
        if (toolCount) {
            toolCount.textContent = `${data.tools.length} tools`;
        }
    } catch (error) {
        console.error('Failed to load tools list:', error);
    }
}

// Note: History loading is now handled by loadSessions() and stored in localStorage

// Send message
function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (!message || !isConnected) return;
    
    // Clear welcome message if exists
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // Add user message to chat
    addUserMessage(message);
    
    // Save and update session list (message is saved in addUserMessage)
    saveSessions();
    updateSessionList();
    
    // Send via WebSocket with session_id
    console.log(`📤 Sending message [Session: ${currentSessionId}]`);
    ws.send(JSON.stringify({ 
        message: message,
        session_id: currentSessionId  // Attach session ID
    }));
    
    // Clear input
    input.value = '';
    adjustTextareaHeight();
    
    // Disable send button temporarily
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = true;
    setTimeout(() => {
        sendBtn.disabled = false;
    }, 1000);
}

// Send example prompt
function sendExample(text) {
    const input = document.getElementById('user-input');
    input.value = text;
    sendMessage();
}

// Add user message to chat
function addUserMessage(content, timestampISO = null, skipSave = false) {
    const chatContainer = document.getElementById('chat-container');
    const timestamp = timestampISO 
        ? new Date(timestampISO).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
        : new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar user-avatar">U</div>
            <span class="message-role">User</span>
            <span class="message-time">${timestamp}</span>
        </div>
        <div class="message-content">${escapeHtml(content)}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    
    // Save to session (unless loading from history)
    if (!skipSave && currentSessionId && sessions[currentSessionId]) {
        const msgData = {
            role: 'user',
            content: content,
            timestamp: timestampISO || new Date().toISOString()
        };
        sessions[currentSessionId].messages.push(msgData);
        
        // Update title if this is the first user message
        if (sessions[currentSessionId].messages.filter(m => m.role === 'user').length === 1) {
            sessions[currentSessionId].title = content.substring(0, 30) + 
                (content.length > 30 ? '...' : '');
        }
        sessions[currentSessionId].timestamp = Date.now();
    }
}

function renderThoughtBlock(content) {
    return `
        <div class="step-thought">
            <div><strong>Thought</strong></div>
            <div>${escapeHtml(content)}</div>
        </div>
    `;
}

// Add assistant message to chat
function addAssistantMessage(data, skipSave = false) {
    const chatContainer = document.getElementById('chat-container');
    const timestamp = new Date(data.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant-avatar">A</div>
            <span class="message-role">AI Agent</span>
            <span class="message-time">${timestamp}</span>
        </div>
        <div class="message-content">${formatMarkdown(data.content)}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    
    // Add molecules if present
    if (data.molecules && data.molecules.length > 0) {
        addMoleculeVisualization(data.molecules, data.content);
    }
    
    scrollToBottom();
    
    // Save to session (unless loading from history)
    if (!skipSave && currentSessionId && sessions[currentSessionId]) {
        const msgData = {
            role: 'assistant',
            content: data.content,
            molecules: data.molecules || [],
            steps: data.steps || [],
            thoughts: data.thoughts || [],
            timestamp: data.timestamp
        };
        sessions[currentSessionId].messages.push(msgData);
        sessions[currentSessionId].timestamp = Date.now();
    }
}

// Add execution step from verbose fallback output
function addThinkingStep(step) {
    let stepsContainer = document.querySelector('.thinking-steps:last-child');
    
    if (!stepsContainer) {
        const chatContainer = document.getElementById('chat-container');
        stepsContainer = document.createElement('div');
        stepsContainer.className = 'thinking-steps';
        chatContainer.appendChild(stepsContainer);
    }
    
    const stepDiv = document.createElement('div');
    stepDiv.className = `step-item step-${step.step_type}`;
    
    let icon = '🧭';
    let header = 'Execution note';
    let content = step.content;
    
    if (step.step_type === 'action') {
        icon = '🔧';
        header = `Tool call: ${step.tool}`;
        content = `<pre>${JSON.stringify(step.input, null, 2)}</pre>`;
    } else if (step.step_type === 'observation') {
        icon = '📊';
        header = 'Tool output';
    }
    
    stepDiv.innerHTML = `
        <div class="step-header">
            <span class="step-icon">${icon}</span>
            <span>${header}</span>
        </div>
        <div class="step-content">${content}</div>
    `;
    
    stepsContainer.appendChild(stepDiv);
    scrollToBottom();
}

function addToolTrace(steps, thoughts = []) {
    const filteredSteps = (steps || []).filter(step => step.tool !== '_Exception');
    const validThoughts = (thoughts || []).filter(Boolean).map(item => String(item).trim()).filter(Boolean);
    if (!filteredSteps.length && !validThoughts.length) return;

    const chatContainer = document.getElementById('chat-container');
    const traceContainer = document.createElement('div');
    traceContainer.className = 'thinking-steps tool-trace';

    const title = document.createElement('div');
    title.className = 'step-item step-trace-title';
    title.innerHTML = `
        <div class="step-header">
            <span class="step-icon">🧰</span>
            <span>Chain of thought</span>
        </div>
        <div class="step-content">Captured thoughts, tool actions, and observations used to produce the answer.</div>
    `;
    traceContainer.appendChild(title);

    validThoughts.forEach((thought) => {
        const thoughtDiv = document.createElement('div');
        thoughtDiv.className = 'step-item step-thought-item';
        thoughtDiv.innerHTML = renderThoughtBlock(thought);
        traceContainer.appendChild(thoughtDiv);
    });

    filteredSteps.forEach((step) => {
        const stepDiv = document.createElement('div');
        stepDiv.className = `step-item step-${step.status || 'success'}`;
        const inputText = JSON.stringify(step.input || {}, null, 2);
        stepDiv.innerHTML = `
            <div class="step-header">
                <span class="step-icon">${step.status === 'error' ? '⚠️' : '🔧'}</span>
                <span>Action ${step.index}: ${escapeHtml(step.tool || 'Unknown tool')}</span>
                <span class="step-status">${escapeHtml(step.status || 'success')}</span>
            </div>
            <div class="step-content">
                <div><strong>Action input</strong></div>
                <pre>${escapeHtml(inputText)}</pre>
                <div><strong>Observation</strong></div>
                <pre>${escapeHtml(step.output_preview || '')}</pre>
            </div>
        `;
        traceContainer.appendChild(stepDiv);
    });

    chatContainer.appendChild(traceContainer);
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const chatContainer = document.getElementById('chat-container');
    
    // Remove existing indicator
    removeTypingIndicator();
    
    const timestamp = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'message assistant-message typing-message';
    indicatorDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant-avatar">A</div>
            <span class="message-role">AI Agent</span>
            <span class="message-time">${timestamp}</span>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;
    
    chatContainer.appendChild(indicatorDiv);
    scrollToBottom();
}

// Remove typing indicator
function removeTypingIndicator() {
    const indicator = document.querySelector('.typing-message');
    if (indicator) {
        indicator.remove();
    }
}

// Add error message
function addErrorMessage(message) {
    const chatContainer = document.getElementById('chat-container');
    const timestamp = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar assistant-avatar">A</div>
            <span class="message-role">AI Agent</span>
            <span class="message-time">${timestamp}</span>
        </div>
        <div class="message-content" style="color: #ef4444;">
            ⚠️ ${escapeHtml(message)}
        </div>
    `;
    
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Add molecule visualization with better error handling
function addMoleculeVisualization(molecules, messageContent = '') {
    const chatContainer = document.getElementById('chat-container');
    
    // Parse molecular property information
    const moleculeProperties = parseMoleculeProperties(messageContent, molecules);
    
    const moleculeDiv = document.createElement('div');
    moleculeDiv.className = 'molecule-container';
    moleculeDiv.innerHTML = `
        <div class="molecule-header">
            🧬 Detected ${molecules.length} molecular structures
        </div>
        <div class="molecule-grid" id="molecule-grid-${Date.now()}">
        </div>
    `;
    
    chatContainer.appendChild(moleculeDiv);
    
    const grid = moleculeDiv.querySelector('.molecule-grid');
    
    molecules.forEach((smiles, index) => {
        // Clean SMILES string
        const cleanSmiles = smiles.trim().replace(/[,;.]$/, '');
        
        const item = document.createElement('div');
        item.className = 'molecule-item';
        
        // Use multiple fallback APIs
        const pubchemUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(cleanSmiles)}/PNG?image_size=300x300`;
        const cdkUrl = `https://www.simolecule.com/cdkdepict/depict/cow/svg?smi=${encodeURIComponent(cleanSmiles)}&annotate=colmap`;
        
        // Get property information for this molecule
        const properties = moleculeProperties[cleanSmiles] || {};
        const hasProperties = Object.keys(properties).length > 0;
        
        // Build properties display HTML
        let propertiesHtml = '';
        if (hasProperties) {
            propertiesHtml = '<div class="molecule-properties">';
            if (properties.homo) propertiesHtml += `<div class="property-item"><span class="property-label">HOMO:</span> <span class="property-value">${properties.homo}</span></div>`;
            if (properties.lumo) propertiesHtml += `<div class="property-item"><span class="property-label">LUMO:</span> <span class="property-value">${properties.lumo}</span></div>`;
            if (properties.dipole) propertiesHtml += `<div class="property-item"><span class="property-label">Dipole:</span> <span class="property-value">${properties.dipole}</span></div>`;
            if (properties.gap) propertiesHtml += `<div class="property-item"><span class="property-label">Band Gap:</span> <span class="property-value">${properties.gap}</span></div>`;
            if (properties.efficiency) propertiesHtml += `<div class="property-item"><span class="property-label">Efficiency:</span> <span class="property-value">${properties.efficiency}</span></div>`;
            propertiesHtml += '</div>';
        }
        
        item.innerHTML = `
            <div class="molecule-canvas" id="mol-canvas-${Date.now()}-${index}">
                <img src="${pubchemUrl}" 
                     alt="Molecule ${index + 1}"
                     style="max-width: 100%; height: auto; background: white; padding: 10px; border-radius: 6px;"
                     onerror="handleMoleculeError(this, '${escapeHtml(cleanSmiles)}')">
            </div>
            <div class="molecule-info">
                <div class="molecule-smiles" title="${escapeHtml(cleanSmiles)}">
                    <span class="smiles-label">SMILES:</span>
                    ${escapeHtml(cleanSmiles.length > 50 ? cleanSmiles.substring(0, 47) + '...' : cleanSmiles)}
                </div>
                ${propertiesHtml}
            </div>
            <button class="copy-smiles-btn" onclick="copyToClipboard('${escapeHtml(cleanSmiles)}')" title="Copy SMILES">
                📋 Copy
            </button>
        `;
        grid.appendChild(item);
    });
    
    scrollToBottom();
}

// Parse molecular property information
function parseMoleculeProperties(messageContent, molecules) {
    const properties = {};
    
    if (!messageContent) return properties;
    
    console.log('Starting to parse molecular properties...');
    console.log('Message content length:', messageContent.length);
    console.log('Number of molecules:', molecules.length);
    
    // Iterate through each molecule and try to match its properties
    molecules.forEach((smiles, idx) => {
        const cleanSmiles = smiles.trim().replace(/[,;.]$/, '');
        properties[cleanSmiles] = {};
        
        console.log(`\nProcessing molecule ${idx + 1}: ${cleanSmiles.substring(0, 30)}...`);
        
        // Find the position of this molecule in the message
        const smilesIndex = messageContent.indexOf(cleanSmiles);
        
        if (smilesIndex === -1) {
            // Try to find the first 20 characters (format might be slightly different in message)
            const shortSmiles = cleanSmiles.substring(0, 20);
            const altIndex = messageContent.indexOf(shortSmiles);
            console.log(`  Full match failed, trying prefix match: ${altIndex >= 0 ? 'success' : 'failed'}`);
            
            if (altIndex === -1) {
                console.log('  ⚠ Molecule not found in message');
                return;
            }
        }
        
        // After finding the molecule, extract the next 500 characters as the section that might contain properties
        const startPos = smilesIndex >= 0 ? smilesIndex : messageContent.indexOf(cleanSmiles.substring(0, 20));
        const section = messageContent.substring(startPos, startPos + 500);
        
        console.log('  Extracted section:', section.substring(0, 100).replace(/\n/g, '↵'));
        
        // Extract HOMO (support multiple formats)
        const homoMatch = section.match(/HOMO[:\s]*(-?\d+\.?\d*)\s*eV/i);
        if (homoMatch) {
            const value = homoMatch[1];
            properties[cleanSmiles].homo = `${value} eV`;
            console.log('  ✓ HOMO:', properties[cleanSmiles].homo);
        }
        
        // Extract LUMO
        const lumoMatch = section.match(/LUMO[:\s]*(-?\d+\.?\d*)\s*eV/i);
        if (lumoMatch) {
            const value = lumoMatch[1];
            properties[cleanSmiles].lumo = `${value} eV`;
            console.log('  ✓ LUMO:', properties[cleanSmiles].lumo);
        }
        
        // Extract dipole moment - very flexible pattern
        const dipoleMatch = section.match(/Dipole\s+Moment[:\s]*(-?\d+\.?\d*)\s*D|Dipole[:\s]*(-?\d+\.?\d*)\s*D|DM[:\s]*(-?\d+\.?\d*)\s*D/i);
        if (dipoleMatch) {
            const value = dipoleMatch[1] || dipoleMatch[2] || dipoleMatch[3];
            if (value) {
                properties[cleanSmiles].dipole = `${value} D`;
                console.log('  ✓ Dipole:', properties[cleanSmiles].dipole);
            }
        }
        
        // Extract band gap
        const gapMatch = section.match(/(?:Band\s+Gap|Gap)[:\s]*(-?\d+\.?\d*)\s*eV/i);
        if (gapMatch) {
            const value = gapMatch[1];
            properties[cleanSmiles].gap = `${value} eV`;
            console.log('  ✓ Band Gap:', properties[cleanSmiles].gap);
        }
        
        // Extract efficiency
        const effMatch = section.match(/(?:Efficiency|PCE)[:\s]*(-?\d+\.?\d*)\s*%?/i);
        if (effMatch) {
            const value = effMatch[1];
            properties[cleanSmiles].efficiency = `${value}%`;
            console.log('  ✓ Efficiency:', properties[cleanSmiles].efficiency);
        }
        
        const propCount = Object.keys(properties[cleanSmiles]).length;
        console.log(`  Found ${propCount} properties in total`);
    });
    
    console.log('\nProperty parsing complete!');
    return properties;
}

// Handle molecule image loading errors
function handleMoleculeError(img, smiles) {
    // Try fallback API
    if (img.src.includes('pubchem')) {
        img.src = `https://www.simolecule.com/cdkdepict/depict/bow/svg?smi=${encodeURIComponent(smiles)}&annotate=colmap`;
    } else {
        // If all fail, show placeholder
        img.style.display = 'none';
        const placeholder = document.createElement('div');
        placeholder.style.cssText = 'padding: 40px 20px; text-align: center; color: #94a3b8; background: #f7f7f8; border-radius: 6px;';
        placeholder.innerHTML = '⚗️<br><small>Molecular Structure</small><br><small style="font-size: 0.7em;">(Image loading failed)</small>';
        img.parentNode.appendChild(placeholder);
    }
}

// Copy to clipboard helper
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('✓ SMILES copied to clipboard!');
    }).catch(() => {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        alert('✓ SMILES copied to clipboard!');
    });
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    // Simple markdown formatting
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/`(.*?)`/g, '<code>$1</code>');
    text = text.replace(/\n/g, '<br>');
    return text;
}

function scrollToBottom() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Handle keyboard shortcuts
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Auto-resize textarea
function adjustTextareaHeight() {
    const textarea = document.getElementById('user-input');
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

document.getElementById('user-input').addEventListener('input', adjustTextareaHeight);

// ===== Session Management Functions =====

// Generate unique session ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Initialize current session
function initCurrentSession() {
    if (!currentSessionId) {
        currentSessionId = generateSessionId();
        sessions[currentSessionId] = {
            id: currentSessionId,
            title: 'New Chat',
            messages: [],
            timestamp: Date.now()
        };
        sessionOrder.unshift(currentSessionId);
        saveSessions();
    }
    updateSessionList();
}

// Save current session before creating new one
function saveCurrentSession() {
    if (!currentSessionId || !sessions[currentSessionId]) return;
    
    // Messages are already saved to session object in addUserMessage/addAssistantMessage
    // Just persist to localStorage
    saveSessions();
}

// Create new chat session
function newChat() {
    console.log('🆕 Creating new chat...');
    
    // Save current session if it has messages
    const hasMessages = document.querySelectorAll('.message').length > 0;
    if (hasMessages) {
        console.log('💾 Saving current session (has', hasMessages, 'messages)');
        saveCurrentSession();
    } else {
        // Remove empty current session
        if (currentSessionId && sessionOrder.includes(currentSessionId)) {
            console.log('🗑️ Deleting empty session:', currentSessionId);
            delete sessions[currentSessionId];
            sessionOrder = sessionOrder.filter(id => id !== currentSessionId);
        }
    }
    
    // Create new session
    currentSessionId = generateSessionId();
    sessions[currentSessionId] = {
        id: currentSessionId,
        title: 'New Chat',
        messages: [],
        timestamp: Date.now()
    };
    sessionOrder.unshift(currentSessionId);
    console.log('✨ Created new session:', currentSessionId);
    saveSessions();
    
    // Clear chat UI
    const chatContainer = document.getElementById('chat-container');
    chatContainer.innerHTML = `
        <div class="welcome-message">
            <h2>👋 Welcome to SAM-Agent</h2>
            <p>I can help you with the following tasks:</p>
            <div class="feature-grid">
                <div class="feature-card">
                    <span class="feature-icon">🧬</span>
                    <h3>Molecule Generation</h3>
                    <p>Generate SAM molecules based on scaffolds and anchoring groups</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">📊</span>
                    <h3>Property Prediction</h3>
                    <p>Predict HOMO, LUMO, and dipole moment</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">⚡</span>
                    <h3>Device Evaluation</h3>
                    <p>Evaluate perovskite solar cell efficiency</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">🔄</span>
                    <h3>Synthesis Planning</h3>
                    <p>Generate retrosynthetic routes</p>
                </div>
            </div>
            <div class="example-prompts">
                <p><strong>Try these questions:</strong></p>
                <button class="example-btn" onclick="sendExample('Generate 3 SAM molecules with carbazole scaffold and phosphonic acid anchoring group')">
                    Generate SAM molecules with carbazole scaffold
                </button>
                <button class="example-btn" onclick="sendExample('Predict HOMO, LUMO and dipole moment for molecule O=P(O)(O)CCCn1c2ccccc2c2ccccc21')">
                    Predict molecular electronic properties
                </button>
                <button class="example-btn" onclick="sendExample('What is SAM and what is its role in perovskite solar cells?')">
                    Learn about SAM applications
                </button>
            </div>
        </div>
    `;
    
    // No longer clear backend memory - each session retains its own memory
    // fetch('/api/clear', { method: 'POST' });  // ❌ Removed
    
    // Update session list
    updateSessionList();
}

// Switch to a different session
async function switchSession(sessionId) {
    if (sessionId === currentSessionId) return;
    
    console.log(`🔄 Switching session: ${currentSessionId} → ${sessionId}`);
    
    // Save current session
    saveCurrentSession();
    
    // Switch to selected session
    currentSessionId = sessionId;
    
    // Load session messages
    const session = sessions[sessionId];
    const chatContainer = document.getElementById('chat-container');
    chatContainer.innerHTML = '';
    
    if (session && session.messages.length > 0) {
        // Render all messages (skipSave = true to avoid duplicate saves)
        session.messages.forEach(msg => {
            if (msg.role === 'user') {
                addUserMessage(msg.content, msg.timestamp, true);
            } else {
                if ((msg.steps && msg.steps.length) || (msg.thoughts && msg.thoughts.length)) {
                    addToolTrace(msg.steps || [], msg.thoughts || []);
                }
                addAssistantMessage({
                    content: msg.content,
                    molecules: msg.molecules || [],
                    steps: msg.steps || [],
                    thoughts: msg.thoughts || [],
                    timestamp: msg.timestamp || new Date().toISOString()
                }, true);
            }
        });
        
        // Sync history messages to backend Agent's memory
        console.log(`📤 Syncing ${session.messages.length} historical messages to backend...`);
        try {
            const response = await fetch('/api/restore_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    messages: session.messages
                })
            });
            
            const result = await response.json();
            if (result.success) {
                console.log(`✅ Backend memory restored: ${result.message_count} messages`);
            } else {
                console.warn(`⚠️ Failed to restore backend memory: ${result.message}`);
            }
        } catch (error) {
            console.error('❌ Failed to sync historical messages:', error);
        }
    } else {
        // Show welcome message
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <h2>👋 Welcome to SAM-Agent</h2>
                <p>I can help you with the following tasks:</p>
                <div class="feature-grid">
                    <div class="feature-card">
                        <span class="feature-icon">🧬</span>
                        <h3>Molecule Generation</h3>
                        <p>Generate SAM molecules based on scaffolds and anchoring groups</p>
                    </div>
                    <div class="feature-card">
                        <span class="feature-icon">📊</span>
                        <h3>Property Prediction</h3>
                        <p>Predict HOMO, LUMO, and dipole moment</p>
                    </div>
                    <div class="feature-card">
                        <span class="feature-icon">⚡</span>
                        <h3>Device Evaluation</h3>
                        <p>Evaluate perovskite solar cell efficiency</p>
                    </div>
                    <div class="feature-card">
                        <span class="feature-icon">🔄</span>
                        <h3>Synthesis Planning</h3>
                        <p>Generate retrosynthetic routes</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Update session list
    updateSessionList();
    
    console.log(`✅ Switched to session: ${sessionId}`);
}

// Update session list in sidebar
function updateSessionList() {
    const conversationList = document.getElementById('conversation-list');
    if (!conversationList) return;
    
    conversationList.innerHTML = '';
    
    // Show sessions in order
    sessionOrder.forEach(sessionId => {
        const session = sessions[sessionId];
        if (!session) return;
        
        const isActive = sessionId === currentSessionId;
        const item = document.createElement('div');
        item.className = 'conversation-item' + (isActive ? ' active' : '');
        
        const preview = session.messages.length > 0
            ? (session.messages.find(m => m.role === 'user')?.content.substring(0, 40) || 'Empty chat')
            : 'Start a new conversation...';
        
        item.innerHTML = `
            <div class="conversation-content" onclick="switchSession('${sessionId}')">
                <div class="conversation-title">${escapeHtml(session.title)}</div>
                <div class="conversation-preview">${escapeHtml(preview)}</div>
            </div>
            <div class="conversation-actions">
                <button class="action-btn action-rename" onclick="event.stopPropagation(); renameSession('${sessionId}')" title="Rename">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                    </svg>
                </button>
                <button class="action-btn action-delete" onclick="event.stopPropagation(); deleteSession('${sessionId}')" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                </button>
            </div>
        `;
        
        conversationList.appendChild(item);
    });
    
    // Limit to 50 sessions
    if (sessionOrder.length > 50) {
        const toRemove = sessionOrder.slice(50);
        toRemove.forEach(id => {
            delete sessions[id];
        });
        sessionOrder = sessionOrder.slice(0, 50);
        saveSessions();
    }
}

// Rename session
function renameSession(sessionId) {
    const session = sessions[sessionId];
    if (!session) return;
    
    const newTitle = prompt('Rename session:', session.title);
    if (newTitle && newTitle.trim() && newTitle !== session.title) {
        session.title = newTitle.trim();
        saveSessions();
        updateSessionList();
        console.log(`✏️ Session renamed: ${sessionId} -> "${newTitle}"`);
    }
}

// Delete session
function deleteSession(sessionId) {
    const session = sessions[sessionId];
    if (!session) return;
    
    const confirmMsg = `Are you sure you want to delete the session "${session.title}"?\n\nThis will delete all messages in this session and cannot be undone.`;
    if (!confirm(confirmMsg)) return;
    
    // Delete session
    delete sessions[sessionId];
    sessionOrder = sessionOrder.filter(id => id !== sessionId);
    
    // If deleting the current session, switch to the latest session or create a new one
    if (sessionId === currentSessionId) {
        if (sessionOrder.length > 0) {
            switchSession(sessionOrder[0]);
        } else {
            // No other sessions, create a new one
            newChat();
        }
    }
    
    // Clear backend memory
    fetch('/api/clear', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: sessionId
        })
    }).catch(err => console.error('Failed to clear backend memory:', err));
    
    saveSessions();
    updateSessionList();
    console.log(`🗑️ Session deleted: ${sessionId}`);
}

// Save sessions to localStorage
function saveSessions() {
    try {
        localStorage.setItem('perovskite_sessions', JSON.stringify(sessions));
        localStorage.setItem('perovskite_session_order', JSON.stringify(sessionOrder));
        localStorage.setItem('perovskite_current_session', currentSessionId);
        console.log('✅ Sessions saved to localStorage:', {
            sessionCount: Object.keys(sessions).length,
            currentSessionId: currentSessionId,
            currentMessages: sessions[currentSessionId]?.messages.length || 0
        });
    } catch (error) {
        console.error('❌ Failed to save sessions:', error);
    }
}

// Load sessions from localStorage
function loadSessions() {
    try {
        const savedSessions = localStorage.getItem('perovskite_sessions');
        const savedOrder = localStorage.getItem('perovskite_session_order');
        const savedCurrent = localStorage.getItem('perovskite_current_session');
        
        console.log('📥 Loading sessions from localStorage...', {
            hasSessions: !!savedSessions,
            hasOrder: !!savedOrder,
            hasCurrent: !!savedCurrent
        });
        
        if (savedSessions) {
            sessions = JSON.parse(savedSessions);
            console.log('✅ Loaded', Object.keys(sessions).length, 'sessions');
        }
        
        if (savedOrder) {
            sessionOrder = JSON.parse(savedOrder);
            console.log('✅ Session order:', sessionOrder);
        }
        
        if (savedCurrent && sessions[savedCurrent]) {
            currentSessionId = savedCurrent;
            // Load current session messages
            const session = sessions[currentSessionId];
            if (session && session.messages.length > 0) {
                const chatContainer = document.getElementById('chat-container');
                chatContainer.innerHTML = '';
                
                session.messages.forEach(msg => {
                    if (msg.role === 'user') {
                        addUserMessage(msg.content, msg.timestamp, true);
                    } else {
                        addAssistantMessage({
                            content: msg.content,
                            molecules: msg.molecules || [],
                            timestamp: msg.timestamp || new Date().toISOString()
                        }, true);
                    }
                });
            }
        }
        
        updateSessionList();
    } catch (error) {
        console.error('Failed to load sessions:', error);
    }
}

