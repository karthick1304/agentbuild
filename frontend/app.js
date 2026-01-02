/**
 * 🤖 Multi-Agent Chatbot Frontend
 * ================================
 * Connects to the FastAPI backend running LangGraph agents
 */

const API_URL = 'http://localhost:8000';

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

// Agent icons mapping
const agentIcons = {
    'SCIENTIST': '🔬',
    'CREATIVE': '🎨',
    'CODER': '💻'
};

/**
 * Add a message to the chat
 */
function addMessage(content, type, agentType = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    if (agentType) {
        messageDiv.classList.add(agentType.toLowerCase());
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (type === 'agent' && agentType) {
        const label = document.createElement('div');
        label.className = `agent-label ${agentType.toLowerCase()}`;
        label.innerHTML = `${agentIcons[agentType] || '🤖'} ${agentType} Agent`;
        contentDiv.appendChild(label);
    }
    
    // Process content for code blocks and formatting
    const textDiv = document.createElement('div');
    textDiv.innerHTML = formatMessage(content);
    contentDiv.appendChild(textDiv);
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

/**
 * Format message content (handle code blocks, etc.)
 */
function formatMessage(content) {
    // Handle code blocks
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || 'text'}">${escapeHtml(code.trim())}</code></pre>`;
    });
    
    // Handle inline code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Handle line breaks
    content = content.replace(/\n/g, '<br>');
    
    return content;
}

/**
 * Escape HTML entities
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show loading indicator
 */
function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message agent loading';
    loadingDiv.id = 'loadingMessage';
    
    loadingDiv.innerHTML = `
        <div class="message-content">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
    `;
    
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Remove loading indicator
 */
function hideLoading() {
    const loadingDiv = document.getElementById('loadingMessage');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

/**
 * Send message to the API
 */
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Add user message
    addMessage(message, 'user');
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Disable send button
    sendBtn.disabled = true;
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide loading
        hideLoading();
        
        // Add agent response
        addMessage(data.response, 'agent', data.agent_used);
        
    } catch (error) {
        console.error('Error:', error);
        hideLoading();
        addMessage(
            '⚠️ Error connecting to the server. Make sure the backend is running on http://localhost:8000',
            'agent'
        );
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

/**
 * Auto-resize textarea
 */
function autoResize() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

userInput.addEventListener('input', autoResize);

// Focus input on load
userInput.focus();

// Health check on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✅ Backend connected');
        }
    } catch (error) {
        console.warn('⚠️ Backend not available. Start with: cd backend && uvicorn app:app --reload');
    }
}

checkHealth();

