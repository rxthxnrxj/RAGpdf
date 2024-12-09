function formatCodeBlocks(text) {
    // Regular expression to match code blocks (text between triple backticks)
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    
    return text.replace(codeBlockRegex, (match, language, code) => {
        return `<div class="code-block-wrapper">
                    <div class="code-block">
                        <div class="code-header">
                            ${language ? `<span class="code-language">${language}</span>` : ''}
                        </div>
                        <pre><code>${escapeHtml(code.trim())}</code></pre>
                    </div>
                    <button class="copy-button" onclick="copyCode(this)">Copy</button>
                </div>`;
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function copyCode(button) {
    const codeBlock = button.previousElementSibling.querySelector('code');
    const text = codeBlock.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        setTimeout(() => {
            button.textContent = originalText;
        }, 2000);
    });
}

function appendMessage(message, isUser = false) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    // Create message header with avatar
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    
    const avatar = document.createElement('img');
    avatar.className = 'message-avatar';
    avatar.src = isUser ? '/static/images/user_avatar.png' : '/static/images/bot_avatar.png';
    avatar.alt = isUser ? 'User Avatar' : 'Bot Avatar';
    
    const name = document.createElement('span');
    name.textContent = isUser ? 'You' : 'Assistant';
    
    headerDiv.appendChild(avatar);
    headerDiv.appendChild(name);
    
    // Create message content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format the message content (including code blocks)
    if (isUser) {
        contentDiv.textContent = message;
    } else {
        contentDiv.innerHTML = formatCodeBlocks(message);
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    
    // Scroll to the bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showLoader(message = 'Processing your file...') {
    document.getElementById('loader').style.display = 'flex';
    document.getElementById('processing-details').textContent = message;
}

function hideLoader() {
    document.getElementById('loader').style.display = 'none';
}

function updateProcessingStatus(message) {
    document.getElementById('processing-details').textContent = message;
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (message) {
        appendMessage(message, true);
        input.value = '';
        
        showLoader('Processing your question...');
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });
            
            const data = await response.json();
            appendMessage(data.response);
        } catch (error) {
            appendMessage('Error: Could not get response');
        } finally {
            hideLoader();
        }
    }
}

async function uploadFiles() {
    const fileInput = document.getElementById('pdf-upload');
    const files = fileInput.files;
    
    if (files.length === 0) return;
    
    showLoader('Starting file upload...');
    
    for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            updateProcessingStatus(`Processing ${file.name}...`);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            
            const data = await response.json();
            
            if (data.error) {
                appendMessage(`Error processing ${file.name}: ${data.error}`);
            } else {
                appendMessage(`File ${file.name}: ${data.message}`);
            }
            
        } catch (error) {
            appendMessage(`Error uploading ${file.name}: ${error.message}`);
        }
    }
    
    hideLoader();
    fileInput.value = '';
}

// Handle Enter key in textarea
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
