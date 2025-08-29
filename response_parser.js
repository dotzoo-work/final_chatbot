/**
 * Response Parser for Dr. Meenakshi Tomar Dental Chatbot
 * Clean, compact formatting like professional examples
 */

class ResponseParser {
    constructor() {
        this.emojiMap = {
            '🦷': '<span class="emoji dental">🦷</span>',
            '😊': '<span class="emoji friendly">😊</span>',
            '🚨': '<span class="emoji emergency">🚨</span>',
            '💡': '<span class="emoji tips">💡</span>',
            '📅': '<span class="emoji appointment">📅</span>',
            '📸': '<span class="emoji diagnostic">📸</span>'
        };
    }

    /**
     * Parse AI response into clean HTML
     */
    parseResponse(rawResponse) {
        if (!rawResponse || typeof rawResponse !== 'string') {
            return '<p>No response available.</p>';
        }

        let cleanResponse = rawResponse.trim();
        
        // Remove any remaining standalone URLs
        cleanResponse = cleanResponse.replace(/\s*https?:\/\/[^\s]*/g, '');
        
        // Split into paragraphs
        let paragraphs = cleanResponse.split(/\n\n+/);
        let html = '';

        for (let paragraph of paragraphs) {
            paragraph = paragraph.trim();
            if (!paragraph) continue;

            if (this.hasBulletPoints(paragraph)) {
                html += this.formatBulletSection(paragraph);
            } else {
                html += this.formatParagraph(paragraph);
            }
        }

        return html || '<p>Response formatting error.</p>';
    }

    /**
     * Check if section has bullet points
     */
    hasBulletPoints(text) {
        return /^[•\-\*]\s+/m.test(text) || text.includes('•');
    }

    /**
     * Format bullet section cleanly
     */
    formatBulletSection(text) {
        let lines = text.split('\n');
        let intro = '';
        let bullets = [];
        let inBullets = false;

        for (let line of lines) {
            line = line.trim();
            if (!line) continue;

            if (line.match(/^[•\-\*]\s+/)) {
                inBullets = true;
                let bulletText = line.replace(/^[•\-\*]\s+/, '');
                // Process links/bold first, then emojis
                bulletText = this.processBoldText(bulletText);
                bulletText = this.processEmojis(bulletText);
                bullets.push(bulletText);
            } else if (!inBullets) {
                intro += (intro ? ' ' : '') + line;
            }
        }

        let html = '';
        
        if (intro) {
            intro = this.processBoldText(intro);
            intro = this.processEmojis(intro);
            if (intro.includes(':')) {
                html += `<p class="clean-paragraph"><strong>${intro}</strong></p>`;
            } else {
                html += `<p class="clean-paragraph">${intro}</p>`;
            }
        }

        if (bullets.length > 0) {
            html += '<ul class="clean-list">';
            for (let bullet of bullets) {
                // Bullets are already fully processed
                html += `<li class="clean-item">${bullet}</li>`;
            }
            html += '</ul>';
        }

        return html;
    }

    /**
     * Format regular paragraph
     */
    formatParagraph(text) {
        if (!text) return '';
        
        text = text.replace(/\n/g, ' ');
        // Process links first, then emojis
        text = this.processBoldText(text);
        text = this.processEmojis(text);
        
        return `<p class="clean-paragraph">${text}</p>`;
    }

    /**
     * Process emojis (avoid processing emojis inside markdown links)
     */
    processEmojis(text) {
        // Skip emoji processing if text contains markdown links
        if (text.includes('[') && text.includes('](')) {
            return text;
        }
        
        // Remove 📍 emoji completely
        text = text.replace(/📍/g, '');
        
        for (let [emoji, replacement] of Object.entries(this.emojiMap)) {
            text = text.replace(new RegExp(emoji, 'g'), replacement);
        }
        return text;
    }

    /**
     * Process bold text and links
     */
    processBoldText(text) {
        // Handle markdown links [text](url) - convert to HTML links
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function(match, linkText, url) {
            // Remove 📍 emoji from link text
            linkText = linkText.replace(/📍\s*/g, '');
            return `<a href="${url}" target="_blank" style="color: #4f46e5 !important; text-decoration: underline !important; font-weight: 700 !important; cursor: pointer !important; background: rgba(79, 70, 229, 0.1) !important; padding: 2px 6px !important; border-radius: 4px !important;">${linkText}</a>`;
        });
        
        // Handle all **text** patterns globally
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        return text;
    }

    /**
     * Extract question
     */
    extractQuestion(text) {
        let sentences = text.split(/[.!?]+/);
        for (let sentence of sentences.reverse()) {
            if (sentence.includes('?')) {
                return sentence.trim() + '?';
            }
        }
        return null;
    }

    /**
     * Create clean response card
     */
    createResponseCard(rawResponse) {
        let content = this.parseResponse(rawResponse);
        let question = this.extractQuestion(rawResponse);
        
        let html = `<div class="clean-response">${content}`;
        
        if (question) {
            html += `<p class="clean-question">${this.processEmojis(question)}</p>`;
        }
        
        html += `</div>`;
        
        return html;
    }
}

// Export for use in HTML
window.ResponseParser = ResponseParser;