/**
 * OmniSnap AI - Main JavaScript file
 * Contains shared utilities and base initializations for all templates
 */

document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    
    // Create toast container if it doesn't exist
    if (!document.getElementById('toast-container')) {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
});

/**
 * Shows a toast notification on the screen.
 * @param {string} message - The message to display
 * @param {string} type - 'info', 'success', 'warning', or 'error'
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    // Simple icon mapping
    const icons = {
        info: 'ℹ️',
        success: '✅',
        warning: '⚠️',
        error: '❌'
    };
    
    const icon = icons[type] || icons.info;
    
    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <span>${icon}</span>
            <span>${message}</span>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => {
            if (container.contains(toast)) {
                container.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

/**
 * Formats a number of seconds into HH:MM:SS or MM:SS format
 * @param {number} seconds - The total seconds
 * @returns {string} Formatted time string
 */
function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "00:00";
    
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    
    const pad = (num) => num.toString().padStart(2, '0');
    
    if (h > 0) {
        return `${pad(h)}:${pad(m)}:${pad(s)}`;
    } else {
        return `${pad(m)}:${pad(s)}`;
    }
}

/**
 * Smoothly animates a number value in an element
 * @param {HTMLElement} element - The DOM element to update
 * @param {number} start - Starting value
 * @param {number} end - Ending value
 * @param {number} duration - Animation duration in ms
 */
function animateValue(element, start, end, duration) {
    if (!element) return;
    
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        
        // Easing function for smoother animation
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        
        const currentVal = Math.floor(easeOutQuart * (end - start) + start);
        element.innerHTML = currentVal;
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            element.innerHTML = end;
        }
    };
    window.requestAnimationFrame(step);
}

/**
 * Initializes the floating particle background
 */
function initParticles() {
    // Check if particles container exists
    let container = document.querySelector('.bg-particles');
    if (!container) {
        container = document.createElement('div');
        container.className = 'bg-particles';
        document.body.prepend(container);
    }
    
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        createParticle(container);
    }
}

/**
 * Creates a single particle with random properties
 * @param {HTMLElement} container - The container to append the particle to
 */
function createParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    // Random size between 2px and 6px
    const size = Math.random() * 4 + 2;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    
    // Random position
    const posX = Math.random() * 100;
    const posY = Math.random() * 100;
    particle.style.left = `${posX}vw`;
    particle.style.top = `${posY}vh`;
    
    // Random animation duration and delay
    const duration = Math.random() * 20 + 10;
    const delay = Math.random() * 5;
    particle.style.animationDuration = `${duration}s`;
    particle.style.animationDelay = `-${delay}s`; // Negative delay so some start already moving
    
    // Random opacity
    particle.style.opacity = Math.random() * 0.5 + 0.1;
    
    container.appendChild(particle);
}
