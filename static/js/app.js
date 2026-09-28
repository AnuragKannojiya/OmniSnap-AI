/**
 * OmniSnap AI — Shared Client Utilities
 * Designed for Qualcomm® Snapdragon® X Elite & HP OmniBook PCs
 */

document.addEventListener('DOMContentLoaded', () => {
    // Top telemetry sync
    syncTopTelemetry();
    setInterval(syncTopTelemetry, 4000);
});

async function syncTopTelemetry() {
    try {
        const res = await fetch('/api/telemetry');
        if(!res.ok) return;
        const data = await res.json();
        
        const topPower = document.getElementById('top-power');
        const topProvider = document.getElementById('top-provider');
        
        if (topPower && data.power_draw_w !== undefined) {
            topPower.textContent = `${data.power_draw_w.toFixed(1)} W`;
        }
        if (topProvider && data.npu_provider) {
            topProvider.textContent = data.npu_provider.includes('Qnn') ? 'QNN HTP' : 'DirectML / CPU';
        }
    } catch(e) {
        // Silent fallback
    }
}

/**
 * Clean Toast notification
 */
function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-stack';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <span style="font-weight: 600;">${type === 'error' ? 'Error' : 'Notification'}:</span>
        <span>${message}</span>
    `;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(8px)';
        toast.style.transition = 'all 0.2s ease';
        setTimeout(() => toast.remove(), 200);
    }, 3000);
}
