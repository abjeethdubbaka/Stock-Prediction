// Activity Feed Refresh
document.getElementById('refreshActivity')?.addEventListener('click', async function() {
    const feed = document.getElementById('activityFeed');
    const btn = this;
    
    btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Loading...';
    btn.disabled = true;
    
    try {
        const response = await fetch('/api/activities/recent');
        const activities = await response.json();
        
        if (activities.length > 0) {
            feed.innerHTML = activities.map(activity => `
                <div class="list-group-item list-group-item-action">
                    <div class="d-flex align-items-center">
                        <div class="activity-icon me-3 bg-${activity.type} text-white">
                            <i class="bi ${getActivityIcon(activity.type)}"></i>
                        </div>
                        <div class="flex-grow-1">
                            <div class="d-flex justify-content-between">
                                <h6 class="mb-1">${activity.title}</h6>
                                <small class="text-muted">${timeAgo(activity.timestamp)}</small>
                            </div>
                            <p class="mb-1 small text-muted">${activity.description}</p>
                        </div>
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to refresh activity feed:', error);
    } finally {
        btn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Refresh';
        btn.disabled = false;
    }
});

// Helper functions
function getActivityIcon(type) {
    const icons = {
        'prediction': 'bi-graph-up',
        'trade': 'bi-journal-text',
        'alert': 'bi-exclamation-triangle',
        'system': 'bi-info-circle'
    };
    return icons[type] || 'bi-activity';
}

function timeAgo(timestamp) {
    // Implement timeago functionality or use a library
    return new Date(timestamp).toLocaleTimeString();
}