document.addEventListener('DOMContentLoaded', function() {
    // Auto-uppercase stock tickers
    document.getElementById('ticker').addEventListener('input', function(e) {
        this.value = this.value.toUpperCase();
    });

    // Form submission handling
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        const ticker = document.getElementById('ticker').value.trim();
        if (!ticker) {
            e.preventDefault();
            alert('Please enter a stock ticker');
        }
    });
});

document.addEventListener('DOMContentLoaded', function() {
    // Function to add new features dynamically
    function addFeature(name, url, iconClass) {
        const featuresMenu = document.getElementById('featuresMenu');
        const newItem = document.createElement('li');
        newItem.innerHTML = `
            <a class="dropdown-item" href="${url}">
                <i class="bi ${iconClass} me-2"></i> ${name}
            </a>
        `;
        featuresMenu.appendChild(newItem);
    }

    // Example: Add a new feature (you can call this from anywhere)
    // addFeature('Market Analysis', '/market-analysis', 'bi-globe');

    // You can also load features from an API
    function loadFeatures() {
        fetch('/api/features')
            .then(response => response.json())
            .then(features => {
                features.forEach(feature => {
                    addFeature(feature.name, feature.url, feature.icon);
                });
            });
    }

    // Call this when needed (e.g., after login)
    // loadFeatures();
});