// Real-time dashboard updates
let marketChart = null;
let cgrChart = null;

// Initialize charts
function initializeCharts() {
    // Market price chart
    const marketCtx = document.getElementById('market-chart').getContext('2d');
    marketChart = new Chart(marketCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'BTC-USDT',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });

    // CGR pattern visualization
    const cgrCtx = document.getElementById('cgr-chart').getContext('2d');
    cgrChart = new Chart(cgrCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'CGR Pattern',
                data: [],
                backgroundColor: 'rgba(75, 192, 192, 0.5)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            }
        }
    });
}

// Update dashboard with new data
function updateDashboard(data) {
    // Update portfolio metrics
    document.getElementById('total-value').textContent = `$${data.total_value.toFixed(2)}`;
    document.getElementById('daily-pnl').textContent = `${data.daily_pnl >= 0 ? '+' : ''}$${data.daily_pnl.toFixed(2)}`;
    document.getElementById('open-positions').textContent = data.open_positions;
    document.getElementById('quantum-coherence').textContent = `${(data.quantum_coherence * 100).toFixed(0)}%`;

    // Update market chart
    if (marketChart) {
        marketChart.data.labels.push(new Date().toLocaleTimeString());
        marketChart.data.datasets[0].data.push(data.current_price);
        
        // Keep last 50 data points
        if (marketChart.data.labels.length > 50) {
            marketChart.data.labels.shift();
            marketChart.data.datasets[0].data.shift();
        }
        marketChart.update();
    }

    // Update CGR visualization
    if (cgrChart && data.cgr_points) {
        cgrChart.data.datasets[0].data = data.cgr_points;
        cgrChart.update();
    }

    // Update active trades table
    const tradesTable = document.getElementById('active-trades');
    tradesTable.innerHTML = '';
    
    data.active_trades.forEach(trade => {
        const row = document.createElement('tr');
        const pnlClass = trade.pnl >= 0 ? 'text-success' : 'text-danger';
        
        row.innerHTML = `
            <td>${trade.symbol}</td>
            <td>${trade.type}</td>
            <td>$${trade.entry_price.toFixed(2)}</td>
            <td>$${trade.current_price.toFixed(2)}</td>
            <td class="${pnlClass}">${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}%</td>
            <td>${trade.quantum_score.toFixed(2)}</td>
            <td>
                <button class="btn btn-sm btn-danger" onclick="closeTrade('${trade.id}')">
                    Close
                </button>
            </td>
        `;
        tradesTable.appendChild(row);
    });
}

// Close trade function
async function closeTrade(tradeId) {
    try {
        const response = await fetch('/api/close_trade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ trade_id: tradeId })
        });
        
        if (!response.ok) {
            throw new Error('Failed to close trade');
        }
        
        // Refresh data after closing trade
        fetchDashboardData();
    } catch (error) {
        console.error('Error closing trade:', error);
        alert('Failed to close trade. Please try again.');
    }
}

// Fetch dashboard data
async function fetchDashboardData() {
    try {
        const response = await fetch('/api/dashboard_data');
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    fetchDashboardData();
    
    // Update every 5 seconds
    setInterval(fetchDashboardData, 5000);
});
