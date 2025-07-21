// script.js

// Surrogate model function
function surrogateYield(T, t, R) {
    return 100 - Math.sqrt(
      0.02 * Math.pow(T - 925, 2) +
      0.001 * Math.pow(t - 1580, 2) +
      20 * Math.pow(R - 0.85, 2)
    );
  }
  
  document.addEventListener('DOMContentLoaded', () => {
    // Sliders
    const tempSlider = document.getElementById('temp');
    const timeSlider = document.getElementById('time');
    const rateSlider = document.getElementById('rate');
  
    // Value spans
    const tempVal = document.getElementById('temp-val');
    const timeVal = document.getElementById('time-val');
    const rateVal = document.getElementById('rate-val');
  
    // Canvas contexts
    const yieldCtx = document.getElementById('yieldChart').getContext('2d');
    const paretoCtx = document.getElementById('paretoChart').getContext('2d');
  
    // Generate sample data for Pareto plot
    const samples = [];
    for (let i = 0; i < 50; i++) {
      const ct = Math.random() * 9 + 1;                         // cycle time
      const yr = 100 - (ct * 6) + (Math.random() * 4 - 2);      // yield rate
      samples.push({ x: ct, y: yr });
    }
    // Compute Pareto frontier
    const pareto = samples
      .filter(a => !samples.some(b =>
        (b.x <= a.x && b.y >= a.y) && (b.x < a.x || b.y > a.y)
      ))
      .sort((a, b) => a.x - b.x);
  
    // Initialize Yield vs Temperature chart
    const yieldChart = new Chart(yieldCtx, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'Yield vs Temperature', data: [], borderColor: 'blue' }] },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'Temperature (°C)' } },
          y: { title: { display: true, text: 'Predicted Yield (%)' } }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  
    // Initialize Pareto Frontier chart
    const paretoChart = new Chart(paretoCtx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Samples',
            data: samples,
            backgroundColor: 'gray',
            pointRadius: 4
          },
          {
            label: 'Pareto Frontier',
            data: pareto,
            borderColor: 'purple',
            showLine: true,
            fill: false,
            pointBackgroundColor: 'purple',
            pointRadius: 5
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'Cycle Time (hours)' } },
          y: { title: { display: true, text: 'Yield Rate (%)' } }
        }
      }
    });
  
    // Update function
    function updateDashboard() {
      const T = +tempSlider.value;
      const t = +timeSlider.value;
      const R = +rateSlider.value;
  
      // Update value displays
      tempVal.innerText = T;
      timeVal.innerText = t;
      rateVal.innerText = R.toFixed(2);
  
      // Compute new Yield vs Temperature data
      const labels = [];
      const data = [];
      for (let Tv = 800; Tv <= 950; Tv += 1) {
        labels.push(Tv);
        data.push(surrogateYield(Tv, t, R));
      }
      yieldChart.data.labels = labels;
      yieldChart.data.datasets[0].data = data;
      yieldChart.update();
  
      // (Pareto chart is static)
    }
  
    // Bind sliders to update
    [tempSlider, timeSlider, rateSlider].forEach(slider =>
      slider.addEventListener('input', updateDashboard)
    );
  
    // Initial render
    updateDashboard();
  });
  
