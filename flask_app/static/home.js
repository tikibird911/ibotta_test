const form = document.getElementById('upload-form');
const results = document.getElementById('results');
const spinner = document.getElementById('spinner');
const timerSpan = document.getElementById('timer');
let interval = null;
let timer = 0;

// Handle model upload and training
form.onsubmit = function(e) {
    e.preventDefault();
    results.innerHTML = '';
    spinner.style.display = 'block';
    timer = 0;
    timerSpan.textContent = timer;
    interval = setInterval(() => {
        timer++;
        timerSpan.textContent = timer;
    }, 1000);

    const formData = new FormData(form);
    fetch('/model', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            console.error("HTTP error:", response.status, response.statusText);
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Response from server:", data);
        clearInterval(interval);
        spinner.style.display = 'none';
        if (data.error) {
            results.innerHTML = `<p style="color:red;">${data.error}</p>`;
        } else {
            let html = `<h2>Classification Report</h2><pre>${data.report}</pre>`;
            if (data.importance_table_html) {
                html += `<h2>Feature Importances</h2>${data.importance_table_html}`;
            }
            results.innerHTML = html;

            // Show random customer IDs if available
            if (data.sample_customer_ids && data.sample_customer_ids.length > 0) {
                let table = `<h3>Try These Customer IDs</h3><table border="1"><tr><th>Customer ID</th></tr>`;
                data.sample_customer_ids.forEach(cid => {
                    table += `<tr><td class="sample-customer-id" style="cursor:pointer;color:blue;" data-cid="${cid}">${cid}</td></tr>`;
                });
                table += `</table>`;
                results.innerHTML += table;

                // Add click event to each sample ID
                document.querySelectorAll('.sample-customer-id').forEach(td => {
                    td.onclick = function() {
                        document.getElementById('customer-id-input').value = this.dataset.cid;
                        document.getElementById('customer-form').dispatchEvent(new Event('submit'));
                    };
                });
            }

            // Show the customer query form after model is ready
            document.getElementById('customer-query').style.display = 'block';
        }
    })
    .catch(error => {
        clearInterval(interval);
        spinner.style.display = 'none';
        results.innerHTML = '<p style="color:red;">An error occurred.</p>';
        console.error("Fetch or parsing error:", error);
    });
};

// Handle customer prediction by ID
document.getElementById('customer-form').onsubmit = function(e) {
    e.preventDefault();
    const customerId = document.getElementById('customer-id-input').value;
    const customerResults = document.getElementById('customer-results');
    customerResults.innerHTML = 'Loading...';
    fetch(`/predict_by_customer?customer_id=${encodeURIComponent(customerId)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                customerResults.innerHTML = `<p style="color:red;">${data.error}</p>`;
            } else {
                let html = `<h3>Predictions</h3><pre>${JSON.stringify(data.predictions, null, 2)}</pre>`;
                if (data.top_features) {
                    html += `<h4>Top Features</h4><ul>`;
                    data.top_features.forEach(f => {
                        html += `<li>${f.feature}: ${f.importance.toFixed(4)}</li>`;
                    });
                    html += `</ul>`;
                }
                if (data.top_feature_averages) {
                    html += `<h4>Top Feature Averages</h4><ul>`;
                    for (const [feat, avg] of Object.entries(data.top_feature_averages)) {
                        html += `<li>${feat}: ${avg.toFixed(4)}</li>`;
                    }
                    html += `</ul>`;
                }
                customerResults.innerHTML = html;
            }
        })
        .catch(error => {
            customerResults.innerHTML = '<p style="color:red;">An error occurred.</p>';
            console.error(error);
        });
};