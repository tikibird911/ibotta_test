        const form = document.getElementById('upload-form');
        const spinner = document.getElementById('spinner');
        const results = document.getElementById('results');
        const timerSpan = document.getElementById('timer');
        let timer = 0, interval;

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
            fetch('{{ url_for("upload") }}', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                clearInterval(interval);
                spinner.style.display = 'none';
                if (data.error) {
                    results.innerHTML = `<p style="color:red;">${data.error}</p>`;
                } else {
                    let html = `<h2>Classification Report</h2><pre>${data.report}</pre>`;
                    if (data.plot_url) {
                        html += `<img src="${data.plot_url}" alt="Feature Importance Plot">`;
                    }
                    results.innerHTML = html;
                }
            })
            .catch(() => {
                clearInterval(interval);
                spinner.style.display = 'none';
                results.innerHTML = '<p style="color:red;">An error occurred.</p>';
            });
        };