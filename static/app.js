// Progressive enhancement: live results when typing on /search
document.addEventListener('DOMContentLoaded', () => {
  const live = document.getElementById('live-results');
  if (!live) return;

  const endpoint = live.dataset.endpoint;
  const form = document.querySelector('form.searchbar');
  const input = form?.querySelector('input[type="search"]');
  if (!input || !endpoint) return;

  let handle;
  input.addEventListener('input', () => {
    clearTimeout(handle);
    handle = setTimeout(async () => {
      const q = input.value.trim();
      if (!q) { live.innerHTML = ''; return; }
      try {
        const res = await fetch(`${endpoint}?q=${encodeURIComponent(q)}&k=5`);
        const data = await res.json();
        if (!data.results) return;
        live.innerHTML = `
          <div class="card">
            <div class="muted">Live preview:</div>
            <ul>
              ${data.results.map(r => `
                <li style="margin:6px 0">
                  <span class="mono">${r.similarity.toFixed(4)}</span>
                  — ${r.filename}
                  <a class="btn tiny" href="/view/${r.id}">Open</a>
                </li>
              `).join('')}
            </ul>
          </div>
        `;
      } catch {
        // silently ignore
      }
    }, 250);
  });
});
