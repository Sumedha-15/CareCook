/* ── CareCook — app.js ──────────────────────────────────────────────────── */

/* ── State ───────────────────────────────────────────────────────────────── */
const state = {
  selectedConditions: new Set(['none']),
  safeIngredients: [],
  currentDish: null,
  currentRecipe: null,
};

/* ── Thresholds (mirrors condition_thresholds.json) ────────────────────── */
const THRESHOLDS = {
  none:            { sugar:10, fat:20,  sodium:300, carbs:50, cholesterol:300 },
  diabetes:        { sugar:3,  fat:15,  sodium:200, carbs:25, cholesterol:200 },
  pcos:            { sugar:3,  fat:12,  sodium:180, carbs:20, cholesterol:150 },
  'high bp':       { sugar:5,  fat:15,  sodium:100, carbs:40, cholesterol:200 },
  'heart disease': { sugar:2,  fat:5,   sodium:80,  carbs:15, cholesterol:100 },
  'kidney disease':{ sugar:5,  fat:10,  sodium:50,  carbs:30, cholesterol:150 },
  obesity:         { sugar:2,  fat:8,   sodium:150, carbs:20, cholesterol:200 },
  jaundice:        { sugar:10, fat:5,   sodium:200, carbs:60, cholesterol:50  },
};

const COND_LABELS = {
  none: 'No condition', diabetes: 'Diabetes', pcos: 'PCOS / PCOD',
  'high bp': 'High Blood Pressure', 'heart disease': 'Heart Disease',
  'kidney disease': 'Kidney Disease', obesity: 'Obesity', jaundice: 'Jaundice',
};

/* ── Sidebar & Tab ───────────────────────────────────────────────────────── */
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('show');
}

function switchTab(btn) {
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  if (btn.dataset.tab === 'saved') loadSaved();
}

/* ── Condition Buttons ───────────────────────────────────────────────────── */
document.getElementById('condGrid').addEventListener('click', e => {
  const btn = e.target.closest('.cond-btn');
  if (!btn) return;
  const cond = btn.dataset.cond;

  if (cond === 'none') {
    state.selectedConditions = new Set(['none']);
    document.querySelectorAll('.cond-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    return;
  }

  document.querySelector('[data-cond="none"]').classList.remove('active');
  state.selectedConditions.delete('none');

  btn.classList.toggle('active');
  if (btn.classList.contains('active')) {
    state.selectedConditions.add(cond);
  } else {
    state.selectedConditions.delete(cond);
  }
  if (state.selectedConditions.size === 0) {
    state.selectedConditions.add('none');
    document.querySelector('[data-cond="none"]').classList.add('active');
  }
});

/* ── Find Dishes ─────────────────────────────────────────────────────────── */
async function findDishes() {
  const ingredients = document.getElementById('ingredientsInput').value.trim();
  if (!ingredients) { showToast('⚠️ Please enter some ingredients'); return; }

  const btn = document.getElementById('findBtn');
  setLoading(btn, true, 'Analysing...');
  showProgress(0, 'Checking ingredients through ML model...');
  clearResults();

  const conditions = [...state.selectedConditions];
  const medical    = [...state.selectedConditions].filter(c => c !== 'none').join(', ');

  try {
    // ── Step 1: safety analysis ──
    showProgress(30, 'Running Random Forest safety check...');
    const condStr = conditions.includes('none') ? 'none' : conditions.join(', ');
    const resp = await fetch('/api/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ingredients, condition: condStr }),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    state.safeIngredients = data.safe;
    renderIngredients(data.safe, data.removed);
    showProgress(60, 'Generating dish suggestions with Llama-3...');

    // ── Step 2: dish suggestions ──
    const dishResp = await fetch('/api/dishes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        safe_ingredients: data.safe,
        condition: medical || condStr,
        allergies:   document.getElementById('allergiesInput').value.trim(),
        constraints: document.getElementById('constraintsInput').value.trim(),
      }),
    });
    const dishData = await dishResp.json();
    if (dishData.error) throw new Error(dishData.error);

    showProgress(100, 'Done!');
    renderDishes(dishData.dishes, dishData.recommendations);

    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });

    const totalRemoved = data.removed.length;
    if (totalRemoved > 0) showToast(`⚠️ ${totalRemoved} ingredient(s) flagged — see safety check below`);
    else showToast('✅ All ingredients cleared by ML safety filter!');
  } catch (err) {
    showToast('❌ ' + (err.message || 'Something went wrong'));
  } finally {
    setLoading(btn, false, '<span class="btn-icon">🔍</span> Find Safe Dishes');
    hideProgress();
  }
}

/* ── Render Ingredients ──────────────────────────────────────────────────── */
function renderIngredients(safe, removed) {
  const grid = document.getElementById('ingGrid');
  grid.innerHTML = '';

  const all = [
    ...safe.map(i => ({ ...i })),
    ...removed.map(i => ({ ...i })),
  ];

  all.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'ing-card';
    card.style.animationDelay = `${idx * 0.05}s`;

    const badgeClass = item.label === 'safe' ? 'badge-safe' : item.label === 'caution' ? 'badge-caution' : 'badge-unsafe';
    const badgeText  = item.label === 'safe' ? '✓ Safe' : item.label === 'caution' ? '⚠ Caution' : '✕ Unsafe';
    const matchedText = item.matched_food && item.matched_food !== item.name
      ? `Matched: ${item.matched_food}`
      : item.explanation || '';

    card.innerHTML = `
      <div class="ing-info">
        <div class="ing-name">${item.name}</div>
        <div class="ing-matched">${matchedText}</div>
      </div>
      <span class="badge ${badgeClass}">${badgeText}</span>`;
    grid.appendChild(card);
  });

  // Subtitle
  const subtitle = document.getElementById('safetySubtitle');
  subtitle.textContent = `${safe.length} safe · ${removed.length} flagged — out of ${all.length} ingredients`;

  // Warning box for removed
  const removedSection = document.getElementById('removedSection');
  const warningBox = document.getElementById('warningBox');
  if (removed.length > 0) {
    removedSection.style.display = 'block';
    warningBox.innerHTML = `<h4>⚠️ Removed ${removed.length} Unsafe Ingredient${removed.length > 1 ? 's' : ''}</h4>` +
      removed.map(item => `
        <div class="warning-item">
          <strong>${item.name}</strong> — ${item.explanation}
          ${item.nutrition && Object.keys(item.nutrition).length ? `
            <div class="nutr-chips">
              ${Object.entries(item.nutrition).map(([k, v]) => `<span class="nutr-chip">${k}: ${Number(v).toFixed(1)}</span>`).join('')}
            </div>` : ''}
        </div>`).join('');
  } else {
    removedSection.style.display = 'none';
  }
}

/* ── Render Dishes ───────────────────────────────────────────────────────── */
function renderDishes(dishes, recommendations) {
  // Tips
  const tipsBox = document.getElementById('tipsBox');
  if (recommendations && recommendations.trim()) {
    document.getElementById('tipsContent').textContent = recommendations;
    tipsBox.style.display = 'block';
  }

  // Dishes
  const grid = document.getElementById('dishesGrid');
  grid.innerHTML = '';

  dishes.forEach((dish, idx) => {
    const card = document.createElement('div');
    card.className = 'dish-card';
    card.style.animationDelay = `${idx * 0.1 + 0.1}s`;

    const rawName   = dish.replace(/^\d+\.\s*/, '');
    const cleanName = rawName.split('(')[0].trim();
    const reason    = rawName.includes('(') ? rawName.split('(')[1].replace(')', '').trim() : '';

    // Lazy-load image
    const imgId = `dishImg_${idx}`;
    card.innerHTML = `
      <div class="dish-img-placeholder" id="${imgId}">🍲</div>
      <div class="dish-body">
        <div class="dish-safe-badge">✅ AI Verified Safe</div>
        <div class="dish-title">${cleanName}</div>
        ${reason ? `<div class="dish-reason">${reason}</div>` : ''}
        <button class="dish-cook-btn" onclick="cookDish('${cleanName.replace(/'/g, "\\'")}', this)">
          👨‍🍳 Cook This Dish
        </button>
      </div>`;
    grid.appendChild(card);

    // Fetch image asynchronously
    fetch(`/api/image?q=${encodeURIComponent(cleanName)}`)
      .then(r => r.json())
      .then(data => {
        if (data.url) {
          const el = document.getElementById(imgId);
          if (el) {
            el.outerHTML = `<img class="dish-img" src="${data.url}" alt="${cleanName}" onerror="this.outerHTML='<div class=\\'dish-img-placeholder\\'>🍲</div>'">`;
          }
        }
      }).catch(() => {});
  });
}

/* ── Cook Dish ───────────────────────────────────────────────────────────── */
async function cookDish(dishName, btn) {
  btn.disabled = true;
  btn.innerHTML = '<span style="display:flex;align-items:center;gap:6px;justify-content:center"><span class="spinner"></span> Generating...</span>';

  document.getElementById('recipeSection').style.display = 'none';

  try {
    const resp = await fetch('/api/recipe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dish_name:   dishName,
        condition:   [...state.selectedConditions].filter(c => c !== 'none').join(', '),
        allergies:   document.getElementById('allergiesInput').value.trim(),
        constraints: document.getElementById('constraintsInput').value.trim(),
      }),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    state.currentDish   = dishName;
    state.currentRecipe = data.recipe;

    document.getElementById('recipeDishTitle').textContent = dishName;
    document.getElementById('recipeContent').textContent   = data.recipe;
    document.getElementById('recipeSection').style.display = 'block';
    document.getElementById('recipeSection').scrollIntoView({ behavior: 'smooth' });
    showToast('🍴 Recipe ready!');
  } catch (err) {
    showToast('❌ ' + (err.message || 'Could not generate recipe'));
  } finally {
    btn.disabled = false;
    btn.textContent = '👨‍🍳 Cook This Dish';
  }
}

/* ── Save Recipe ─────────────────────────────────────────────────────────── */
async function saveCurrentRecipe() {
  if (!state.currentRecipe) return;
  const conditions = [...state.selectedConditions].join(', ');
  try {
    await fetch('/api/recipes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name:      state.currentDish,
        recipe:    state.currentRecipe,
        condition: conditions,
      }),
    });
    showToast('📌 Recipe saved! View in Saved Recipes tab.');
    updateSavedCount();
  } catch {
    showToast('❌ Could not save recipe');
  }
}

/* ── Load Saved Recipes ──────────────────────────────────────────────────── */
async function loadSaved() {
  const list = document.getElementById('savedList');
  try {
    const resp = await fetch('/api/recipes');
    const recipes = await resp.json();

    updateSavedCount(recipes.length);

    if (!recipes.length) {
      list.innerHTML = `<div class="empty-state">
        <div class="empty-icon">🍲</div>
        <div class="empty-title">No saved recipes yet</div>
        <div class="empty-sub">Generate and save recipes from the Generate tab</div>
      </div>`;
      return;
    }

    list.innerHTML = '';
    recipes.forEach((r, idx) => {
      const card = document.createElement('div');
      card.className = 'saved-card';
      card.innerHTML = `
        <div class="saved-card-header" onclick="toggleSaved(this)">
          <div>
            <div class="saved-card-title">📕 ${r.name}</div>
            <div class="saved-card-meta">${r.condition ? `Condition: ${r.condition} · ` : ''}${r.date_saved || ''}</div>
          </div>
          <div class="saved-card-actions">
            <button class="del-btn" onclick="deleteRecipe(${idx}, event)">🗑 Delete</button>
            <button class="toggle-btn">▼</button>
          </div>
        </div>
        <div class="saved-body">${r.recipe}</div>`;
      list.appendChild(card);
    });
  } catch {
    list.innerHTML = '<div class="empty-state"><div class="empty-sub">Could not load saved recipes</div></div>';
  }
}

function toggleSaved(header) {
  header.closest('.saved-card').classList.toggle('open');
  const btn = header.querySelector('.toggle-btn');
  btn.textContent = header.closest('.saved-card').classList.contains('open') ? '▲' : '▼';
}

async function deleteRecipe(idx, e) {
  e.stopPropagation();
  await fetch(`/api/recipes/${idx}`, { method: 'DELETE' });
  loadSaved();
  showToast('🗑 Recipe deleted');
}

/* ── Threshold Table ─────────────────────────────────────────────────────── */
function buildThresholdTable() {
  const tb = document.querySelector('#threshTable tbody');
  Object.entries(THRESHOLDS).forEach(([key, t]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${COND_LABELS[key] || key}</td>
      <td><span class="thresh-val">${t.sugar}g</span></td>
      <td><span class="thresh-val">${t.fat}g</span></td>
      <td><span class="thresh-val">${t.sodium}mg</span></td>
      <td><span class="thresh-val">${t.carbs}g</span></td>
      <td><span class="thresh-val">${t.cholesterol}mg</span></td>`;
    tb.appendChild(tr);
  });
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function setLoading(btn, on, label) {
  btn.disabled = on;
  btn.innerHTML = on
    ? `<span class="spinner"></span> ${label}`
    : label;
}

function showProgress(pct, label) {
  document.getElementById('progressWrap').style.display = 'block';
  document.getElementById('progressBar').style.width = pct + '%';
  document.getElementById('progressLabel').textContent = label;
}

function hideProgress() {
  setTimeout(() => { document.getElementById('progressWrap').style.display = 'none'; }, 600);
}

function clearResults() {
  document.getElementById('resultsSection').style.display = 'none';
  document.getElementById('recipeSection').style.display = 'none';
  document.getElementById('ingGrid').innerHTML = '';
  document.getElementById('dishesGrid').innerHTML = '';
  document.getElementById('tipsBox').style.display = 'none';
  document.getElementById('removedSection').style.display = 'none';
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

let _toastTimer;
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove('show'), 3500);
}

async function updateSavedCount(n) {
  if (n === undefined) {
    try { const r = await fetch('/api/recipes'); const d = await r.json(); n = d.length; } catch { n = 0; }
  }
  document.getElementById('savedBadge').textContent   = n;
  document.getElementById('topSavedCount').textContent = n;
}

/* ── Init ────────────────────────────────────────────────────────────────── */
buildThresholdTable();
updateSavedCount();

// Enter key on ingredients field triggers search
document.getElementById('ingredientsInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') findDishes();
});