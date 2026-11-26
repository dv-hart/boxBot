/* Icon registry. 24×24 viewBox, outline style (stroke comes from .ico in core.css).
   Add an icon from a block file:  ICONS.umbrella = '<path d="…"/>';
   core.js mounts the <symbol> sprite once every script has loaded.
   ico(name) accepts these names and the Lucide names the data sources emit (see ICON_ALIAS). */
const ICONS = {
  sun: '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6"/>',
  partly: '<circle cx="8.5" cy="8.5" r="3.2"/><path d="M8.5 2.3v1.5M2.3 8.5h1.5M4.1 4.1l1.1 1.1M12.9 4.1l-1.1 1.1"/><path d="M9 20h8.2a3.8 3.8 0 0 0 .5-7.57A5 5 0 0 0 8.2 13.6 3.25 3.25 0 0 0 9 20z"/>',
  cloud: '<path d="M7 18.5h10a4.2 4.2 0 0 0 .6-8.36A5.6 5.6 0 0 0 6.9 11 3.8 3.8 0 0 0 7 18.5z"/>',
  rain: '<path d="M7 15h10a4.2 4.2 0 0 0 .6-8.36A5.6 5.6 0 0 0 6.9 7.5 3.8 3.8 0 0 0 7 15z"/><path d="M8.5 18l-1 3M12.5 18l-1 3M16.5 18l-1 3"/>',
  storm: '<path d="M7 14h10a4.2 4.2 0 0 0 .6-8.36A5.6 5.6 0 0 0 6.9 6.5 3.8 3.8 0 0 0 7 14z"/><path d="M12.5 15.5l-2.2 3.2h3l-1.8 3"/>',
  fog: '<path d="M7 13h10a4.2 4.2 0 0 0 .6-8.36A5.6 5.6 0 0 0 6.9 5.5 3.8 3.8 0 0 0 7 13z"/><path d="M4 16.8h16M6.5 20.3h11"/>',
  wind: '<path d="M3 9h10.5a2.5 2.5 0 1 0-2.4-3.2M3 13h15a2.5 2.5 0 1 1-2.4 3.2M3 17h7"/>',
  moon: '<path d="M19.5 14.2A8 8 0 0 1 9.8 4.5a8 8 0 1 0 9.7 9.7z"/>',
  snow: '<path d="M7 14h10a4.2 4.2 0 0 0 .6-8.36A5.6 5.6 0 0 0 6.9 6.5 3.8 3.8 0 0 0 7 14z"/><path d="M8 18.2h.01M12 17.2h.01M16 18.2h.01M10 21h.01M14 21h.01" stroke-width="2.4"/>',
  circle: '<circle cx="12" cy="12" r="8.5"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3"/>',
  reply: '<path d="M10 7L4.5 12 10 17M5 12h8.5a6 6 0 0 1 6 6v.5"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  cal: '<rect x="4" y="5.5" width="16" height="14.5" rx="3"/><path d="M4 10h16M8.5 3.5v4M15.5 3.5v4"/>',
  car: '<path d="M4 16.5v-3.2l1.8-4.6A2 2 0 0 1 7.7 7.5h8.6a2 2 0 0 1 1.9 1.2l1.8 4.6v3.2M4 13.3h16M6.5 16.5v2M17.5 16.5v2"/>',
  sunset: '<path d="M3 18.5h18M6.2 18.5a5.8 5.8 0 0 1 11.6 0M12 4v5M9.6 6.8L12 9.2l2.4-2.4M3.5 13.5l1.8.8M20.5 13.5l-1.8.8"/>',
  sunrise: '<path d="M3 18.5h18M6.2 18.5a5.8 5.8 0 0 1 11.6 0M12 9.2V4M9.6 6.4L12 4l2.4 2.4M3.5 13.5l1.8.8M20.5 13.5l-1.8.8"/>',
  todo: '<rect x="4.5" y="4.5" width="15" height="15" rx="4"/><path d="M8.5 12.3l2.5 2.5 4.6-5"/>',
};

/* Names the data sources emit (Lucide) → the sprite above. Unknown names fall back to `circle`. */
const ICON_ALIAS = {
  'cloud-sun': 'partly', 'cloud-moon': 'partly', 'cloudy': 'cloud', 'cloud-rain': 'rain', 'cloud-drizzle': 'rain',
  'cloud-snow': 'snow', 'snowflake': 'snow', 'cloud-lightning': 'storm', 'cloud-fog': 'fog', 'haze': 'fog',
  'calendar': 'cal', 'check-square': 'todo', 'list-todo': 'todo', 'mail': 'reply',
};
const icoName = n => { n = String(n || '').toLowerCase(); n = ICON_ALIAS[n] || n; return ICONS[n] ? n : 'circle'; };
const ico = (n, st = '') => `<svg class="ico" style="${st}"><use href="#i-${icoName(n)}"/></svg>`;
function mountIcons() {
  const old = document.getElementById('bb-icons'); if (old) old.remove();
  document.body.insertAdjacentHTML('afterbegin', `<svg id="bb-icons" width="0" height="0" style="position:absolute"><defs>${
    Object.entries(ICONS).map(([k, v]) => `<symbol id="i-${k}" viewBox="0 0 24 24">${v}</symbol>`).join('')}</defs></svg>`);
}
