// On-screen text: subtitles, service heartbeats, stage hints and notices, in Hebrew or English.
export const STR = {
  he: {
    tagline: 'עצמו עיניים. ועכשיו פקחו אותן.', enter: 'היכנסו לחלום',
    hint: 'החזיקו את הכדור המאיר (או את מקש הרווח) ודברו. אפשר גם לכתוב.',
    say: 'או כתבו כאן…', newDream: 'חלום חדש',
    stages: { hearing: 'מפענחים את הקול…', dreaming: 'החלום נרקם…', recording: 'מקשיבים…', idle: '' },
    notices: { busy: 'רגע… החלום עוד נרקם', stt_down: 'השמיעה לא זמינה כרגע. אפשר לכתוב.', mic: 'אין גישה למיקרופון. אפשר לכתוב.', link: 'החיבור לחלום אבד. מתחברים מחדש…', short: 'החזיקו את הכדור בזמן שאתם מדברים' },
    services: { web: 'אתר', director: 'מספר הסיפור', voice: 'קול', imagery: 'תמונות' },
    fin: 'החלום התפוגג אל האור.', again: 'לחלום שוב',
  },
  en: {
    tagline: 'Close your eyes. Now open them.', enter: 'Enter the dream',
    hint: 'Hold the glowing orb (or the space bar) and speak. You can also type.',
    say: 'or type here…', newDream: 'New dream',
    stages: { hearing: 'listening to your words…', dreaming: 'the dream is weaving…', recording: 'listening…', idle: '' },
    notices: { busy: 'One moment… the dream is still weaving', stt_down: "Hearing isn't available right now. You can type.", mic: 'No microphone access. You can type.', link: 'Lost the dream. Reconnecting…', short: 'Hold the orb while you speak' },
    services: { web: 'web', director: 'storyteller', voice: 'voice', imagery: 'images' },
    fin: 'The dream dissolves into light.', again: 'Dream again',
  },
};

export class UI {
  constructor() {
    this.el = (id) => document.getElementById(id);
    this.lang = 'he';
    this.dreamerTimer = null; this.playerTimer = null; this.noticeTimer = null;
  }

  setLang(lang) {
    this.lang = lang === 'en' ? 'en' : 'he';
    const s = STR[this.lang];
    document.documentElement.lang = this.lang;
    this.el('tagline').textContent = s.tagline; this.el('enter').textContent = s.enter; this.el('hint').textContent = s.hint;
    this.el('say').placeholder = s.say; this.el('new-dream').textContent = s.newDream;
    this.el('fin-text').textContent = s.fin; this.el('fin-again').textContent = s.again;
    if (this.lastServices) this.setStatus(this.lastServices);
  }

  dreamer(text) {
    const p = this.el('dreamer-line');
    p.classList.remove('show');
    clearTimeout(this.dreamerTimer);
    setTimeout(() => { p.textContent = text; p.classList.add('show'); }, 250);
    this.dreamerTimer = setTimeout(() => p.classList.remove('show'), 16000 + text.length * 60);
  }

  player(text) {
    const p = this.el('player-line'); p.textContent = `“${text}”`; p.classList.add('show');
    clearTimeout(this.playerTimer); this.playerTimer = setTimeout(() => p.classList.remove('show'), 9000);
  }

  stage(name) {
    this.el('stage').textContent = STR[this.lang].stages[name] ?? '';
    this.el('talk').classList.toggle('waiting', name === 'hearing' || name === 'dreaming');
  }

  notice(key) {
    const n = this.el('notice'); n.textContent = STR[this.lang].notices[key] || key; n.classList.add('show');
    clearTimeout(this.noticeTimer); this.noticeTimer = setTimeout(() => n.classList.remove('show'), 4500);
  }

  setStatus(services) {
    this.lastServices = services;
    const ul = this.el('services'); const names = STR[this.lang].services;
    for (const key of ['web', 'director', 'voice', 'imagery']) {
      const s = services[key]; if (!s) continue;
      let li = ul.querySelector(`[data-k="${key}"]`);
      if (!li) { li = document.createElement('li'); li.dataset.k = key; li.innerHTML = '<span class="dot"></span><span class="name"></span>'; ul.appendChild(li); }
      const prev = li.dataset.status;
      li.className = s.status || 'down'; li.dataset.status = s.status;
      li.querySelector('.name').textContent = names[key];
      const extra = [];
      if (s.model) extra.push(s.model);
      if (s.tokens_per_s) extra.push(`${s.tokens_per_s} tok/s`);
      if (s.stt) extra.push(`STT: ${s.stt.state}`);
      if (s.tts) extra.push(`TTS: ${s.tts.state}`);
      if (s.workers !== undefined) extra.push(`GPU workers: ${s.workers}`);
      li.title = [s.detail, ...extra].filter(Boolean).join(' · ');
      if (prev && prev !== s.status) { li.classList.add('beat'); setTimeout(() => li.classList.remove('beat'), 700); }
    }
  }

  setMicLevel(v) { this.el('talk').style.setProperty('--level', (1 + v * 0.35).toFixed(3)); }

  timings(m) {
    const el = this.el('timings');
    el.textContent = ['stt_s', 'director_s', 'tts_s', 'total_s'].filter((k) => m[k] !== undefined).map((k) => `${k.replace('_s', '')} ${m[k]}s`).join(' · ');
  }
}
