// On-screen text: subtitles, service heartbeats, stage hints and notices, in Hebrew or English.
export const STR = {
  he: {
    tagline: 'עצמו עיניים. ועכשיו פקחו אותן.', enter: 'היכנסו לחלום',
    hint: 'החזיקו את הכדור המאיר (או את מקש הרווח) ודברו. אפשר גם לכתוב.',
    say: 'או כתבו כאן…', newDream: 'חלום חדש',
    stages: { hearing: 'מפענחים את הקול…', dreaming: 'החלום נרקם…', recording: 'מקשיבים…', idle: '' },
    notices: { busy: 'שמעתי. עוד רגע החלום עונה…', stt_down: 'השמיעה לא זמינה כרגע. אפשר לכתוב.', mic: 'אין גישה למיקרופון. אפשר לכתוב.', link: 'החיבור לחלום אבד. מתחברים מחדש…', short: 'החזיקו את הכדור בזמן שאתם מדברים',
      not_heard: 'לא נקלטו מילים. החזיקו את הכדור לאורך כל הדיבור, או כתבו.', mic_silent: 'המיקרופון שקט לגמרי. בדקו שהוא לא מושתק ושנבחר המיקרופון הנכון.', stumbled: 'מספר הסיפור איבד את החוט. נסו לומר את זה שוב.' },
    services: { web: 'אתר', director: 'מספר הסיפור', voice: 'קול', imagery: 'תמונות' },
    parts: { director: 'מספר הסיפור', stt: 'שמיעה', tts: 'קול' },
    waking: 'החלום מתעורר…', ready: 'החלום מוכן.',
    waiting: { director: 'מספר הסיפור עוד מתעורר. המילים שלכם מחכות ויענו ברגע שיהיה מוכן.', stt: 'השמיעה עוד נטענת. ההקלטה שלכם מחכה ותפוענח ברגע שתהיה מוכנה.' },
    fin: 'החלום התפוגג אל האור.', again: 'לחלום שוב',
  },
  en: {
    tagline: 'Close your eyes. Now open them.', enter: 'Enter the dream',
    hint: 'Hold the glowing orb (or the space bar) and speak. You can also type.',
    say: 'or type here…', newDream: 'New dream',
    stages: { hearing: 'listening to your words…', dreaming: 'the dream is weaving…', recording: 'listening…', idle: '' },
    notices: { busy: 'Heard you. The dream answers in a moment…', stt_down: "Hearing isn't available right now. You can type.", mic: 'No microphone access. You can type.', link: 'Lost the dream. Reconnecting…', short: 'Hold the orb while you speak',
      not_heard: 'No words came through. Hold the orb the whole time you speak, or type.', mic_silent: 'The microphone is completely silent. Check that it is unmuted and the right one is selected.', stumbled: 'The storyteller lost the thread. Try saying that again.' },
    services: { web: 'web', director: 'storyteller', voice: 'voice', imagery: 'images' },
    parts: { director: 'storyteller', stt: 'hearing', tts: 'voice' },
    waking: 'The dream is waking up…', ready: 'The dream is ready.',
    waiting: { director: 'The storyteller is still waking up. Your words are waiting and will be answered as soon as it is ready.', stt: 'Hearing is still loading. Your recording is waiting and will be heard as soon as it is ready.' },
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
    if (this.lastServices) { this.setStatus(this.lastServices); this.waking(this.lastServices); }
  }

  dreamer(text) {
    const p = this.el('dreamer-line');
    clearTimeout(this.dreamerTimer);
    const shown = p.classList.contains('show') ? p.textContent : '';
    if (shown && text.startsWith(shown)) {
      p.textContent = text;                 // the rest of the same reply grows in place
    } else {
      p.classList.remove('show');
      setTimeout(() => { p.textContent = text; p.classList.add('show'); }, 250);
    }
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

  notice(key, detail = '') {
    const n = this.el('notice'); n.textContent = STR[this.lang].notices[key] || key; n.title = detail || ''; n.classList.add('show');
    clearTimeout(this.noticeTimer); this.noticeTimer = setTimeout(() => n.classList.remove('show'), 6500);
  }

  // What is still loading, in words, with download progress. Empty when everything is ready.
  wakingLines(services) {
    const s = STR[this.lang], lines = [];
    const dir = services.director, voice = services.voice || {};
    if (dir && dir.status !== 'ok') lines.push(`${s.parts.director}: ${dir.detail || dir.status}`);
    for (const part of ['stt', 'tts']) {
      const p = voice[part];
      if (!p) { if (voice.status && voice.status !== 'ok') lines.push(`${s.parts[part]}: ${voice.detail || voice.status}`); continue; }
      if (p.state !== 'ready') lines.push(`${s.parts[part]}: ${p.detail || p.state}`);
    }
    return [...new Set(lines)];
  }

  waking(services) {
    const lines = this.wakingLines(services);
    for (const id of ['waking', 'veil-status']) {
      const el = this.el(id); if (!el) continue;
      if (!lines.length) { el.classList.remove('show'); el.replaceChildren(); continue; }
      const head = document.createElement('div'); head.className = 'head'; head.textContent = STR[this.lang].waking;
      el.replaceChildren(head, ...lines.map((t) => { const d = document.createElement('div'); d.textContent = t; d.dir = 'auto'; return d; }));
      el.classList.add('show');
    }
    if (!lines.length) this.clearWaiting();
  }

  waiting(part) {
    const w = this.el('waiting'); w.textContent = STR[this.lang].waiting[part] || ''; w.classList.add('show');
  }

  clearWaiting() { this.el('waiting').classList.remove('show'); }

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
    const names = { stt_s: 'hearing', first_words_s: 'first words', director_s: 'storyteller', tts_s: 'voice', to_first_sound_s: 'first sound', total_s: 'total' };
    el.textContent = Object.keys(names).filter((k) => m[k] !== undefined).map((k) => `${names[k]} ${m[k]}s`).join(' · ');
  }
}
