// Sound: microphone capture, the dreamer's voice (with a level meter that makes the
// dreamer glow as it speaks), a browser-voice fallback, and a soft ambient drone.
export class Sound {
  constructor() {
    this.ctx = null; this.level = 0; this.micLevel = 0; this.muted = false;
    this.queue = []; this.playing = null; this.onLevel = () => {};
  }

  unlock() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
      this.master = this.ctx.createGain(); this.master.connect(this.ctx.destination);
      this.analyser = this.ctx.createAnalyser(); this.analyser.fftSize = 512; this.analyser.connect(this.master);
      this.buf = new Uint8Array(this.analyser.fftSize);
      this.ambient = new Ambient(this.ctx, this.master);
      this._meter();
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
  }

  setMuted(m) { this.muted = m; if (this.master) this.master.gain.setTargetAtTime(m ? 0 : 1, this.ctx.currentTime, 0.2); }

  _meter() {
    const tick = () => {
      if (this.playing) {
        this.analyser.getByteTimeDomainData(this.buf);
        let s = 0; for (let i = 0; i < this.buf.length; i++) { const v = (this.buf[i] - 128) / 128; s += v * v; }
        const rms = Math.sqrt(s / this.buf.length);
        this.level += (Math.min(1, rms * 4) - this.level) * 0.35;
      } else this.level *= 0.9;
      if (this.mic) {
        this.mic.analyser.getByteTimeDomainData(this.mic.buf);
        let s = 0; for (let i = 0; i < this.mic.buf.length; i++) { const v = (this.mic.buf[i] - 128) / 128; s += v * v; }
        this.micLevel += (Math.min(1, Math.sqrt(s / this.mic.buf.length) * 5) - this.micLevel) * 0.4;
      } else this.micLevel *= 0.85;
      this.onLevel(this.level, this.micLevel);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------ the dreamer's voice
  async play(url) {
    this.unlock();
    try {
      const data = await (await fetch(url)).arrayBuffer();
      const audio = await this.ctx.decodeAudioData(data);
      this.queue.push(audio); if (!this.playing) this._next();
    } catch (e) { console.warn('could not play voice', e); }
  }

  _next() {
    const audio = this.queue.shift(); if (!audio) { this.playing = null; return; }
    const src = this.ctx.createBufferSource(); src.buffer = audio; src.connect(this.analyser);
    src.onended = () => this._next(); this.playing = src;
    this.ambient && this.ambient.duck(audio.duration);
    src.start();
  }

  stopVoice() { this.queue = []; if (this.playing) { try { this.playing.stop(); } catch { /* already stopped */ } } this.playing = null; speechSynthesis.cancel(); }

  // Fallback when the voice service is down: the browser speaks.
  speakFallback(text, lang, voice) {
    if (!('speechSynthesis' in window) || this.muted) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === 'he' ? 'he-IL' : 'en-US'; u.rate = 0.92; u.pitch = voice === 'male' ? 0.85 : voice === 'female' ? 1.15 : 1;
    const voices = speechSynthesis.getVoices().filter((v) => v.lang && v.lang.toLowerCase().startsWith(lang === 'he' ? 'he' : 'en'));
    const local = voices.find((v) => v.localService) || voices[0]; if (local) u.voice = local;
    let fake = 0; const iv = setInterval(() => { fake = 0.3 + 0.3 * Math.random(); this.level = fake; }, 120);
    u.onend = () => { clearInterval(iv); this.level = 0; };
    speechSynthesis.speak(u);
  }

  // ------------------------------------------------------------------ microphone
  async startRecording() {
    this.unlock();
    this.stopVoice();
    if (!this.stream) {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    }
    const src = this.ctx.createMediaStreamSource(this.stream);
    const analyser = this.ctx.createAnalyser(); analyser.fftSize = 512; src.connect(analyser);
    this.mic = { src, analyser, buf: new Uint8Array(512) };
    const mime = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/mp4'].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
    this.chunks = []; this.mime = mime || 'audio/webm';
    this.rec = new MediaRecorder(this.stream, mime ? { mimeType: mime, audioBitsPerSecond: 48000 } : undefined);
    this.rec.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };
    this.startedAt = performance.now();
    this.rec.start();
  }

  stopRecording() {
    return new Promise((resolve) => {
      if (!this.rec || this.rec.state === 'inactive') return resolve(null);
      const took = performance.now() - this.startedAt;
      this.rec.onstop = () => {
        if (this.mic) { this.mic.src.disconnect(); this.mic = null; }
        resolve(took < 350 ? null : new Blob(this.chunks, { type: this.mime }));
      };
      this.rec.stop();
    });
  }

  setPlace(kind, time) { if (this.ambient) this.ambient.setPlace(kind, time); }
}

// A quiet, slowly shifting drone plus wind; each place tunes it differently.
class Ambient {
  constructor(ctx, out) {
    this.ctx = ctx;
    this.gain = ctx.createGain(); this.gain.gain.value = 0; this.gain.connect(out);
    this.filter = ctx.createBiquadFilter(); this.filter.type = 'lowpass'; this.filter.frequency.value = 600; this.filter.connect(this.gain);
    this.oscs = [55, 82.4, 110.2].map((f, i) => {
      const o = ctx.createOscillator(); o.type = i === 2 ? 'sine' : 'triangle'; o.frequency.value = f;
      const g = ctx.createGain(); g.gain.value = i === 2 ? 0.05 : 0.12; o.connect(g).connect(this.filter); o.start();
      const lfo = ctx.createOscillator(); lfo.frequency.value = 0.03 + i * 0.017; const lg = ctx.createGain(); lg.gain.value = 1.8;
      lfo.connect(lg).connect(o.detune); lfo.start();
      return o;
    });
    const len = ctx.sampleRate * 3, nb = ctx.createBuffer(1, len, ctx.sampleRate), d = nb.getChannelData(0);
    let last = 0; for (let i = 0; i < len; i++) { last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02; d[i] = last * 3.5; }
    const noise = ctx.createBufferSource(); noise.buffer = nb; noise.loop = true;
    this.wind = ctx.createGain(); this.wind.gain.value = 0.25;
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 400; bp.Q.value = 0.6;
    noise.connect(bp).connect(this.wind).connect(this.filter); noise.start();
    const wl = ctx.createOscillator(); wl.frequency.value = 0.07; const wlg = ctx.createGain(); wlg.gain.value = 260; wl.connect(wlg).connect(bp.frequency); wl.start();
    this.gain.gain.setTargetAtTime(0.06, ctx.currentTime, 3);
  }

  setPlace(kind, time) {
    const t = this.ctx.currentTime;
    const base = { space: 41.2, moon: 46.2, underwater: 49, cave: 51.9, void: 55, city: 61.7, desert: 58.3, ocean: 55, clouds: 65.4 }[kind] || 55;
    this.oscs[0].frequency.setTargetAtTime(base, t, 4); this.oscs[1].frequency.setTargetAtTime(base * 1.4983, t, 4); this.oscs[2].frequency.setTargetAtTime(base * 2.0, t, 4);
    this.filter.frequency.setTargetAtTime(kind === 'underwater' ? 320 : kind === 'space' ? 450 : time === 'day' ? 1100 : 700, t, 3);
    this.wind.gain.setTargetAtTime({ space: 0.02, moon: 0.03, underwater: 0.12, cave: 0.1, ocean: 0.5, mountains: 0.45, snow: 0.4, desert: 0.35 }[kind] ?? 0.25, t, 3);
  }

  duck(seconds) {
    const t = this.ctx.currentTime;
    this.gain.gain.setTargetAtTime(0.03, t, 0.3);
    this.gain.gain.setTargetAtTime(0.06, t + seconds, 1.5);
  }
}
