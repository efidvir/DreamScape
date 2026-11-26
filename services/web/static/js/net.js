// WebSocket link to the web service, with automatic reconnect.
export class Net {
  constructor(session, getLang) {
    this.session = session; this.getLang = getLang;
    this.handlers = {}; this.ws = null; this.retry = 0; this.open = false;
  }

  on(type, fn) { (this.handlers[type] ||= []).push(fn); return this; }
  // One handler's mistake must not keep the others from hearing the message.
  emit(type, msg) {
    for (const fn of this.handlers[type] || []) {
      try { fn(msg); } catch (e) { console.error(`dreamscape: the "${type}" message failed`, e); }
    }
  }

  connect() {
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
    const ws = new WebSocket(url); ws.binaryType = 'arraybuffer'; this.ws = ws;
    ws.onopen = () => {
      this.open = true; this.retry = 0; this.emit('link', { up: true });
      this.send({ type: 'hello', session: this.session, lang: this.lang() });
    };
    ws.onmessage = (ev) => { try { const m = JSON.parse(ev.data); this.emit(m.type, m); } catch { /* ignore */ } };
    ws.onclose = () => {
      this.open = false; this.emit('link', { up: false });
      const wait = Math.min(8000, 600 * 2 ** this.retry++);
      setTimeout(() => this.connect(), wait);
    };
  }

  lang() { const l = this.getLang(); return l === 'auto' ? (document.documentElement.lang || 'he') : l; }
  send(obj) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj)); }
  say(text) { const l = this.getLang(); this.send({ type: 'text', text, lang: l === 'auto' ? null : l }); }
  speakBlob(blob, mime) {
    const l = this.getLang();
    this.send({ type: 'audio', mime, lang: l === 'auto' ? null : l });
    blob.arrayBuffer().then((buf) => { if (this.ws && this.ws.readyState === 1) this.ws.send(buf); });
  }
  reset() { this.send({ type: 'reset', lang: this.lang() }); }
}
