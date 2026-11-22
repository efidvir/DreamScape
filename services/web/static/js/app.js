// DreamScape page: wires the dream renderer, the voice, the UI and the link to the server.
import { DreamRenderer } from './dream/renderer.js';
import { World } from './dream/world.js';
import { Net } from './net.js';
import { Sound } from './audio.js';
import { UI } from './ui.js';

const $ = (id) => document.getElementById(id);
const ui = new UI();
const sound = new Sound();

// Language: remembered choice, else the browser's.
const saved = localStorage.getItem('dreamscape.lang');
const initial = saved || ((navigator.language || '').toLowerCase().startsWith('he') ? 'he' : 'en');
$('lang').value = saved || 'auto';
ui.setLang(initial);
const chosenLang = () => $('lang').value;

// The dream itself.
const dream = new DreamRenderer($('dream'));
const world = new World(dream);
dream.world = world;
dream.start((t, dt) => world.update(t, dt));
sound.onLevel = (voice, mic) => { world.setVoice(voice); ui.setMicLevel(mic); };

// Session: one dream per browser until "new dream". The id works on any address, a LAN one
// included, and the page keeps whichever id the server answers with.
const SESSION_ID = /^[A-Za-z0-9_-]{6,64}$/;
const remember = (id) => { try { localStorage.setItem('dreamscape.session', id); } catch { /* storage off */ } };
let session = '';
try { session = localStorage.getItem('dreamscape.session') || ''; } catch { /* storage off */ }
if (!SESSION_ID.test(session)) {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  session = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  remember(session);
}
const net = new Net(session, chosenLang);

net.on('status', (m) => { ui.setStatus(m.services); ui.waking(m.services); })
  .on('waiting', (m) => { ui.waiting(m.for); if (m.services) ui.waking(m.services); })
  .on('link', (m) => {
    if (m.up) return;
    // An answer cut off by the lost link is over: the orb and the dream stop waiting for it.
    ui.notice('link'); ui.clearWaiting(); world.setThinking(false);
    if (!recording) ui.stage('idle');
  })
  .on('session', (m) => {
    if (m.session && m.session !== net.session) { net.session = m.session; remember(m.session); }
    if (m.scene) world.applyScene(m.scene, { initial: true });
    if (m.scene && m.scene.setting) sound.setPlace(m.scene.setting.kind, m.scene.setting.time);
    if ($('lang').value === 'auto' && m.lang) ui.setLang(m.lang);
    if (m.reply) ui.dreamer(m.reply);
  })
  .on('transcript', (m) => ui.player(m.text))
  .on('stage', (m) => { ui.stage(m.stage); world.setThinking(m.stage === 'dreaming' || m.stage === 'hearing'); if (m.stage === 'dreaming') ui.clearWaiting(); })
  .on('scene', (m) => {
    world.applyScene(m.scene, m);
    if (m.scene.setting) sound.setPlace(m.scene.setting.kind, m.scene.setting.time);
    if (m.scene.ended) setTimeout(() => { document.body.classList.add('finished'); $('fin').hidden = false; $('fin').classList.remove('gone'); }, 12000);
  })
  .on('reply', (m) => { ui.clearWaiting(); ui.dreamer(m.text); if ($('lang').value === 'auto' && m.lang) ui.setLang(m.lang); })
  .on('audio', (m) => sound.play(m.url))
  .on('speak', (m) => sound.speakFallback(m.text, m.lang, m.voice))
  .on('backdrop', (m) => world.setBackdrop(m.url))
  .on('sprite', (m) => world.setSprite(m.id, m.url))
  .on('notice', (m) => { ui.notice(m.key, m.detail); if (m.key === 'stumbled') ui.clearWaiting(); if (m.detail) console.warn('dreamscape:', m.key, m.detail); })
  .on('timings', (m) => ui.timings(m));

// Before entering, the card shows what is still loading (the live link starts on entering).
async function pollStatus() {
  if ($('veil').hidden || net.open) return;
  try { const s = await (await fetch('/api/status', { cache: 'no-store' })).json(); ui.setStatus(s); ui.waking(s); } catch { /* server still starting */ }
  setTimeout(pollStatus, 2500);
}
pollStatus();

// Entering the dream unlocks audio (browsers require a click).
$('enter').addEventListener('click', () => {
  sound.unlock();
  $('veil').classList.add('gone');
  setTimeout(() => { $('veil').hidden = true; }, 2500);
  net.connect();
});

// ------------------------------------------------------------------ talking
let recording = false;
async function beginTalk() {
  if (recording) return;
  try { await sound.startRecording(); recording = true; $('talk').classList.add('recording'); ui.stage('recording'); }
  catch (e) { console.warn(e); ui.notice('mic'); }
}
async function endTalk() {
  if (!recording) return;
  recording = false; $('talk').classList.remove('recording'); ui.stage('idle');
  const blob = await sound.stopRecording();
  if (!blob) { ui.notice('short'); return; }
  if (sound.peak < 0.03) { ui.notice('mic_silent'); return; }
  net.speakBlob(blob, blob.type);
}
const talk = $('talk');
talk.addEventListener('pointerdown', (e) => { e.preventDefault(); talk.setPointerCapture(e.pointerId); beginTalk(); });
talk.addEventListener('pointerup', endTalk);
talk.addEventListener('pointercancel', endTalk);
window.addEventListener('keydown', (e) => { if (e.code === 'Space' && !e.repeat && document.activeElement !== $('say')) { e.preventDefault(); beginTalk(); } });
window.addEventListener('keyup', (e) => { if (e.code === 'Space' && document.activeElement !== $('say')) { e.preventDefault(); endTalk(); } });

$('typebox').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = $('say').value.trim(); if (!text) return;
  sound.unlock(); sound.stopVoice();
  net.say(text); $('say').value = '';
});

// ------------------------------------------------------------------ tools
$('lang').addEventListener('change', () => {
  const v = $('lang').value; localStorage.setItem('dreamscape.lang', v);
  if (v !== 'auto') ui.setLang(v);
});
$('mute').addEventListener('click', () => { const m = !sound.muted; sound.setMuted(m); $('mute').setAttribute('aria-pressed', String(m)); });
function newDream() {
  sound.stopVoice(); world.ended = false; document.body.classList.remove('finished');
  $('fin').classList.add('gone'); setTimeout(() => { $('fin').hidden = true; }, 2400);
  net.reset();
}
$('new-dream').addEventListener('click', newDream);
$('fin-again').addEventListener('click', newDream);
window.addEventListener('keydown', (e) => { if (e.key === 'F2') $('timings').hidden = !$('timings').hidden; });

// Handy for testing from the console: dreamscape.say('you are a fairy').
window.dreamscape = { world, dream, net, say: (t) => net.say(t) };
