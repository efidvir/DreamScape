// The World keeps the 3D dream in step with the director's scene: it spawns things,
// dissolves them in and out, moves them, blends one place into the next, and
// carries the sky, fog, light, mist and weather.
import * as THREE from 'three';
import { BUILDERS, GROUND_KINDS, SKY_KINDS, wisp, guessKind } from './things.js';
import { buildSetting, palette } from './settings.js';
import { pointify, dreamU, imageCloud } from './points.js';
import { rng, color, collectFade, applyFade, disposeDeep, Particles, mistTex, softDot, snoise, glowSprite, sparkleTex, dreamClock } from './kit.js';

const FADE_IN = 3.6, FADE_OUT = 2.6;
const _tmpC = new THREE.Color();
const ease = (x) => x * x * (3 - 2 * x);

function layout(where, kind, i, n, r, heightAt, size = 'medium') {
  const ground = GROUND_KINDS.has(kind);
  let x = 0, y = 1, z = 0;
  if (SKY_KINDS.has(kind) && kind !== 'cloud' && n === 1 && where !== 'near' && where !== 'center') where = 'sky';
  switch (where) {
    case 'center': x = r.sign() * (1.5 + r() * 1.2); z = -0.5 + r() * 1.3; y = 0.9 + r() * 0.8; break;   // beside the dreamer, in view
    case 'near': x = r.sign() * (2.1 + r() * 2.0); z = 0.6 + r() * 2.0; y = 0.7 + r() * 0.8; break;
    case 'far': x = (r() - 0.5) * 20; z = -9 - r() * 9; y = 1.2 + r() * 1.5; break;
    case 'left': x = -3.8 - r() * 3.2; z = -2.5 + r() * 4.5; y = 0.9 + r() * 1.2; break;
    case 'right': x = 3.8 + r() * 3.2; z = -2.5 + r() * 4.5; y = 0.9 + r() * 1.2; break;
    case 'above': x = (r() - 0.5) * 4; z = -1 + (r() - 0.5) * 3; y = 3.4 + r() * 1.6; break;
    case 'sky':
      if (n === 1) { x = (r() - 0.5) * 26; y = 12 + r() * 5; z = -42 - r() * 10; }
      else { x = (r() - 0.5) * 70; y = 8 + r() * 20; z = -25 - r() * 30; }
      break;
    case 'ground': { const a = r() * 6.28, d = 1.5 + Math.sqrt(r()) * 6.5; x = Math.cos(a) * d; z = Math.sin(a) * d - 1; break; }
    default: { // around
      const big = ground && n > 3;            // groves keep their distance from the camera
      // Small groups stay in the half the camera looks at; big ones surround the dreamer.
      const a = n <= 3 ? Math.PI * (1.18 + 0.64 * ((i + r()) / n)) : ((i + r() * 0.7) / Math.max(1, n)) * Math.PI * 2;
      const d = n <= 3 ? 3.2 + r() * 5 : (big ? 5.5 : 3.6) + r() * (big ? 11 : 9.5);
      x = Math.cos(a) * d; z = Math.sin(a) * d;
      if (z > 0 && Math.abs(x) < 3.2) x += Math.sign(x || 1) * 3.2;
      y = 0.8 + r() * 2.2;
    }
  }
  // Big structures and big things keep their distance so they never swallow the dreamer.
  const minD = Math.max(MIN_DIST[kind] || 0, SIZE_DIST[size] || 0);
  if (minD && where !== 'sky') {
    const d = Math.hypot(x, z);
    if (d < minD) { const k = minD / Math.max(0.001, d); x *= k; z *= k; if (z > 0) z = -z; }
  }
  if (ground) y = heightAt(x, z);
  return new THREE.Vector3(x, y, z);
}

const SIZE_DIST = { large: 4.5, huge: 9 };
const MIN_DIST = { castle: 16, house: 8, portal: 6, dragon: 6, horse: 3.5, boat: 3.5, spaceship: 3.2, water: 5, volcano: 15 };

// The dreamer is drawn densest of all: a white or pale one is drawn a shade greyer, or the
// bloom turns it into a shapeless glow.
const PROFILE_KINDS = new Set(['cat', 'dog', 'wolf', 'horse', 'bird', 'fish', 'dragon']);

function notTooPale(value) {
  const c = color(value, '#d8ecff');
  const lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  return lum > 0.6 ? '#' + c.multiplyScalar(0.6 / lum).getHexString() : value;
}

class Entity {
  constructor(world, spec, isSelf = false) {
    if (isSelf && spec.kind !== 'wisp' && spec.color) spec = { ...spec, color: notTooPale(spec.color) };
    this.world = world; this.spec = { ...spec }; this.id = spec.id; this.isSelf = isSelf;
    const r = rng(spec.id + ':' + spec.kind);
    const n = spec.count || 1;
    const heightAt = world.heightAt;
    // Something filed as "other" is still drawn as itself when its label names a shape we have.
    const shape = spec.kind === 'other' ? (guessKind(spec.label) || 'other') : spec.kind;
    const place = isSelf
      ? () => new THREE.Vector3(0, GROUND_KINDS.has(shape) ? heightAt(0, 0) : 1.25, 0)
      : (i) => layout(spec.where, shape, i, n, r, heightAt, spec.size);
    const builder = shape === 'wisp' ? wisp : (BUILDERS[shape] || BUILDERS.other);
    this.b = builder(spec, r, place, { setting: world.settingKind });
    this.b.members ||= []; this.b.world ||= []; this.b.tintables ||= [];
    if (isSelf && spec.kind !== 'wisp' && this.b.members.length) {
      const m0 = this.b.members[0];
      // Animals show their side to the viewer, so a horse or a bird reads by its shape;
      // people, ghosts and fairies face the viewer.
      m0.rotation.y = PROFILE_KINDS.has(spec.kind) ? 1.45 : 0;
      m0.userData.yaw0 = m0.rotation.y;
      // A small being (a bird, a fish) would be a speck: the dreamer's body is drawn at least
      // 1.2 m big (glows and halos do not count).
      this.b.root.updateMatrixWorld(true);
      const body = new THREE.Box3(), part = new THREE.Box3();
      m0.traverse((o) => {
        if (!o.isMesh || !o.geometry) return;
        if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
        body.union(part.copy(o.geometry.boundingBox).applyMatrix4(o.matrixWorld));
      });
      const size = body.isEmpty() ? null : body.getSize(new THREE.Vector3());
      const big = size ? Math.max(size.x, size.y, size.z) : 0;
      if (big > 0.01 && big < 1.2) m0.scale.multiplyScalar(1.2 / big);
      else if (big > 3.2) m0.scale.multiplyScalar(3.2 / big);      // a whale still fits the view
    }
    this.b.members.forEach((m, k) => { m.userData.base = m.position.clone(); m.userData.phase = r() * 6.283 + k; m.userData.scale0 = m.scale.x; });
    this.root = this.b.root;
    // Everything solid is redrawn as a soft particle cloud.
    // The dreamer is drawn a little denser than other things, but not so dense that it
    // blooms into a white blot.
    const dense = isSelf ? { density: 1000, max: 8000 } : { density: 700, max: (spec.count || 1) > 3 ? 30000 : 14000 };
    pointify(this.root, dense);
    for (const w of this.b.world) pointify(w, dense);
    world.scene.add(this.root);
    for (const w of this.b.world) world.scene.add(w);
    if (isSelf && spec.kind !== 'wisp') {
      // A white or pale dreamer already shines: a full halo on top would bloom it into a blot.
      const c0 = color(spec.color, '#d8ecff');
      this.auraDim = 1 - 0.6 * Math.min(1, 0.2126 * c0.r + 0.7152 * c0.g + 0.0722 * c0.b);
      this.aura = glowSprite(spec.color || '#d8ecff', 1.7, 0.16);
      this.auraLight = new THREE.PointLight(c0, 1.0, 6, 1.5);
      const m0 = this.b.members[0] || this.root;
      m0.add(this.aura, this.auraLight);
    }
    this.fades = [collectFade(this.root), ...this.b.world.map(collectFade)];
    this.fade = 0; this.target = 1; this.dying = false;
    this.motion = spec.motion || 'still';
    this.env = { moving: false, voice: 0, thinking: false };
    this.pictures = []; this.pictureFade = 0;
    this.flickerSeed = r() * 100;
    this.anchor = this.b.members[0] ? this.b.members[0].userData.base.clone() : place(0);
    for (const f of this.fades) applyFade(f, 0);
  }

  retint(c) {
    if (!c) return;
    if (this.isSelf) c = notTooPale(c);
    this.tintTo = color(c, '#ffffff'); this.tintT = 0;
    this.tintFrom = this.fades.flatMap((f) => f.points).map((m) => ({ u: m.userData.dreamPoints, c: m.userData.dreamPoints.uTint.value.clone(), a: m.userData.dreamPoints.uTintAmt.value }));
  }

  get position() { return (this.b.members[0] || this.root).getWorldPosition(new THREE.Vector3()); }

  update(t, dt) {
    const speed = this.target > this.fade ? dt / FADE_IN : dt / FADE_OUT;
    this.fade += Math.sign(this.target - this.fade) * Math.min(Math.abs(this.target - this.fade), speed);
    // Things never sit perfectly still in a dream: they waver between clear and torn.
    const flicker = 0.55 + 0.45 * (0.5 + 0.5 * snoise(t * 0.23, this.flickerSeed)) * (0.6 + 0.4 * this.world.clarityNow);
    const e = ease(Math.max(0, Math.min(1, this.fade)));
    for (const f of this.fades) applyFade(f, e, flicker);
    if (this.pictureFade > 0) {
      this.pictureFade = Math.min(1, this.pictureFade + dt / 3.5);
      const k = ease(this.pictureFade);
      for (const f of this.fades) for (const m of f.points) m.userData.dreamPoints.uOpacity.value *= 1 - 0.82 * k;
      const cam = this.world.dream.camera, q = new THREE.Quaternion();
      for (const p of this.pictures) {
        p.u.uForm.value = k * e; p.u.uOpacity.value = Math.min(1, k * e * 1.4) * (0.8 + 0.2 * flicker);
        p.u.uWander.value = 0.7 * (0.6 + 1.4 * (1 - flicker));
        p.holder.parent.getWorldQuaternion(q); p.holder.quaternion.copy(q.invert().multiply(cam.quaternion));
      }
    }
    this.root.visible = this.fade > 0.001;
    if (this.tintTo) {
      this.tintT = Math.min(1, this.tintT + dt / 2.5);
      const k = ease(this.tintT);
      for (const f of this.tintFrom) { f.u.uTint.value.copy(f.c).lerp(this.tintTo, k); f.u.uTintAmt.value = f.a + (0.75 - f.a) * k; }
      if (this.tintT >= 1) this.tintTo = null;
    }
    this.move(t, dt, e);
    this.env.voice = this.isSelf ? this.world.voice : 0;
    this.env.thinking = this.world.thinking;
    if (this.b.update) this.b.update(t, dt, this.env);
    if (this.aura) {
      const v = this.world.voice;
      this.aura.scale.setScalar(2.0 + 0.4 * Math.sin(t * 1.4) + 2.2 * v);
      this.aura.material.opacity = (0.28 + 0.3 * v) * e * this.auraDim;
      this.auraLight.intensity = (1.2 + 3 * v) * e * this.auraDim;
    }
  }

  move(t, dt, e) {
    const members = this.b.members; if (!members.length) return;
    this.env.moving = false;
    for (const m of members) {
      const b = m.userData.base, ph = m.userData.phase, prev = m.position.clone();
      const rise = (1 - e) * -0.35;             // things float up a little as they appear
      switch (this.isSelf ? 'self' : this.motion) {
        case 'self': {
          const ground = GROUND_KINDS.has(this.spec.kind);
          const amp = this.spec.kind === 'fairy' || this.spec.kind === 'butterfly' ? 0.55 : 0.18;
          m.position.set(b.x + snoise(t * 0.25, ph) * amp, ground ? b.y : b.y + Math.sin(t * 0.8) * 0.12 + snoise(t * 0.3, ph + 2) * amp * 0.4, b.z + snoise(t * 0.21, ph + 5) * amp * 0.6);
          if (!ground) m.rotation.y = (m.userData.yaw0 || 0) + Math.sin(t * 0.3) * 0.4;
          break;
        }
        case 'float': m.position.set(b.x, b.y + Math.sin(t * 0.9 + ph) * 0.2 + rise, b.z); m.rotation.y += dt * 0.2; break;
        case 'drift': m.position.set(b.x + snoise(t * 0.06, ph) * 2, b.y + snoise(t * 0.05, ph + 3) * 0.5 + rise, b.z + snoise(t * 0.055, ph + 7) * 2); break;
        case 'orbit': {
          const R = Math.max(1.5, Math.hypot(b.x, b.z)), a = Math.atan2(b.z, b.x) + t * 0.22;
          m.position.set(Math.cos(a) * R, b.y + Math.sin(t + ph) * 0.2, Math.sin(a) * R); break;
        }
        case 'buzz': m.position.set(b.x + snoise(t * 0.55, ph) * 1.4 + Math.sin(t * 7 + ph) * 0.04, b.y + snoise(t * 0.7, ph + 1) * 0.6 + rise, b.z + snoise(t * 0.5, ph + 2) * 1.4); break;
        case 'fly': {
          const R = Math.max(6, Math.hypot(b.x, b.z)), a = t * 0.16 + ph;
          m.position.set(Math.cos(a) * R, Math.max(2, b.y) + Math.sin(t * 0.33 + ph) * 1.5, Math.sin(a) * R * 0.75 - 4); break;
        }
        case 'swim': m.position.set(b.x + Math.sin(t * 0.35 + ph) * 2.8, b.y + Math.sin(t * 0.9 + ph) * 0.15, b.z + Math.cos(t * 0.27 + ph) * 1.4); break;
        case 'walk': {
          const R = Math.max(2.5, Math.hypot(b.x, b.z)), a = Math.atan2(b.z, b.x) + t * (0.35 / R) * (ph > 3.1 ? 1 : -1);
          const x = Math.cos(a) * R, z = Math.sin(a) * R;
          m.position.set(x, GROUND_KINDS.has(this.spec.kind) ? this.world.heightAt(x, z) : b.y, z);
          this.env.moving = true; break;
        }
        case 'spin': m.position.set(b.x, b.y + Math.sin(t * 0.8 + ph) * 0.1 + rise, b.z); m.rotation.y += dt * 0.9; break;
        case 'pulse': m.position.set(b.x, b.y + rise, b.z); m.scale.setScalar(m.userData.scale0 * (1 + 0.07 * Math.sin(t * 2.2 + ph))); break;
        default: m.position.set(b.x, b.y + Math.sin(t * 0.5 + ph) * 0.02 + rise, b.z);
      }
      // Face the way we are going.
      if (['buzz', 'fly', 'swim', 'walk', 'orbit', 'drift'].includes(this.motion) && !this.isSelf) {
        const dx = m.position.x - prev.x, dz = m.position.z - prev.z;
        if (dx * dx + dz * dz > 1e-7) {
          const yaw = Math.atan2(dx, dz); let d = yaw - m.rotation.y;
          d = Math.atan2(Math.sin(d), Math.cos(d)); m.rotation.y += d * Math.min(1, dt * 4);
          if (this.motion === 'fly') m.rotation.z = THREE.MathUtils.clamp(-d * 2, -0.5, 0.5);
        }
        if (this.motion !== 'walk') this.env.moving = true;
      }
    }
  }

  dispose() {
    this.world.scene.remove(this.root);
    for (const w of this.b.world) this.world.scene.remove(w);
    disposeDeep(this.root); for (const w of this.b.world) disposeDeep(w);
  }
}

// ------------------------------------------------------------------ the world
export class World {
  constructor(dream) {
    this.dream = dream; this.scene = dream.scene;
    this.entities = new Map(); this.dying = [];
    this.self = null; this.selfKey = '';
    this.voice = 0; this.thinking = false; this.clarityNow = 0.15;
    this.settingKind = 'void'; this.settingKey = '';
    this.setting = null; this.oldSettings = [];
    this.heightAt = () => 0;
    this.effects = [];
    this.pal = palette('void', 'night', 'mist'); this.palTo = this.pal;
    this._buildAtmosphere();
    this.applyScene({ self: { form: 'wisp', color: '#d8ecff' }, setting: { kind: 'void', time: 'night', weather: 'mist' }, things: [], clarity: 0.12 }, { initial: true });
  }

  // ---------------------------------------------------------------- atmosphere
  _buildAtmosphere() {
    const s = this.scene;
    this.skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { uTop: { value: this.pal.skyTop.clone() }, uHorizon: { value: this.pal.skyHorizon.clone() }, uTime: { value: 0 } },
      vertexShader: 'varying vec3 vD; void main(){ vD = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: `uniform vec3 uTop, uHorizon; uniform float uTime; varying vec3 vD;
        void main(){ float h = vD.y; vec3 c = mix(uHorizon, uTop, smoothstep(-0.05, 0.55, h));
          c = mix(c, uHorizon * 0.6, smoothstep(0.0, -0.4, h));
          float swirl = sin(vD.x * 6.0 + uTime * 0.05) * sin(vD.z * 5.0 - uTime * 0.04) * 0.025;
          gl_FragColor = vec4(c + swirl * uHorizon, 1.0); }`,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(460, 32, 20), this.skyMat); this.sky.renderOrder = -10; s.add(this.sky);

    this.hemi = new THREE.HemisphereLight(this.pal.hemiSky, this.pal.hemiGround, this.pal.hemiI); s.add(this.hemi);
    this.key = new THREE.DirectionalLight(this.pal.key, this.pal.keyI); this.key.position.set(-30, 40, 20); s.add(this.key);
    s.fog = new THREE.FogExp2(this.pal.fog.clone(), this.pal.fogDensity);

    this.stars = new Particles(1600, { size: 3.4, sizeJitter: 1.2, colorA: '#ffffff', colorB: '#bcd4ff', twinkle: 0.7, seed: 99 });
    const r = rng(7);
    for (let i = 0; i < 1600; i++) { const a = r() * 6.283, b = Math.asin(0.04 + 0.96 * r()), d = 330; this.stars.setBase(i, Math.cos(a) * Math.cos(b) * d, Math.sin(b) * d, Math.sin(a) * Math.cos(b) * d); }
    this.stars.material.fog = false; s.add(this.stars);

    // Mist: big soft puffs that drift through the scene and make everything dream-like.
    this.mist = [];
    for (let i = 0; i < 34; i++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex(i % 4), color: '#ffffff', transparent: true, opacity: 0, depthWrite: false }));
      sp.position.set((r() - 0.5) * 44, r() * 4.5 - 0.3, (r() - 0.5) * 44 - 4);
      sp.scale.setScalar(7 + r() * 9); sp.material.rotation = r() * 6.28;
      sp.userData = { vx: (r() - 0.5) * 0.35, vz: (r() - 0.5) * 0.25, spin: (r() - 0.5) * 0.02, ph: r() * 6.28 };
      s.add(sp); this.mist.push(sp);
    }
    this.motes = new Particles(260, { size: 0.05, sizeJitter: 0.8, colorA: '#ffffff', colorB: '#cfe3ff', twinkle: 0.8, opacity: 0.7, seed: 17 });
    for (let i = 0; i < 260; i++) this.motes.setBase(i, (r() - 0.5) * 18, r() * 5, (r() - 0.5) * 18);
    s.add(this.motes);

    // Weather: rain streaks and snow flakes share one particle system each.
    const streak = document.createElement('canvas'); streak.width = 8; streak.height = 64;
    const sg = streak.getContext('2d'); const lg = sg.createLinearGradient(0, 0, 0, 64);
    lg.addColorStop(0, 'rgba(255,255,255,0)'); lg.addColorStop(1, 'rgba(255,255,255,0.9)'); sg.fillStyle = lg; sg.fillRect(3, 0, 2, 64);
    const streakTex = new THREE.CanvasTexture(streak);
    this.rain = new Particles(900, { size: 0.9, colorA: '#bfd6ff', twinkle: 0, opacity: 0, map: streakTex, seed: 3, additive: true });
    this.snow = new Particles(700, { size: 0.16, colorA: '#ffffff', twinkle: 0.2, opacity: 0, seed: 4, additive: false });
    for (const p of [this.rain, this.snow]) { for (let i = 0; i < p.count; i++) p.setBase(i, (r() - 0.5) * 40, r() * 16, (r() - 0.5) * 40); s.add(p); }
    this.weatherAmt = { rain: 0, snow: 0 };
    this.nextLightning = 0;

    // Backdrop: a generated picture far behind the dream, when the imagery service is on.
    this.backdrops = [];
  }

  _setPalette(kind, time, weather) {
    this.palFrom = this._palNow(); this.palTo = palette(kind, time, weather); this.palT = 0;
  }

  _palNow() {
    return { skyTop: this.skyMat.uniforms.uTop.value.clone(), skyHorizon: this.skyMat.uniforms.uHorizon.value.clone(), fog: this.scene.fog.color.clone(),
      hemiSky: this.hemi.color.clone(), hemiGround: this.hemi.groundColor.clone(), key: this.key.color.clone(),
      hemiI: this.hemi.intensity, keyI: this.key.intensity, stars: this.stars.material.uniforms.uOpacity.value,
      fogDensity: this.baseDensity ?? this.pal.fogDensity, mist: this.mistAmt ?? this.pal.mist, weather: this.palTo?.weather };
  }

  _blendPalette(dt) {
    if (this.palT === undefined || this.palT >= 1) return;
    this.palT = Math.min(1, this.palT + dt / 4.5);
    const k = ease(this.palT), a = this.palFrom, b = this.palTo;
    this.skyMat.uniforms.uTop.value.copy(a.skyTop).lerp(b.skyTop, k);
    this.skyMat.uniforms.uHorizon.value.copy(a.skyHorizon).lerp(b.skyHorizon, k);
    this.scene.fog.color.copy(a.fog).lerp(b.fog, k);
    this.hemi.color.copy(a.hemiSky).lerp(b.hemiSky, k); this.hemi.groundColor.copy(a.hemiGround).lerp(b.hemiGround, k);
    this.key.color.copy(a.key).lerp(b.key, k);
    this.hemi.intensity = a.hemiI + (b.hemiI - a.hemiI) * k; this.key.intensity = a.keyI + (b.keyI - a.keyI) * k;
    this.stars.material.uniforms.uOpacity.value = a.stars + (b.stars - a.stars) * k;
    this.baseDensity = a.fogDensity + (b.fogDensity - a.fogDensity) * k;
    this.mistAmt = a.mist + (b.mist - a.mist) * k;
  }

  // ---------------------------------------------------------------- reconcile
  applyScene(scene, change = {}) {
    if (!scene) return;
    const st = scene.setting || { kind: 'void', time: 'night', weather: 'mist' };
    const key = `${st.kind}|${st.time}`;
    if (key !== this.settingKey) {
      const travelling = this.settingKind !== st.kind && !change.initial;
      if (this.setting) { this.setting.target = 0; this.oldSettings.push(this.setting); }
      this.settingKind = st.kind; this.settingKey = key;
      const built = buildSetting(st.kind, st.time, 'dream');
      pointify(built.group, { density: 80, min: 40, max: 36000, maxSize: 0.42, wander: 0.5 });
      this.heightAt = built.heightAt;
      this.scene.add(built.group);
      this.setting = { ...built, fades: [collectFade(built.group)], fade: 0, target: 1, delay: travelling ? 1.6 : 0 };
      for (const f of this.setting.fades) applyFade(f, 0);
      if (travelling) this.dream.warp();
    }
    this._setPalette(st.kind, st.time, st.weather);
    this.weather = st.weather;

    // the dreamer itself
    const me = scene.self || { form: 'wisp' };
    const selfKey = `${me.form}|${me.color}`;
    if (selfKey !== this.selfKey) {
      const formChanged = this.self && this.self.spec.kind !== me.form;
      if (this.self && !formChanged) this.self.retint(me.color);
      else {
        if (this.self) { this.self.target = 0; this.dying.push(this.self); }
        this.self = new Entity(this, { id: 'self', kind: me.form, color: me.color, label: me.label, size: 'medium', count: 1, where: 'center', motion: 'self' }, true);
        if (formChanged || (!change.initial && me.form !== 'wisp')) this.burst(new THREE.Vector3(0, 1.25, 0), me.color || '#ffffff', 160);
      }
      this.selfKey = selfKey;
    }

    // everything else
    const seen = new Set();
    for (const t of scene.things || []) {
      seen.add(t.id);
      const cur = this.entities.get(t.id);
      const rebuild = !cur || cur.spec.kind !== t.kind || cur.spec.count !== t.count || cur.spec.where !== t.where || cur.spec.size !== t.size || cur.spec.label !== t.label;
      if (rebuild) {
        if (cur) { cur.target = 0; this.dying.push(cur); }
        const ent = new Entity(this, t);
        this.entities.set(t.id, ent);
        if (!change.initial && !cur) {
          this.materialize(ent);
          if (t.kind === 'volcano' || guessKind(t.label) === 'volcano') {   // it erupts as it appears
            this.burst(ent.anchor.clone().add(new THREE.Vector3(0, 3, 0)), '#ff7a2a', 320);
            this.dream.flash(new THREE.Color('#ff6a2a'), 0.6);
          }
        }
      } else {
        if (cur.spec.color !== t.color) { cur.retint(t.color); cur.spec.color = t.color; }
        if (cur.spec.motion !== t.motion) { cur.motion = t.motion; cur.spec.motion = t.motion; }
      }
    }
    for (const [id, ent] of this.entities) {
      if (!seen.has(id)) { ent.target = 0; this.dying.push(ent); this.entities.delete(id); }
    }

    this.clarityTarget = scene.clarity ?? 0.2;
    if (change.initial) this.clarityNow = this.clarityTarget;

    // attention: look at what just appeared
    const firstNew = (change.added || []).map((id) => this.entities.get(id)).find(Boolean);
    this.dream.attend(firstNew ? firstNew.anchor : new THREE.Vector3(0, 1.25, 0), !change.initial);
    if (!change.initial) this.dream.pulseClarity();
    this.runEffect(change.effect, firstNew);
    if (scene.ended && !this.ended) { this.ended = true; this.runEffect('ascend'); }
  }

  runEffect(effect, near) {
    const at = near ? near.anchor.clone() : new THREE.Vector3(0, 1.25, 0);
    switch (effect) {
      case 'sparkle': this.burst(at, '#ffffff', 220); break;
      case 'flash': this.dream.flash(new THREE.Color('#ffffff'), 0.85); break;
      case 'ripple': this.dream.ripple(); break;
      case 'warp': this.dream.warp(); break;
      case 'darken': this.dream.darken(); break;
      case 'bloom': this.dream.bloomBurst(); break;
      case 'ascend': this.ascend(); break;
      default: break;
    }
  }

  // ---------------------------------------------------------------- effects
  burst(at, c, n = 160) {
    const p = new Particles(n, { size: 0.16, colorA: '#ffffff', colorB: c, twinkle: 1, map: sparkleTex(), seed: Math.floor(Math.random() * 1e6) });
    const vel = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      p.setBase(i, at.x, at.y, at.z);
      const a = Math.random() * 6.283, b = Math.acos(2 * Math.random() - 1), sp = 0.8 + Math.random() * 2.6;
      vel.set([Math.sin(b) * Math.cos(a) * sp, Math.cos(b) * sp, Math.sin(b) * Math.sin(a) * sp], i * 3);
    }
    this.scene.add(p);
    this.effects.push({ life: 0, dur: 2.6, update: (e, t, dt) => {
      const pos = p.geometry.attributes.position.array;
      for (let i = 0; i < n * 3; i++) pos[i] += vel[i] * dt * (1 - e);
      p.material.uniforms.uOpacity.value = 1 - e; p.touch(); p.tick(t);
    }, done: () => { this.scene.remove(p); p.geometry.dispose(); p.material.dispose(); } });
  }

  materialize(ent) {
    const at = ent.anchor.clone(), n = 90;
    const p = new Particles(n, { size: 0.12, colorA: '#ffffff', colorB: '#bfe3ff', twinkle: 0.9, map: sparkleTex(), seed: 33 });
    const from = [];
    for (let i = 0; i < n; i++) { const a = Math.random() * 6.283, d = 2 + Math.random() * 3; from.push(new THREE.Vector3(at.x + Math.cos(a) * d, at.y + (Math.random() - 0.3) * 3, at.z + Math.sin(a) * d)); }
    this.scene.add(p);
    this.effects.push({ life: 0, dur: 2.2, update: (e, t) => {
      const k = ease(e);
      for (let i = 0; i < n; i++) { const f = from[i]; p.setBase(i, f.x + (at.x - f.x) * k, f.y + (at.y - f.y) * k, f.z + (at.z - f.z) * k); }
      p.material.uniforms.uOpacity.value = Math.sin(e * Math.PI); p.touch(); p.tick(t);
    }, done: () => { this.scene.remove(p); p.geometry.dispose(); p.material.dispose(); } });
  }

  ascend() {
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 3.2, 40, 32, 1, true),
      new THREE.MeshBasicMaterial({ color: '#ffe7a3', transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    beam.position.set(0, 20, 0); this.scene.add(beam);
    const halo = glowSprite('#ffe7a3', 1, 0); halo.position.set(0, 1.25, 0); this.scene.add(halo);
    this.dream.bloomBurst(6); this.dream.flash(new THREE.Color('#fff2c8'), 0.6);
    this.burst(new THREE.Vector3(0, 1.25, 0), '#ffd56a', 320);
    this.effects.push({ life: 0, dur: 9, update: (e, t) => {
      beam.material.opacity = Math.sin(Math.min(1, e * 1.4) * Math.PI) * 0.35;
      beam.rotation.y = t * 0.2;
      halo.scale.setScalar(2 + e * 10); halo.material.opacity = Math.sin(e * Math.PI) * 0.6;
      if (this.self) { const m = this.self.b.members[0] || this.self.root; m.scale.setScalar((m.userData.scale0 || 1) * (1 + 0.8 * ease(Math.min(1, e * 1.5)))); }
    }, done: () => { this.scene.remove(beam, halo); beam.geometry.dispose(); beam.material.dispose(); halo.material.dispose(); } });
  }

  setBackdrop(url) {
    new THREE.TextureLoader().load(url, (tex) => {
      const img = tex.image, c = document.createElement('canvas'); c.width = 512; c.height = 512;
      const g = c.getContext('2d'); g.filter = 'blur(6px)'; g.drawImage(img, 0, 0, 512, 512); g.filter = 'none';
      g.globalCompositeOperation = 'destination-in';
      const rg = g.createRadialGradient(256, 256, 120, 256, 256, 256); rg.addColorStop(0, '#000'); rg.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = rg; g.fillRect(0, 0, 512, 512);
      const t2 = new THREE.CanvasTexture(c); t2.colorSpace = THREE.SRGBColorSpace; tex.dispose();
      const m = new THREE.Mesh(new THREE.PlaneGeometry(420, 260), new THREE.MeshBasicMaterial({ map: t2, transparent: true, opacity: 0, depthWrite: false, fog: false }));
      m.position.set(0, 60, -300); m.renderOrder = -9; this.scene.add(m);
      for (const b of this.backdrops) b.target = 0;
      this.backdrops.push({ mesh: m, fade: 0, target: 0.42 });
    });
  }

  // A generated picture of a thing arrives: it becomes the thing's particle cloud, and
  // the geometric shape fades back into a faint aura.
  setSprite(id, url) {
    const ent = id === 'self' ? this.self : this.entities.get(id);
    if (!ent || !ent.b.members.length) return;
    const img = new Image();
    img.onload = () => {
      if (ent !== (id === 'self' ? this.self : this.entities.get(id))) return;   // replaced meanwhile
      const box = new THREE.Box3(), size = new THREE.Vector3(), center = new THREE.Vector3();
      for (const m of ent.b.members) {
        box.setFromObject(m); box.getSize(size); box.getCenter(center); m.worldToLocal(center);
        const S = THREE.MathUtils.clamp(Math.max(size.x, size.y, size.z) * 1.3, 0.8, 14);
        const pts = imageCloud(img, S); if (!pts) continue;
        const holder = new THREE.Group(); holder.position.copy(center); holder.add(pts); m.add(holder);
        ent.pictures.push({ holder, u: pts.material.userData.dreamPoints });
      }
      ent.pictureFade = 0.0001;
    };
    img.src = url;
  }

  setVoice(v) { this.voice = v; }
  setThinking(on) { this.thinking = on; }

  // ---------------------------------------------------------------- frame
  update(t, dt) {
    dreamClock.value = t;
    this._blendPalette(dt);
    this.skyMat.uniforms.uTime.value = t;
    this.clarityNow += ((this.clarityTarget ?? 0.2) - this.clarityNow) * Math.min(1, dt * 0.3);
    const haze = this.dream.haze;   // 0 = sharp, 1 = deep dream
    // Particles carry their own lighting and fog.
    const L = dreamU.uLight.value;
    L.copy(this.hemi.color).multiplyScalar(this.hemi.intensity * 0.6).add(_tmpC.copy(this.key.color).multiplyScalar(this.key.intensity * 0.42));
    L.setRGB(Math.min(L.r, 1.15), Math.min(L.g, 1.15), Math.min(L.b, 1.15));
    dreamU.uFogColor.value.copy(this.scene.fog.color);
    this.scene.fog.density = (this.baseDensity ?? this.pal.fogDensity) * (0.65 + 0.55 * haze) * this.dream.fogBoost;
    dreamU.uFogDensity.value = this.scene.fog.density;

    // settings
    if (this.setting) {
      const s = this.setting;
      if (s.delay > 0) s.delay -= dt; else s.fade = Math.min(1, s.fade + dt / 3.5);
      for (const f of s.fades) applyFade(f, ease(s.fade));
      if (s.update) s.update(t, dt);
    }
    this.oldSettings = this.oldSettings.filter((s) => {
      s.fade = Math.max(0, s.fade - dt / 2.2);
      for (const f of s.fades) applyFade(f, ease(s.fade));
      if (s.update) s.update(t, dt);
      if (s.fade <= 0) { this.scene.remove(s.group); disposeDeep(s.group); return false; }
      return true;
    });

    // entities
    if (this.self) this.self.update(t, dt);
    for (const e of this.entities.values()) e.update(t, dt);
    this.dying = this.dying.filter((e) => { e.update(t, dt); if (e.fade <= 0.001) { e.dispose(); return false; } return true; });

    // mist & motes
    const voidish = this.settingKind === 'void';
    const fogC = this.scene.fog.color, mistColor = fogC.clone().lerp(new THREE.Color('#ffffff'), voidish ? 0.1 : 0.14);
    const fogLum = 0.2126 * fogC.r + 0.7152 * fogC.g + 0.0722 * fogC.b;
    const mistA = (this.mistAmt ?? this.pal.mist) * (0.25 + 0.6 * haze) * (this.thinking ? 1.25 : 1) * (voidish ? 0.6 : 1) * (1 - 0.6 * Math.min(1, fogLum * 1.6));
    for (const m of this.mist) {
      const u = m.userData;
      m.position.x += u.vx * dt * (this.thinking ? 2.5 : 1); m.position.z += u.vz * dt;
      if (m.position.x > 24) m.position.x = -24; if (m.position.x < -24) m.position.x = 24;
      if (m.position.z > 18) m.position.z = -26; if (m.position.z < -26) m.position.z = 18;
      m.material.rotation += u.spin * dt;
      m.material.color.copy(mistColor);
      // Puffs drifting between the viewer and the dreamer thin out, so the dreamer stays seen.
      const front = THREE.MathUtils.smoothstep(m.position.z, -1.5, 2.5);
      m.material.opacity = mistA * (0.55 + 0.45 * Math.sin(t * 0.17 + u.ph)) * (1 - 0.85 * front);
    }
    const mp = this.motes.geometry.attributes.position.array;
    for (let i = 0; i < this.motes.count; i++) {
      mp[i * 3] = this.motes.base[i * 3] + Math.sin(t * 0.11 + i) * 0.6;
      mp[i * 3 + 1] = this.motes.base[i * 3 + 1] + Math.sin(t * 0.13 + i * 0.7) * 0.4;
      mp[i * 3 + 2] = this.motes.base[i * 3 + 2] + Math.cos(t * 0.09 + i) * 0.6;
    }
    this.motes.touch(); this.motes.tick(t);
    this.stars.tick(t);

    // weather
    const w = this.weather || 'mist';
    this.weatherAmt.rain += (((w === 'rain' || w === 'storm') ? 1 : 0) - this.weatherAmt.rain) * Math.min(1, dt * 0.6);
    this.weatherAmt.snow += ((w === 'snow' || this.settingKind === 'snow' ? 1 : 0) - this.weatherAmt.snow) * Math.min(1, dt * 0.6);
    this._fall(this.rain, 16, 0, t, dt, this.weatherAmt.rain * 0.55);
    this._fall(this.snow, 1.1, 0.4, t, dt, this.weatherAmt.snow * 0.9);
    if (w === 'storm' && t > this.nextLightning) { this.dream.flash(new THREE.Color('#dfe8ff'), 0.7); this.nextLightning = t + 5 + Math.random() * 9; }

    // backdrops
    this.backdrops = this.backdrops.filter((b) => {
      b.fade += Math.sign(b.target - b.fade) * Math.min(Math.abs(b.target - b.fade), dt * 0.2);
      b.mesh.material.opacity = b.fade * (0.75 + 0.25 * haze);
      b.mesh.lookAt(this.dream.camera.position);
      if (b.target === 0 && b.fade <= 0) { this.scene.remove(b.mesh); disposeDeep(b.mesh); return false; }
      return true;
    });

    // effects
    this.effects = this.effects.filter((e) => {
      e.life += dt; const k = Math.min(1, e.life / e.dur); e.update(k, t, dt);
      if (k >= 1) { e.done(); return false; } return true;
    });
  }

  _fall(p, speed, sway, t, dt, amount) {
    p.visible = amount > 0.01; if (!p.visible) return;
    p.material.uniforms.uOpacity.value = amount;
    const pos = p.geometry.attributes.position.array;
    for (let i = 0; i < p.count; i++) {
      pos[i * 3 + 1] -= speed * dt * (0.8 + (i % 5) * 0.1);
      if (pos[i * 3 + 1] < -0.5) pos[i * 3 + 1] += 16;
      if (sway) pos[i * 3] = p.base[i * 3] + Math.sin(t * 0.7 + i) * sway;
    }
    p.touch(); p.tick(t);
  }
}
