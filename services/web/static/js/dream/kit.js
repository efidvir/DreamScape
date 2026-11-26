// Shared helpers for the dream renderer: seeded randomness, colours, soft textures,
// the "dream dissolve" material patch, particles and fading.
import * as THREE from 'three';

// ------------------------------------------------------------------ randomness
export function hashString(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

export function rng(seed) {
  let a = typeof seed === 'string' ? hashString(seed) : (seed >>> 0);
  const next = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.range = (lo, hi) => lo + (hi - lo) * next();
  next.pick = (arr) => arr[Math.floor(next() * arr.length) % arr.length];
  next.sign = () => (next() < 0.5 ? -1 : 1);
  return next;
}

// Smooth 1D/3D value noise, good enough for wandering motion.
function fract(x) { return x - Math.floor(x); }
function h1(n) { return fract(Math.sin(n) * 43758.5453123); }
export function noise1(x) {
  const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f);
  return h1(i) * (1 - u) + h1(i + 1) * u;   // 0..1
}
export function noise3(x, y, z) {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = x - ix, fy = y - iy, fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy), uz = fz * fz * (3 - 2 * fz);
  const n = ix + iy * 57 + iz * 113;
  const lerp = (a, b, t) => a + (b - a) * t;
  return lerp(
    lerp(lerp(h1(n), h1(n + 1), ux), lerp(h1(n + 57), h1(n + 58), ux), uy),
    lerp(lerp(h1(n + 113), h1(n + 114), ux), lerp(h1(n + 170), h1(n + 171), ux), uy), uz);
}
export const snoise = (x, seed = 0) => noise1(x + seed * 17.13) * 2 - 1;   // -1..1

// ------------------------------------------------------------------ colours
const _probe = new THREE.Color();
export function color(value, fallback = '#ffffff') {
  if (value instanceof THREE.Color) return value.clone();
  const v = (value || '').toString().trim().toLowerCase().replace(/\s+/g, '');
  const aliases = { silver: '#c8d4e6', gold: '#ffcf5a', golden: '#ffcf5a', darkgreen: '#1e4d2b',
    turquoise: '#40e0d0', rainbow: '#ff9ad5', glowing: '#fff2c4', transparent: '#dfefff',
    bronze: '#cd7f32', copper: '#c87533', ivory: '#fffff0', pearl: '#f0eadf', emerald: '#2ecc71',
    ruby: '#e0115f', sapphire: '#0f52ba', amethyst: '#9966cc', crimson: '#dc143c',
    // Plain colour words, as dream colours: pure blue or black would vanish as particles.
    blue: '#5b8cff', darkblue: '#3a5bd0', lightblue: '#9cc8ff', red: '#ff5a5a', darkred: '#b8333a',
    green: '#58d68d', yellow: '#ffe066', purple: '#b07cff', violet: '#b07cff', pink: '#ff9ad5',
    orange: '#ffa24c', black: '#33334a', brown: '#a8744a', grey: '#a9b2c0', gray: '#a9b2c0',
    navy: '#3a4f9e', indigo: '#6a5cff', magenta: '#ff5ad1', cyan: '#5ee7ff' };
  const hex = aliases[v] || v;
  try {
    if (/^#?[0-9a-f]{3}([0-9a-f]{3})?$/.test(hex)) return _probe.set(hex.startsWith('#') ? hex : '#' + hex).clone();
    const c = new THREE.Color();
    c.setStyle(hex, THREE.SRGBColorSpace);
    // setStyle leaves the colour untouched (white) for unknown names; detect that.
    if (hex !== 'white' && c.r === 1 && c.g === 1 && c.b === 1) throw new Error('unknown');
    return c;
  } catch { return new THREE.Color(fallback); }
}
export const SIZE = { tiny: 0.35, small: 0.65, medium: 1.0, large: 1.9, huge: 4.0 };

// ------------------------------------------------------------------ canvas textures
const _texCache = new Map();
function canvasTexture(key, w, h, draw, opts = {}) {
  if (_texCache.has(key)) return _texCache.get(key);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = opts.srgb === false ? THREE.NoColorSpace : THREE.SRGBColorSpace;
  if (opts.repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
  _texCache.set(key, t);
  return t;
}

export function softDot() {
  return canvasTexture('softdot', 64, 64, (g, w) => {
    const r = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    r.addColorStop(0, 'rgba(255,255,255,1)');
    r.addColorStop(0.25, 'rgba(255,255,255,0.75)');
    r.addColorStop(0.6, 'rgba(255,255,255,0.18)');
    r.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = r; g.fillRect(0, 0, w, w);
  });
}

export function sparkleTex() {
  return canvasTexture('sparkle', 64, 64, (g, w) => {
    const c = w / 2;
    const r = g.createRadialGradient(c, c, 0, c, c, c);
    r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.15, 'rgba(255,255,255,0.6)');
    r.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = r; g.fillRect(0, 0, w, w);
    g.globalCompositeOperation = 'lighter';
    for (const [dx, dy] of [[1, 0], [0, 1]]) {
      const lg = g.createLinearGradient(c - dx * c, c - dy * c, c + dx * c, c + dy * c);
      lg.addColorStop(0, 'rgba(255,255,255,0)'); lg.addColorStop(0.5, 'rgba(255,255,255,0.9)');
      lg.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = lg;
      if (dx) g.fillRect(0, c - 1.5, w, 3); else g.fillRect(c - 1.5, 0, 3, w);
    }
  });
}

// Wispy mist puff: layered blurred blobs.
export function mistTex(variant = 0) {
  return canvasTexture('mist' + variant, 256, 256, (g, w) => {
    const r = rng(1000 + variant);
    g.filter = 'blur(14px)';
    for (let i = 0; i < 26; i++) {
      const x = w * (0.2 + 0.6 * r()), y = w * (0.3 + 0.4 * r()), rad = w * (0.08 + 0.16 * r());
      const gr = g.createRadialGradient(x, y, 0, x, y, rad);
      gr.addColorStop(0, `rgba(255,255,255,${0.10 + 0.12 * r()})`);
      gr.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fill();
    }
    g.filter = 'none';
    // fade the edges so no square ever shows
    g.globalCompositeOperation = 'destination-in';
    const edge = g.createRadialGradient(w / 2, w / 2, w * 0.18, w / 2, w / 2, w * 0.5);
    edge.addColorStop(0, 'rgba(0,0,0,1)'); edge.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = edge; g.fillRect(0, 0, w, w);
  });
}

export function noiseTex(key, size, draw) { return canvasTexture(key, size, size, draw, { repeat: true }); }

// A gradient strip (top -> bottom) used by skies and wings.
export function gradientTex(key, stops, w = 4, h = 256) {
  return canvasTexture(key, w, h, (g) => {
    const gr = g.createLinearGradient(0, 0, 0, h);
    for (const [o, c] of stops) gr.addColorStop(o, c);
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
  });
}

// ------------------------------------------------------------------ dream dissolve
// Patches a standard/physical/lambert material so it can materialise out of the
// mist: fragments appear through 3D noise as `dissolve` goes 0 -> 1, with a
// glowing seam at the edge. Values just under 1 leave the object misty and torn.
const DISSOLVE_GLSL = `
uniform float uDissolve; uniform float uDreamTime; uniform vec3 uSeam;
varying vec3 vDreamPos;
float dHash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float dNoise(vec3 x) {
  vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(dHash(i + vec3(0,0,0)), dHash(i + vec3(1,0,0)), f.x),
                 mix(dHash(i + vec3(0,1,0)), dHash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(dHash(i + vec3(0,0,1)), dHash(i + vec3(1,0,1)), f.x),
                 mix(dHash(i + vec3(0,1,1)), dHash(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float dField(vec3 p) { return 0.65 * dNoise(p * 1.3 + uDreamTime * 0.07) + 0.35 * dNoise(p * 3.1 - uDreamTime * 0.05); }
`;

export const dreamClock = { value: 0 };

export function dreamify(mat, seam = '#bfe3ff') {
  if (mat.userData.dream) return mat;
  const u = { uDissolve: { value: 0 }, uSeam: { value: color(seam) } };
  mat.userData.dream = u;
  mat.transparent = true;
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uDissolve = u.uDissolve;
    shader.uniforms.uSeam = u.uSeam;
    shader.uniforms.uDreamTime = dreamClock;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vDreamPos;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        #ifdef USE_INSTANCING
          vDreamPos = (modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xyz;
        #else
          vDreamPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        #endif`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\n' + DISSOLVE_GLSL)
      .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
        float dN = dField(vDreamPos);
        float dT = uDissolve * 1.12 - 0.06;
        if (dN > dT) discard;
        float dEdge = 1.0 - smoothstep(0.0, 0.07, dT - dN);`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        totalEmissiveRadiance += uSeam * dEdge * 2.2 * step(uDissolve, 0.995);`);
  };
  mat.customProgramCacheKey = () => 'dream-dissolve';
  return mat;
}

// Collect everything fadeable under an object: dissolve uniforms + plain opacity materials.
export function collectFade(root) {
  const dissolves = [], opac = [], lights = [], points = [];
  root.traverse((o) => {
    if (o.isLight) { lights.push({ light: o, base: o.intensity }); return; }
    const mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
    for (const m of mats) {
      if (m.userData.dreamPoints) { points.push(m); continue; }
      if (m.visible === false) continue;
      if (m.userData.dream) dissolves.push(m.userData.dream.uDissolve);
      else if (m.userData.fadeBase === undefined) {
        m.userData.fadeBase = m.opacity ?? 1; m.transparent = true; opac.push(m);
      } else opac.push(m);
    }
  });
  return { dissolves, opac, lights, points };
}

export function applyFade(f, amount, flicker = 1) {
  const a = Math.max(0, Math.min(1, amount));
  for (const m of f.points || []) {
    const u = m.userData.dreamPoints;
    u.uForm.value = a;
    u.uOpacity.value = Math.min(1, a * 1.5) * (0.78 + 0.22 * flicker);
    u.uWander.value = m.userData.wander0 * (0.6 + 1.4 * (1 - flicker));
  }
  for (const u of f.dissolves) u.value = a * (0.9 + 0.1 * flicker);
  for (const m of f.opac) m.opacity = m.userData.fadeBase * a * (0.75 + 0.25 * flicker);
  for (const l of f.lights) l.light.intensity = l.base * a;
}

// ------------------------------------------------------------------ materials
export function solid(c, opts = {}) {
  return dreamify(new THREE.MeshStandardMaterial({
    color: color(c), roughness: opts.roughness ?? 0.75, metalness: opts.metalness ?? 0.05,
    emissive: color(opts.emissive ?? c).multiplyScalar(opts.glow ?? 0.12),
    flatShading: opts.flat ?? false, side: opts.side ?? THREE.FrontSide,
  }), opts.seam);
}

export function glowMat(c, opacity = 1) {
  return new THREE.MeshBasicMaterial({ color: color(c), transparent: true, opacity,
    blending: THREE.AdditiveBlending, depthWrite: false });
}

export function glowSprite(c, size = 1, opacity = 0.9) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: softDot(), color: color(c), transparent: true,
    opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
  s.scale.setScalar(size);
  return s;
}

// ------------------------------------------------------------------ particles
const POINTS_VS = `
attribute float aSize; attribute float aPhase; attribute vec3 aColor;
uniform float uTime; uniform float uPixel; uniform float uTwinkle;
varying vec3 vColor; varying float vAlpha;
void main() {
  vColor = aColor;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float tw = 1.0 - uTwinkle + uTwinkle * (0.5 + 0.5 * sin(uTime * (1.5 + aPhase) + aPhase * 6.283));
  vAlpha = tw;
  gl_PointSize = aSize * uPixel * tw * (300.0 / max(0.5, -mv.z));
  gl_Position = projectionMatrix * mv;
}`;
const POINTS_FS = `
uniform sampler2D uMap; uniform float uOpacity;
varying vec3 vColor; varying float vAlpha;
void main() {
  vec4 t = texture2D(uMap, gl_PointCoord);
  gl_FragColor = vec4(vColor * t.rgb, t.a * uOpacity * vAlpha);
}`;

export class Particles extends THREE.Points {
  constructor(n, { size = 0.1, sizeJitter = 0.5, colorA = '#ffffff', colorB = null, twinkle = 0.6,
    opacity = 1, map = null, seed = 1, additive = true } = {}) {
    const g = new THREE.BufferGeometry();
    const r = rng(seed);
    const pos = new Float32Array(n * 3), sz = new Float32Array(n), ph = new Float32Array(n), col = new Float32Array(n * 3);
    const ca = color(colorA), cb = colorB ? color(colorB) : ca, tmp = new THREE.Color();
    for (let i = 0; i < n; i++) {
      sz[i] = size * (1 - sizeJitter / 2 + sizeJitter * r());
      ph[i] = r() * 6.283;
      tmp.copy(ca).lerp(cb, r());
      col[i * 3] = tmp.r; col[i * 3 + 1] = tmp.g; col[i * 3 + 2] = tmp.b;
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(sz, 1));
    g.setAttribute('aPhase', new THREE.BufferAttribute(ph, 1));
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    const m = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uPixel: { value: Math.min(window.devicePixelRatio || 1, 1.5) },
        uTwinkle: { value: twinkle }, uOpacity: { value: opacity }, uMap: { value: map || softDot() } },
      vertexShader: POINTS_VS, fragmentShader: POINTS_FS, transparent: true, depthWrite: false,
      blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    super(g, m);
    this.count = n;
    this.base = new Float32Array(n * 3);   // resting positions for custom motion
    this.phase = ph;
    this.frustumCulled = false;
    m.userData.fadeBase = opacity;
    Object.defineProperty(m, 'opacity', {   // lets applyFade() drive the shader uniform
      get() { return m.uniforms.uOpacity.value; }, set(v) { m.uniforms.uOpacity.value = v; },
    });
  }

  setBase(i, x, y, z) {
    this.base[i * 3] = x; this.base[i * 3 + 1] = y; this.base[i * 3 + 2] = z;
    const p = this.geometry.attributes.position.array;
    p[i * 3] = x; p[i * 3 + 1] = y; p[i * 3 + 2] = z;
  }

  tick(t) { this.material.uniforms.uTime.value = t; }
  touch() { this.geometry.attributes.position.needsUpdate = true; }
}

// ------------------------------------------------------------------ disposal
export function disposeDeep(root) {
  root.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    const mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
    for (const m of mats) {
      for (const k of ['map', 'alphaMap', 'emissiveMap']) if (m[k] && !_isCached(m[k])) m[k].dispose();
      m.dispose();
    }
  });
}
function _isCached(tex) { for (const v of _texCache.values()) if (v === tex) return true; return false; }

export const tmpV = new THREE.Vector3();
export const UP = new THREE.Vector3(0, 1, 0);
