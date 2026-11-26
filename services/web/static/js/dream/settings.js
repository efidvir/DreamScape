// Places the dream can be. A setting = palette (sky, fog, light) + a landscape group.
// The world blends palettes smoothly and dissolves landscapes in and out.
import * as THREE from 'three';
import { color, solid, rng, noise3, mistTex, Particles, glowSprite, dreamify } from './kit.js';
import { roughen, instanced, merge, moved } from './geo.js';
import * as flora from './flora.js';

const TIME = {
  night: { skyTop: '#03050d', skyHorizon: '#1f2d4d', fog: '#1d2a45', hemiSky: '#6f86c4', hemiGround: '#10111a', hemiI: 0.75, key: '#b4c8ff', keyI: 0.9, stars: 1 },
  dawn: { skyTop: '#22336a', skyHorizon: '#f0a58a', fog: '#b98a8f', hemiSky: '#ffc9a8', hemiGround: '#3a2f3a', hemiI: 0.95, key: '#ffb38a', keyI: 1.4, stars: 0.25 },
  day: { skyTop: '#3c7fd8', skyHorizon: '#cfe8ff', fog: '#b9d2e8', hemiSky: '#dff0ff', hemiGround: '#5a6b4a', hemiI: 1.3, key: '#fff1d6', keyI: 2.1, stars: 0 },
  dusk: { skyTop: '#1b1442', skyHorizon: '#ff8f63', fog: '#7a4c6c', hemiSky: '#ffb08a', hemiGround: '#2a1f2f', hemiI: 0.9, key: '#ff9a6a', keyI: 1.2, stars: 0.4 },
};

const SETTING = {
  void: { density: 0.1, override: { skyTop: '#000000', skyHorizon: '#04060c', fog: '#05070d', hemiI: 0.28, keyI: 0.15, stars: 0.12 }, mist: 1 },
  forest: { density: 0.045, mix: ['#1b3a31', 0.45], mist: 0.8 },
  meadow: { density: 0.02, mist: 0.3 },
  desert: { density: 0.02, mix: ['#e0b98a', 0.4], mist: 0.15 },
  ocean: { density: 0.022, mix: ['#5f86ad', 0.3], mist: 0.45 },
  mountains: { density: 0.016, mist: 0.4 },
  cave: { density: 0.065, override: { fog: '#0c0a12', skyTop: '#050408', skyHorizon: '#0c0a12', hemiI: 0.35, key: '#9f8cff', keyI: 0.35, stars: 0 }, mist: 0.6 },
  city: { density: 0.032, mix: ['#2a2438', 0.3], mist: 0.4 },
  space: { density: 0.005, override: { skyTop: '#010106', skyHorizon: '#0a0a1f', fog: '#05050e', hemiI: 0.4, keyI: 0.9, stars: 1 }, mist: 0.1 },
  moon: { density: 0.012, override: { skyTop: '#000002', skyHorizon: '#07080d', fog: '#0b0c12', hemiSky: '#9aa3b8', hemiGround: '#1b1b20', hemiI: 0.4, key: '#f2f4ff', keyI: 1.05, stars: 1 }, mist: 0.05 },
  clouds: { density: 0.006, override: { skyTop: '#3f78d0' }, mix: ['#b9cce6', 0.3], mist: 0.2 },
  snow: { density: 0.026, mix: ['#9fb2c9', 0.4], mist: 0.35 },
  underwater: { density: 0.055, override: { skyTop: '#021a26', skyHorizon: '#0b5a6e', fog: '#0a4a5c', hemiSky: '#5fd0e0', hemiGround: '#05202a', hemiI: 0.8, key: '#9fe8ff', keyI: 0.8, stars: 0 }, mist: 0.3 },
};

const WEATHER = { clear: 0.7, mist: 1.3, rain: 1.2, snow: 1.1, storm: 1.35 };

export function palette(kind, time, weather) {
  const t = TIME[time] || TIME.night;
  const s = SETTING[kind] || SETTING.void;
  const p = {};
  for (const k of ['skyTop', 'skyHorizon', 'fog', 'hemiSky', 'hemiGround', 'key']) p[k] = color(t[k]);
  p.hemiI = t.hemiI; p.keyI = t.keyI; p.stars = t.stars;
  if (s.mix) { const m = color(s.mix[0]); for (const k of ['fog', 'skyHorizon']) p[k].lerp(m, s.mix[1]); }
  if (s.override) for (const [k, v] of Object.entries(s.override)) p[k] = typeof v === 'number' ? v : color(v);
  p.fogDensity = s.density * (WEATHER[weather] || 1);
  p.mist = Math.min(1.3, s.mist * (weather === 'mist' ? 1.6 : weather === 'clear' ? 0.6 : 1));
  p.weather = weather;
  return p;
}

// ------------------------------------------------------------------ ground
function makeGround(opts) {
  const size = 280, seg = 140;
  const geo = new THREE.PlaneGeometry(size, size, seg, seg); geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position, cols = new Float32Array(pos.count * 3);
  const c1 = color(opts.color), c2 = color(opts.color2 || opts.color), tmp = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    const y = opts.height(x, z); pos.setY(i, y);
    const n = noise3(x * 0.08, 0, z * 0.08);
    tmp.copy(c1).lerp(c2, n); cols[i * 3] = tmp.r; cols[i * 3 + 1] = tmp.g; cols[i * 3 + 2] = tmp.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  const mat = dreamify(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: opts.rough ?? 0.95, metalness: 0,
    emissive: color(opts.color), emissiveIntensity: opts.glow ?? 0.06, flatShading: !!opts.flat }), opts.seam || '#9fc4ff');
  const ground = new THREE.Mesh(geo, mat);
  ground.userData.points = { count: opts.points ?? 60000, weight: (x, z) => 1 / (1 + (Math.hypot(x, z) / 13) ** 2), sizeByDistance: 8, size: 1.4, maxSize: 0.34, wander: 0.35, thickness: 0.04, warp: 0, flow: 0.07, swirl: 3 };
  return ground;
}

const flatten = (x, z, inner = 6, outer = 26) => {
  const d = Math.hypot(x, z);
  return Math.min(1, Math.max(0, (d - inner) / (outer - inner)));
};
const fbm = (x, z, f) => noise3(x * f, 0.5, z * f) * 0.65 + noise3(x * f * 2.3, 1.7, z * f * 2.3) * 0.35;

function ringPlace(r, n, rmin, rmax, heightAt, avoidFront = true) {
  const out = [];
  let guard = 0;
  while (out.length < n && guard++ < n * 20) {
    const a = r() * Math.PI * 2, d = rmin + (rmax - rmin) * Math.sqrt(r());
    const x = Math.cos(a) * d, z = Math.sin(a) * d;
    if (avoidFront && z > 0 && Math.abs(x) < 7 && d < 22) continue;    // keep the view open
    out.push(new THREE.Vector3(x, heightAt(x, z), z));
  }
  return out;
}

// ------------------------------------------------------------------ landscapes
const BUILD = {
  void(r) {
    const g = new THREE.Group();
    return { group: g, heightAt: () => 0 };
  },

  forest(r, time) {
    const g = new THREE.Group();
    const heightAt = (x, z) => fbm(x, z, 0.05) * 3.2 * flatten(x, z) - 0.02;
    g.add(makeGround({ color: '#14251a', color2: '#2a2416', height: heightAt }));
    const pts = ringPlace(r, 90, 15, 60, heightAt);
    const far = flora.trees({ count: pts.length, size: 'large', color: time === 'day' ? '#2d5e3a' : '#173823' }, r, (i) => pts[i], { glow: 0.07 });
    g.add(far.root);
    const ferns = ringPlace(r, 160, 3, 28, heightAt);
    g.add(instanced(roughen(new THREE.IcosahedronGeometry(0.45, 0), 0.3, 2), solid('#1d3a24', { glow: 0.08, flat: true }),
      ferns.map((p) => ({ p, s: 0.6 + r(), sy: 0.5 + 0.4 * r(), ry: r() * 6 }))));
    return { group: g, heightAt };
  },

  meadow(r) {
    const g = new THREE.Group();
    const heightAt = (x, z) => fbm(x, z, 0.03) * 4.5 * flatten(x, z, 8, 40);
    g.add(makeGround({ color: '#3b6a35', color2: '#6f8f3a', height: heightAt }));
    const tufts = ringPlace(r, 900, 1.5, 45, heightAt, false);
    const blade = merge([0, 1, 2].map((k) => moved(new THREE.ConeGeometry(0.035, 0.45, 3), Math.cos(k * 2.1) * 0.05, 0.2, Math.sin(k * 2.1) * 0.05, Math.cos(k) * 0.2, 0, Math.sin(k) * 0.2)));
    g.add(instanced(blade, solid('#4f8a3e', { glow: 0.12 }), tufts.map((p) => ({ p, s: 0.7 + r() * 0.9, ry: r() * 6 }))));
    const fl = ringPlace(r, 120, 2.5, 30, heightAt, false);
    const f = flora.flowers({ count: fl.length, size: 'small', color: '#ffd6f0' }, r, (i) => fl[i]);
    g.add(f.root);
    const trees = ringPlace(r, 14, 32, 70, heightAt);
    g.add(flora.trees({ count: trees.length, size: 'large', label: 'oak' }, r, (i) => trees[i]).root);
    return { group: g, heightAt };
  },

  desert(r) {
    const g = new THREE.Group();
    const heightAt = (x, z) => (Math.sin(x * 0.07 + Math.sin(z * 0.05) * 2) * 2.2 + fbm(x, z, 0.04) * 2) * flatten(x, z, 6, 30);
    g.add(makeGround({ color: '#c9a26b', color2: '#e2c08a', height: heightAt, glow: 0.12 }));
    const mesas = ringPlace(r, 9, 70, 130, () => 0, false);
    const mesaGeo = roughen(new THREE.CylinderGeometry(9, 13, 16, 9, 3), 0.08, 4);
    g.add(instanced(mesaGeo, solid('#a8673f', { glow: 0.1, flat: true }), mesas.map((p) => ({ p: p.setY(6), s: 0.7 + r(), sy: 0.6 + r() * 0.8, ry: r() * 6 }))));
    const rk = ringPlace(r, 14, 6, 30, heightAt);
    g.add(flora.rocks({ count: rk.length, size: 'medium' }, r, (i) => rk[i], { color: '#9a6a48' }).root);
    return { group: g, heightAt };
  },

  ocean(r, time) {
    const g = new THREE.Group();
    const mat = new THREE.ShaderMaterial({
      transparent: true, fog: true,
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uTime: { value: 0 }, uDeep: { value: color(time === 'day' ? '#1d5a8a' : '#0b2340') }, uShine: { value: color('#bfe3ff') }, uOpacity: { value: 1 } }]),
      vertexShader: `uniform float uTime; varying vec3 vW; varying float vH;
        #include <fog_pars_vertex>
        void main(){ vec3 p = position; float h = sin(p.x*0.18+uTime*0.9)*0.35 + sin(p.z*0.23-uTime*0.7)*0.28 + sin((p.x+p.z)*0.5+uTime*1.7)*0.08;
          p.y += h; vH = h; vec4 w = modelMatrix * vec4(p,1.0); vW = w.xyz; vec4 mvPosition = viewMatrix * w; gl_Position = projectionMatrix * mvPosition;
          #include <fog_vertex>
        }`,
      fragmentShader: `uniform vec3 uDeep, uShine; uniform float uOpacity; varying vec3 vW; varying float vH;
        #include <fog_pars_fragment>
        void main(){ vec3 col = mix(uDeep, uShine, smoothstep(0.2, 0.7, vH + 0.35) * 0.35);
          float sp = pow(max(0.0, sin(vW.x*1.7)*sin(vW.z*1.3)), 18.0); col += uShine * sp * 0.8;
          gl_FragColor = vec4(col, uOpacity);
          #include <fog_fragment>
        }`,
    });
    mat.userData.fadeBase = 1;
    Object.defineProperty(mat, 'opacity', { get() { return mat.uniforms.uOpacity.value; }, set(v) { mat.uniforms.uOpacity.value = v; } });
    const sea = new THREE.Mesh(new THREE.PlaneGeometry(320, 320, 160, 160).rotateX(-Math.PI / 2), mat);
    sea.position.y = -0.6; g.add(sea);
    const rocks = flora.rocks({ count: 5, size: 'large' }, r, (i) => new THREE.Vector3(Math.cos(i * 1.3 + 2) * 3.5, -0.7, Math.sin(i * 1.3 + 2) * 3.5 - 1));
    g.add(rocks.root);
    return { group: g, heightAt: () => -0.6, update: (t) => { mat.uniforms.uTime.value = t; } };
  },

  mountains(r) {
    const g = new THREE.Group();
    const heightAt = (x, z) => fbm(x, z, 0.045) * 5 * flatten(x, z, 7, 35);
    g.add(makeGround({ color: '#3d4a3f', color2: '#6b6a64', height: heightAt, flat: true }));
    const peaks = ringPlace(r, 16, 75, 150, () => 0, false);
    const peakGeo = roughen(new THREE.ConeGeometry(28, 60, 7, 4), 0.12, 6);
    const cols = new Float32Array(peakGeo.attributes.position.count * 3);
    for (let i = 0; i < peakGeo.attributes.position.count; i++) {
      const y = peakGeo.attributes.position.getY(i); const snow = y > 12 ? 1 : 0;
      cols.set(snow ? [0.92, 0.95, 1] : [0.33, 0.36, 0.42], i * 3);
    }
    peakGeo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    const pm = dreamify(new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 1, emissive: '#2a3040', emissiveIntensity: 0.25 }));
    g.add(instanced(peakGeo, pm, peaks.map((p) => ({ p: p.setY(18), s: 0.6 + r() * 0.8, sy: 0.6 + r() * 0.7, ry: r() * 6 }))));
    const pines = ringPlace(r, 40, 14, 50, heightAt);
    g.add(flora.trees({ count: pines.length, size: 'medium', label: 'pine', color: '#20402e' }, r, (i) => pines[i]).root);
    return { group: g, heightAt };
  },

  cave(r) {
    const g = new THREE.Group();
    const heightAt = (x, z) => fbm(x, z, 0.12) * 1.2 * flatten(x, z, 4, 14);
    g.add(makeGround({ color: '#1b1820', color2: '#2d2633', height: heightAt, flat: true }));
    const dome = roughen(new THREE.SphereGeometry(30, 48, 32), 0.12, 8, 0.6);
    const dm = solid('#1a1620', { glow: 0.06, side: THREE.BackSide, flat: true });
    const d = new THREE.Mesh(dome, dm); d.scale.y = 0.45; g.add(d);
    const tips = ringPlace(r, 70, 4, 26, () => 12, false);
    g.add(instanced(new THREE.ConeGeometry(0.5, 4, 6).rotateX(Math.PI), solid('#2a2430', { glow: 0.06, flat: true }),
      tips.map((p) => ({ p: p.setY(11 - r() * 2), s: 0.6 + r() * 1.4, sy: 0.6 + r() * 1.2 }))));
    const cr = ringPlace(r, 8, 4, 14, heightAt);
    g.add(flora.crystals({ count: cr.length, size: 'medium', color: r() < 0.5 ? '#8f7bff' : '#54e0ff' }, r, (i) => cr[i]).root);
    return { group: g, heightAt };
  },

  city(r, time) {
    const g = new THREE.Group();
    const heightAt = () => 0;
    g.add(makeGround({ color: '#16171d', color2: '#22232b', height: heightAt, glow: 0.04 }));
    const spots = ringPlace(r, 130, 14, 80, heightAt);
    const win = document.createElement('canvas'); win.width = 64; win.height = 128; const wg = win.getContext('2d');
    wg.fillStyle = '#0d0f16'; wg.fillRect(0, 0, 64, 128);
    for (let y = 4; y < 128; y += 10) for (let x = 4; x < 64; x += 10) { if (Math.random() < 0.45) { wg.fillStyle = Math.random() < 0.7 ? '#ffd98a' : '#9fd8ff'; wg.fillRect(x, y, 5, 6); } }
    const tex = new THREE.CanvasTexture(win); tex.colorSpace = THREE.SRGBColorSpace; tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(1, 2);
    const bm = dreamify(new THREE.MeshStandardMaterial({ color: '#1a1c26', map: tex, emissiveMap: tex, emissive: '#ffffff', emissiveIntensity: time === 'day' ? 0.25 : 0.9, roughness: 0.8 }));
    const box = new THREE.BoxGeometry(1, 1, 1); box.translate(0, 0.5, 0);
    g.add(instanced(box, bm, spots.map((p) => ({ p, sx: 4 + r() * 6, sz: 4 + r() * 6, sy: 6 + Math.pow(r(), 2) * 40, ry: Math.round(r() * 4) * Math.PI / 2 }))));
    for (let i = 0; i < 16; i++) { const a = i / 16 * 6.28, s = glowSprite('#ffcf8a', 2.2, 0.55); s.position.set(Math.cos(a) * 9, 3.2, Math.sin(a) * 9); g.add(s); }
    return { group: g, heightAt };
  },

  space(r) {
    const g = new THREE.Group();
    for (let i = 0; i < 16; i++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex(i % 4), color: color(r.pick(['#5a3fa8', '#1f6f8f', '#8f2f6a', '#2f3f9f'])), transparent: true, opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false }));
      const a = r() * 6.28, b = (r() - 0.5) * 1.2, d = 90 + r() * 80;
      sp.position.set(Math.cos(a) * Math.cos(b) * d, Math.sin(b) * d, Math.sin(a) * Math.cos(b) * d);
      sp.scale.setScalar(60 + r() * 80); sp.material.rotation = r() * 6; g.add(sp);
    }
    return { group: g, heightAt: () => -50 };
  },

  moon(r) {
    const g = new THREE.Group();
    const craters = Array.from({ length: 26 }, () => ({ x: (r() - 0.5) * 140, z: (r() - 0.5) * 140, rad: 2 + Math.pow(r(), 2) * 12 }));
    const heightAt = (x, z) => {
      let h = fbm(x, z, 0.06) * 0.8;
      for (const c of craters) {
        const d = Math.hypot(x - c.x, z - c.z) / c.rad;
        if (d < 1.3) h += d < 1 ? -0.9 * c.rad * 0.18 * (1 - d * d) : 0.7 * c.rad * 0.12 * (1.3 - d) / 0.3;
      }
      return h * flatten(x, z, 3, 12);
    };
    g.add(makeGround({ color: '#4d4e56', color2: '#2e2f35', height: heightAt, glow: 0.03 }));
    const rk = ringPlace(r, 30, 5, 40, heightAt, true);
    g.add(flora.rocks({ count: rk.length, size: 'small' }, r, (i) => rk[i], { color: '#7d7e86' }).root);
    return { group: g, heightAt };
  },

  clouds(r, time) {
    const g = new THREE.Group();
    const c = color(time === 'night' ? '#55607f' : time === 'dusk' ? '#e89f8c' : time === 'dawn' ? '#f0c2b6' : '#e8eef8');
    for (let i = 0; i < 70; i++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex(i % 4), color: c, transparent: true, opacity: 0.85, depthWrite: false }));
      const a = r() * 6.28, d = 2 + Math.sqrt(r()) * 70;
      sp.position.set(Math.cos(a) * d, -2.2 - r() * 1.5, Math.sin(a) * d);
      sp.scale.setScalar(10 + r() * 14); sp.material.rotation = r() * 6; g.add(sp);
    }
    return { group: g, heightAt: () => -1.2 };
  },

  snow(r, time) {
    const g = new THREE.Group();
    const heightAt = (x, z) => fbm(x, z, 0.04) * 3.5 * flatten(x, z);
    // Snow by night is moonlit blue-grey; a glowing white field would swallow the dreamer.
    const dark = time === 'night' || time === 'dusk';
    g.add(makeGround({ color: dark ? '#93a6c0' : '#dfe7f0', color2: dark ? '#7488a6' : '#c4d2e2', height: heightAt,
      glow: dark ? 0.05 : 0.14, seam: dark ? '#c9d6ea' : '#ffffff' }));
    const pines = ringPlace(r, 70, 14, 55, heightAt);
    g.add(flora.trees({ count: pines.length, size: 'large', label: 'pine', color: dark ? '#7f93ae' : '#cfdde8' }, r,
      (i) => pines[i], { glow: dark ? 0.05 : 0.12 }).root);
    return { group: g, heightAt };
  },

  underwater(r) {
    const g = new THREE.Group();
    const heightAt = (x, z) => fbm(x, z, 0.06) * 2 * flatten(x, z, 5, 25);
    g.add(makeGround({ color: '#7d7354', color2: '#5a6b5c', height: heightAt, glow: 0.08 }));
    const spots = ringPlace(r, 60, 3, 30, heightAt, false);
    const blade = new THREE.CylinderGeometry(0.04, 0.08, 1, 5); blade.translate(0, 0.5, 0);
    const weeds = instanced(blade, solid('#2f8a5a', { glow: 0.35 }), spots.map((p) => ({ p, sx: 1, sy: 1.5 + r() * 3, sz: 1 })));
    g.add(weeds);
    const bubbles = new Particles(140, { size: 0.12, colorA: '#bff6ff', twinkle: 0.2, opacity: 0.6, seed: 5 });
    for (let i = 0; i < 140; i++) bubbles.setBase(i, (r() - 0.5) * 30, r() * 12, (r() - 0.5) * 30);
    g.add(bubbles);
    for (let i = 0; i < 6; i++) {
      const shaft = new THREE.Mesh(new THREE.PlaneGeometry(2.5, 26), new THREE.MeshBasicMaterial({ color: '#bff6ff', transparent: true, opacity: 0.06, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
      shaft.position.set((r() - 0.5) * 18, 9, -4 - r() * 12); shaft.rotation.z = 0.25; g.add(shaft);
    }
    const pos = bubbles.geometry.attributes.position.array;
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3();
    return {
      group: g, heightAt,
      update: (t) => {
        for (let i = 0; i < 140; i++) { pos[i * 3 + 1] = (bubbles.base[i * 3 + 1] + t * 0.6) % 12; pos[i * 3] = bubbles.base[i * 3] + Math.sin(t + i) * 0.1; }
        bubbles.touch(); bubbles.tick(t);
        spots.forEach((p, i) => { e.set(Math.sin(t * 0.8 + i) * 0.12, 0, Math.cos(t * 0.6 + i) * 0.12); q.setFromEuler(e); s.set(1, 1.5 + (i * 37 % 10) / 3, 1); m.compose(p, q, s); weeds.setMatrixAt(i, m); });
        weeds.instanceMatrix.needsUpdate = true;
      },
    };
  },
};

export function buildSetting(kind, time, seed) {
  const r = rng(seed + ':' + kind);
  const make = BUILD[kind] || BUILD.void;
  const built = make(r, time);
  built.kind = kind;
  return built;
}

export const SETTING_KINDS = Object.keys(BUILD);
