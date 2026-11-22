// Creatures, structures and objects. Each builder takes (spec, r, place) and returns
//   { root, members, height, update?(t, dt, env), world?: [Object3D], tintables?: [Material] }
// Members face +Z; the world moves them according to the spec's motion.
import * as THREE from 'three';
import { color, solid, glowSprite, glowMat, Particles, SIZE, softDot, sparkleTex, noiseTex, rng } from './kit.js';
import { merge, moved } from './geo.js';
import * as flora from './flora.js';

const has = (label, ...words) => words.some((w) => (label || '').toLowerCase().includes(w));
const v3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

function manyOf(spec, r, place, makeOne, opts = {}) {
  const root = new THREE.Group(); const members = []; const updates = []; const world = []; const tintables = [];
  const n = spec.count || 1;
  for (let i = 0; i < n; i++) {
    const one = makeOne(i, r);
    one.group.position.copy(place(i));
    if (opts.randomYaw !== false) one.group.rotation.y = r() * 6.28;
    root.add(one.group); members.push(one.group);
    if (one.update) updates.push(one.update);
    if (one.world) world.push(...one.world);
    if (one.tint) tintables.push(...one.tint);
  }
  return {
    root, members, world, tintables, height: opts.height ?? 1,
    update: (t, dt, env) => { for (const u of updates) u(t, dt, env); },
  };
}

// ------------------------------------------------------------------ wings
const _wingTex = {};
function wingTexture(kind) {
  if (_wingTex[kind]) return _wingTex[kind];
  const c = document.createElement('canvas'); c.width = 128; c.height = 128; const g = c.getContext('2d');
  g.translate(64, 64);
  if (kind === 'butterfly') {
    const gr = g.createRadialGradient(-10, -10, 4, 0, 0, 62);
    gr.addColorStop(0, 'rgba(255,255,255,0.95)'); gr.addColorStop(0.65, 'rgba(255,255,255,0.8)'); gr.addColorStop(1, 'rgba(40,20,60,0.95)');
    g.fillStyle = gr; g.beginPath(); g.ellipse(0, 0, 60, 46, 0.5, 0, Math.PI * 2); g.fill();
    g.fillStyle = 'rgba(30,10,40,0.7)';
    for (const [x, y, rr] of [[18, -14, 9], [-6, 18, 6], [26, 12, 5]]) { g.beginPath(); g.arc(x, y, rr, 0, 6.28); g.fill(); }
  } else {
    const gr = g.createRadialGradient(-20, 0, 2, 0, 0, 64);
    gr.addColorStop(0, 'rgba(255,255,255,0.85)'); gr.addColorStop(0.7, 'rgba(255,255,255,0.35)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.beginPath(); g.ellipse(0, 0, 62, 30, 0, 0, Math.PI * 2); g.fill();
    g.strokeStyle = 'rgba(255,255,255,0.55)'; g.lineWidth = 1.2;
    for (let k = -2; k <= 2; k++) { g.beginPath(); g.moveTo(-58, 0); g.quadraticCurveTo(0, k * 9, 56, k * 12); g.stroke(); }
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; _wingTex[kind] = t; return t;
}

function wing(kind, c, w, h, additive = true) {
  const geo = new THREE.PlaneGeometry(w, h); geo.translate(w / 2, 0, 0);   // hinge at the body
  const mat = new THREE.MeshBasicMaterial({ map: wingTexture(kind), color: c, transparent: true, opacity: 0.9,
    side: THREE.DoubleSide, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });
  return new THREE.Mesh(geo, mat);
}

// A trail of glittering dust that stays behind in world space.
function dustTrail(c, n = 110, size = 0.17) {
  const p = new Particles(n, { size, colorA: '#ffffff', colorB: c, twinkle: 0.9, map: sparkleTex(), seed: 5 });
  const age = new Float32Array(n).fill(1); let head = 0; const wp = new THREE.Vector3();
  return {
    points: p,
    follow(obj, t, dt) {
      obj.getWorldPosition(wp);
      for (let k = 0; k < 2; k++) {
        head = (head + 1) % n; age[head] = 0;
        p.setBase(head, wp.x + (Math.random() - 0.5) * 0.06, wp.y + (Math.random() - 0.5) * 0.06, wp.z + (Math.random() - 0.5) * 0.06);
      }
      const pos = p.geometry.attributes.position.array, sz = p.geometry.attributes.aSize.array;
      for (let i = 0; i < n; i++) {
        age[i] = Math.min(1, age[i] + dt * 0.4);
        pos[i * 3 + 1] -= dt * 0.12;
        sz[i] = size * (1 - age[i]);
      }
      p.geometry.attributes.aSize.needsUpdate = true; p.touch(); p.tick(t);
    },
  };
}

// ------------------------------------------------------------------ fairy, butterfly, fireflies
function fairy(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 3.4;
  const c = color(spec.color, '#bfe6ff');
  return manyOf(spec, r, place, (i) => {
    const g = new THREE.Group(); const body = new THREE.Group(); body.scale.setScalar(s); g.add(body);
    const skin = solid('#fff4e6', { glow: 0.9, emissive: '#ffe9cf' });
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.03, 0.1, 4, 8), skin); torso.position.y = 0;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.04, 12, 8), skin); head.position.y = 0.1;
    const dress = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.12, 10, 1, true), solid(c, { glow: 0.9, side: THREE.DoubleSide }));
    dress.position.y = -0.04;
    body.add(torso, head, dress);
    const wings = [];
    for (const [side, up, w, h] of [[1, 0.06, 0.3, 0.22], [-1, 0.06, 0.3, 0.22], [1, -0.03, 0.2, 0.14], [-1, -0.03, 0.2, 0.14]]) {
      const wg = wing('fairy', c, w, h); wg.position.set(0, up, -0.02); wg.scale.x = side; body.add(wg); wings.push(wg);
    }
    const halo = glowSprite(c, 0.5 * s, 0.22); const core = glowSprite('#ffffff', 0.08 * s, 0.6);
    g.add(halo, core);
    const light = new THREE.PointLight(c, 1.6, 4.5 * s, 1.5); g.add(light);
    const trail = dustTrail(c);
    const phase = i * 1.7;
    return {
      group: g, world: [trail.points], tint: [dress.material],
      update: (t, dt) => {
        const f = 0.9 + 0.6 * Math.sin(t * 34 + phase);
        wings.forEach((wg, k) => { wg.rotation.y = (k % 2 ? -1 : 1) * f * (k < 2 ? 0.9 : 0.75); });
        halo.material.opacity = 0.16 + 0.08 * Math.sin(t * 3 + phase);
        trail.follow(g, t, dt);
      },
    };
  }, { height: 0.3 * s });
}

function butterfly(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 1.1;
  return manyOf(spec, r, place, (i, rr) => {
    const c = color(spec.color, rr.pick(['#7fd3ff', '#ffb36b', '#d58bff', '#ff7fb0']));
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    b.add(new THREE.Mesh(new THREE.CapsuleGeometry(0.012, 0.08, 3, 6), solid('#2a1f2e', { glow: 0.2 })));
    b.children[0].rotation.x = Math.PI / 2;
    const L = wing('butterfly', c, 0.12, 0.1, false), R = wing('butterfly', c, 0.12, 0.1, false);
    R.scale.x = -1; b.add(L, R);
    for (const w of [L, R]) { w.rotation.x = -Math.PI / 2; w.material.opacity = 0.95; }
    g.add(glowSprite(c, 0.3 * s, 0.35));
    const ph = i * 2.1;
    return { group: g, update: (t) => { const a = 0.15 + 1.1 * (0.5 + 0.5 * Math.sin(t * 15 + ph)); L.rotation.y = a; R.rotation.y = -a; } };
  }, { height: 0.1 * s });
}

function fireflies(spec, r, place) {
  const n = Math.max(12, spec.count || 20);
  const c = color(spec.color, '#ffd56a');
  const p = new Particles(n, { size: 0.24 * Math.pow(SIZE[spec.size] || 1, 0.35), colorA: c, colorB: '#fff7d1', twinkle: 0.95, seed: 11 });
  for (let i = 0; i < n; i++) { const q = place(i); p.setBase(i, q.x, q.y + 0.4 + r() * 2.2, q.z); }
  const root = new THREE.Group(); root.add(p);
  const pos = p.geometry.attributes.position.array;
  return {
    root, members: [], height: 1,
    update: (t) => {
      for (let i = 0; i < n; i++) {
        const ph = p.phase[i];
        pos[i * 3] = p.base[i * 3] + Math.sin(t * 0.31 + ph) * 0.7;
        pos[i * 3 + 1] = p.base[i * 3 + 1] + Math.sin(t * 0.47 + ph * 1.3) * 0.35;
        pos[i * 3 + 2] = p.base[i * 3 + 2] + Math.cos(t * 0.27 + ph) * 0.7;
      }
      p.touch(); p.tick(t);
    },
  };
}

// ------------------------------------------------------------------ birds & fish
function bird(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 0.9;
  const c = color(spec.color, '#2b2f3a');
  return manyOf(spec, r, place, (i) => {
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    const mat = solid(c, { glow: 0.35 });
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 8), mat); body.scale.set(0.8, 0.7, 1.9);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.055, 8, 6), mat); head.position.set(0, 0.04, 0.17);
    const beak = new THREE.Mesh(new THREE.ConeGeometry(0.018, 0.06, 5), solid('#e8b04a', { glow: 0.3 }));
    beak.rotation.x = Math.PI / 2; beak.position.set(0, 0.035, 0.235);
    const wg = new THREE.BufferGeometry().setFromPoints([v3(0, 0, 0.07), v3(0.42, 0, -0.02), v3(0, 0, -0.1)]);
    wg.computeVertexNormals();
    const wm = solid(c, { glow: 0.35, side: THREE.DoubleSide });
    const L = new THREE.Mesh(wg, wm), R = new THREE.Mesh(wg, wm); R.scale.x = -1;
    b.add(body, head, beak, L, R);
    const ph = i * 1.3;
    return { group: g, tint: [mat, wm], update: (t) => { const a = Math.sin(t * 9 + ph) * 0.7; L.rotation.z = a; R.rotation.z = -a; } };
  }, { height: 0.15 * s });
}

function fish(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 0.9;
  return manyOf(spec, r, place, (i, rr) => {
    const c = color(spec.color, rr.pick(['#ff9f43', '#4fc3f7', '#f06292', '#ffd54f']));
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    const mat = solid(c, { glow: 0.55, roughness: 0.3, metalness: 0.3 });
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 10), mat); body.scale.set(0.45, 0.8, 1.4);
    const tailG = new THREE.ConeGeometry(0.09, 0.14, 4); tailG.rotateX(-Math.PI / 2); tailG.scale(0.25, 1, 1);
    const tail = new THREE.Mesh(tailG, mat); tail.position.z = -0.2;
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.015, 6, 4), solid('#111111', { glow: 0 }));
    eye.position.set(0.045, 0.03, 0.1); const eye2 = eye.clone(); eye2.position.x = -0.045;
    b.add(body, tail, eye, eye2);
    const ph = i * 0.9;
    return { group: g, tint: [mat], update: (t) => { tail.rotation.y = Math.sin(t * 8 + ph) * 0.5; } };
  }, { height: 0.2 * s });
}

// ------------------------------------------------------------------ four-legged animals
const ANIMALS = {
  cat: { len: 0.42, r: 0.1, leg: 0.2, head: 0.095, snout: 0.03, ears: 'pointy', tail: 'curl', color: '#b9a48f' },
  dog: { len: 0.55, r: 0.13, leg: 0.28, head: 0.11, snout: 0.09, ears: 'floppy', tail: 'up', color: '#a5703f' },
  wolf: { len: 0.72, r: 0.15, leg: 0.4, head: 0.13, snout: 0.15, ears: 'pointy', tail: 'bushy', color: '#8d939e' },
  horse: { len: 1.15, r: 0.26, leg: 0.85, head: 0.15, snout: 0.25, ears: 'small', tail: 'long', neck: 0.55, color: '#7a5a43' },
};

function quadruped(kind) {
  return (spec, r, place) => {
    const P = ANIMALS[kind];
    const s = (SIZE[spec.size] || 1) * (kind === 'horse' ? 1.0 : 1.3);
    const unicorn = kind === 'horse' && has(spec.label, 'unicorn', 'חד-קרן', 'חד קרן');
    const deer = kind === 'horse' && has(spec.label, 'deer', 'stag', 'elk', 'אייל', 'צבי');
    const fox = kind === 'wolf' && has(spec.label, 'fox', 'שועל');
    const base = unicorn ? '#f2f0ff' : fox ? '#d9772f' : deer ? '#a0703f' : P.color;
    return manyOf(spec, r, place, (i) => {
      const c = color(spec.color, base);
      const mat = solid(c, { glow: unicorn ? 0.45 : 0.22, roughness: 0.8 });
      const g = new THREE.Group(); const a = new THREE.Group(); a.scale.setScalar(s); g.add(a);
      const L = P.len * (deer ? 0.85 : 1), R = P.r * (deer ? 0.8 : 1), legH = P.leg * (deer ? 1.15 : 1);
      const body = new THREE.Mesh(new THREE.CapsuleGeometry(R, L, 6, 12), mat);
      body.rotation.x = Math.PI / 2; body.position.y = legH + R * 0.9; a.add(body);
      const legs = [];
      for (const [x, z] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
        const leg = new THREE.Group(); leg.position.set(x * R * 0.6, legH + R * 0.4, z * L * 0.42);
        const m = new THREE.Mesh(new THREE.CapsuleGeometry(R * 0.28, legH, 3, 6), mat); m.position.y = -legH / 2;
        leg.add(m); a.add(leg); legs.push(leg);
      }
      const neckLen = P.neck || R * 1.2;
      const neck = new THREE.Group(); neck.position.set(0, legH + R * 1.2, L * 0.5); neck.rotation.x = kind === 'horse' ? -0.55 : -0.3;
      a.add(neck);
      const nk = new THREE.Mesh(new THREE.CapsuleGeometry(R * 0.55, neckLen, 4, 8), mat); nk.position.y = neckLen / 2; neck.add(nk);
      const head = new THREE.Group(); head.position.y = neckLen + P.head * 0.3; head.rotation.x = kind === 'horse' ? 0.9 : 0.35; neck.add(head);
      const skull = new THREE.Mesh(new THREE.SphereGeometry(P.head, 12, 10), mat); head.add(skull);
      if (P.snout > 0.02) {
        const sn = new THREE.Mesh(new THREE.CapsuleGeometry(P.head * 0.55, P.snout, 4, 8), mat);
        sn.rotation.x = Math.PI / 2; sn.position.set(0, -P.head * 0.25, P.head * 0.6 + P.snout * 0.4); head.add(sn);
      }
      const eyeM = solid(unicorn ? '#7ad7ff' : '#ffe9a8', { glow: 1.6 });
      for (const x of [1, -1]) {
        const e = new THREE.Mesh(new THREE.SphereGeometry(P.head * 0.14, 6, 4), eyeM); e.position.set(x * P.head * 0.55, P.head * 0.2, P.head * 0.72); head.add(e);
        if (P.ears === 'pointy' || P.ears === 'small') {
          const ear = new THREE.Mesh(new THREE.ConeGeometry(P.head * 0.32, P.head * (P.ears === 'small' ? 0.6 : 0.9), 4), mat);
          ear.position.set(x * P.head * 0.55, P.head * 0.95, -P.head * 0.1); ear.rotation.z = -x * 0.25; head.add(ear);
        } else if (P.ears === 'floppy') {
          const ear = new THREE.Mesh(new THREE.SphereGeometry(P.head * 0.4, 8, 6), mat); ear.scale.set(0.45, 1.2, 0.8);
          ear.position.set(x * P.head * 0.95, P.head * 0.1, -P.head * 0.1); head.add(ear);
        }
      }
      if (unicorn) {
        const horn = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.42, 8), solid('#ffe7a3', { glow: 1.4 }));
        horn.position.set(0, P.head * 1.25, P.head * 0.55); horn.rotation.x = 0.55; head.add(horn);
        head.add(glowSprite('#bfe3ff', 0.7, 0.5));
      }
      if (deer) {
        const am = solid('#d8c3a0', { glow: 0.3 });
        for (const x of [1, -1]) {
          const ant = new THREE.Group(); ant.position.set(x * P.head * 0.45, P.head * 0.9, 0); ant.rotation.z = -x * 0.45; head.add(ant);
          const main = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.02, 0.4, 4), am); main.position.y = 0.2; ant.add(main);
          for (const h of [0.14, 0.26]) { const tine = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.012, 0.16, 4), am); tine.position.set(-x * 0.05, h, 0.04); tine.rotation.z = x * 0.8; ant.add(tine); }
        }
      }
      const tail = new THREE.Group(); tail.position.set(0, legH + R * 1.1, -L * 0.55); a.add(tail);
      const tl = P.tail === 'long' ? 0.6 : P.tail === 'bushy' ? 0.4 : 0.35;
      const tr = P.tail === 'bushy' ? R * 0.45 : R * 0.16;
      const tm = new THREE.Mesh(new THREE.CapsuleGeometry(tr, tl, 3, 6), mat); tm.position.y = tl / 2; tail.add(tm);
      tail.rotation.x = P.tail === 'up' || P.tail === 'curl' ? -0.5 : P.tail === 'long' ? -2.6 : -2.2;
      if (kind === 'horse') {
        const mane = new THREE.Mesh(new THREE.BoxGeometry(0.05, neckLen * 0.95, 0.12), solid(unicorn ? '#c9b3ff' : '#3b2a20', { glow: unicorn ? 0.9 : 0.1 }));
        mane.position.set(0, neckLen / 2, -R * 0.45); neck.add(mane);
      }
      const ph = i * 1.9;
      return {
        group: g, tint: [mat],
        update: (t, dt, env) => {
          const walking = env && env.moving;
          const sp = walking ? 7 : 0;
          legs.forEach((lg, k) => { lg.rotation.x = walking ? Math.sin(t * sp + ph + (k === 0 || k === 3 ? 0 : Math.PI)) * 0.45 : 0; });
          tail.rotation.z = Math.sin(t * 2.2 + ph) * 0.25;
          head.rotation.y = Math.sin(t * 0.4 + ph) * 0.25;
          body.scale.x = 1 + Math.sin(t * 1.6 + ph) * 0.015;
        },
      };
    }, { height: P.leg + P.r * 2 });
  };
}

// ------------------------------------------------------------------ dragon
function dragon(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 1.6;
  const c = color(spec.color, '#5a2fb0');
  const root = new THREE.Group(); const members = []; const world = [];
  const head = new THREE.Group(); head.position.copy(place(0)); root.add(head); members.push(head);
  const mat = solid(c, { glow: 0.45, roughness: 0.55, metalness: 0.2 });
  const skull = new THREE.Mesh(new THREE.SphereGeometry(0.16 * s, 12, 10), mat); skull.scale.set(0.85, 0.7, 1.4); head.add(skull);
  const eyeM = solid('#ffd84a', { glow: 2.2 });
  for (const x of [1, -1]) {
    const e = new THREE.Mesh(new THREE.SphereGeometry(0.03 * s, 6, 4), eyeM); e.position.set(x * 0.08 * s, 0.05 * s, 0.12 * s); head.add(e);
    const horn = new THREE.Mesh(new THREE.ConeGeometry(0.03 * s, 0.22 * s, 6), solid('#e6d6b0', { glow: 0.3 }));
    horn.position.set(x * 0.07 * s, 0.12 * s, -0.08 * s); horn.rotation.x = -0.9; head.add(horn);
  }
  const SEG = 16; const segs = [];
  for (let k = 0; k < SEG; k++) {
    const rad = 0.13 * s * (1 - k / (SEG + 2)) + 0.02 * s;
    const m = new THREE.Mesh(new THREE.SphereGeometry(rad, 10, 8), mat); world.push(m); segs.push(m);
  }
  const wingG = new THREE.BufferGeometry().setFromPoints([v3(0, 0, 0.2 * s), v3(1.4 * s, 0.3 * s, -0.1 * s), v3(0.9 * s, 0, -0.6 * s), v3(0, 0, -0.3 * s)]);
  wingG.setIndex([0, 1, 2, 0, 2, 3]); wingG.computeVertexNormals();
  const wingM = solid(c.clone().offsetHSL(0.03, 0, 0.12), { glow: 0.55, side: THREE.DoubleSide });
  wingM.opacity = 0.92;
  const wl = new THREE.Mesh(wingG, wingM), wr = new THREE.Mesh(wingG, wingM); wr.scale.x = -1;
  const wings = new THREE.Group(); wings.add(wl, wr); world.push(wings);
  const glow = glowSprite(c, 2.4 * s, 0.25); world.push(glow);
  const hist = []; const wp = new THREE.Vector3();
  const update = (t) => {
    head.getWorldPosition(wp);
    hist.unshift(wp.clone()); if (hist.length > SEG * 4 + 4) hist.pop();
    const tail = hist[Math.min(hist.length - 1, SEG * 4)] || wp;
    const resting = tail.distanceToSquared(wp) < 0.04 * s * s;
    for (let k = 0; k < SEG; k++) {
      if (resting) {   // perched: the body curls behind the head
        const a = k * 0.42 + Math.sin(t * 0.6) * 0.15;
        segs[k].position.set(wp.x + Math.sin(a) * 0.28 * s * (k * 0.3), wp.y - 0.06 * s * k, wp.z - Math.cos(a) * 0.22 * s * (k * 0.35 + 1));
      } else {
        segs[k].position.copy(hist[Math.min(hist.length - 1, (k + 1) * 4)] || wp);
      }
      segs[k].position.y += Math.sin(t * 3 - k * 0.5) * 0.04 * s;
    }
    const anchor = segs[3].position;
    wings.position.copy(anchor);
    const ahead = segs[1].position, behind = segs[5].position;
    if (ahead.distanceToSquared(behind) > 1e-6) wings.lookAt(ahead.clone().multiplyScalar(2).sub(behind));
    const f = Math.sin(t * 3.2) * 0.6; wl.rotation.z = f; wr.rotation.z = -f;
    glow.position.copy(segs[6].position);
  };
  return { root, members, world, update, height: 0.5 * s, tintables: [mat, wingM] };
}

// ------------------------------------------------------------------ people & spirits
function person(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 1.05;
  const divine = has(spec.label, 'god', 'goddess', 'angel', 'king', 'queen', 'אל', 'מלך', 'מלכה', 'מלאך');
  return manyOf(spec, r, place, (i, rr) => {
    const c = color(spec.color, rr.pick(['#9fb4d8', '#c9a0dc', '#d8b49f']));
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    const mat = solid(c, { glow: 0.35, roughness: 0.9, side: THREE.DoubleSide });
    const pts = [v3(0, 0, 0), v3(0.26, 0, 0), v3(0.22, 0.3, 0), v3(0.15, 0.9, 0), v3(0.18, 1.18, 0), v3(0.06, 1.32, 0), v3(0, 1.34, 0)];
    const robe = new THREE.Mesh(new THREE.LatheGeometry(pts.map((p) => new THREE.Vector2(p.x, p.y)), 16), mat);
    const skin = solid('#f1d9c2', { glow: 0.45 });
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 14, 10), skin); head.position.y = 1.47;
    const arms = [];
    for (const x of [1, -1]) {
      const arm = new THREE.Group(); arm.position.set(x * 0.19, 1.22, 0);
      const am = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.5, 3, 6), mat); am.position.y = -0.27; arm.add(am);
      arm.rotation.z = x * 0.18; b.add(arm); arms.push(arm);
    }
    b.add(robe, head);
    if (divine) { const halo = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.012, 6, 32), glowMat('#ffe08a', 0.95)); halo.position.y = 1.72; halo.rotation.x = Math.PI / 2; b.add(halo); }
    g.add(glowSprite(divine ? '#ffe08a' : c, 1.6 * s, divine ? 0.45 : 0.18)); g.children[1].position.y = 0.9 * s;
    const ph = i * 1.4;
    return { group: g, tint: [mat], update: (t) => { arms.forEach((a, k) => { a.rotation.x = Math.sin(t * 0.9 + ph + k) * 0.12; }); b.scale.y = s * (1 + Math.sin(t * 1.3 + ph) * 0.01); } };
  }, { height: 1.6 * s });
}

function ghost(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 1.1;
  const c = color(spec.color, '#cfe6ff');
  return manyOf(spec, r, place, (i) => {
    const g = new THREE.Group();
    const pts = []; for (let k = 0; k <= 12; k++) { const y = k / 12; pts.push(new THREE.Vector2(0.32 * Math.sqrt(Math.sin(Math.min(1, y * 1.15) * Math.PI * 0.5 + 0.0001)) * (y > 0.9 ? Math.cos((y - 0.9) * 15) : 1), y * 1.1)); }
    pts.reverse();
    const geo = new THREE.LatheGeometry(pts, 20);
    const pos = geo.attributes.position; const base = pos.array.slice();
    const mat = new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
    const sheet = new THREE.Mesh(geo, mat); sheet.scale.setScalar(s); g.add(sheet);
    for (const x of [1, -1]) { const e = new THREE.Mesh(new THREE.SphereGeometry(0.035 * s, 8, 6), new THREE.MeshBasicMaterial({ color: '#0b1020', transparent: true, opacity: 0.9 })); e.position.set(x * 0.09 * s, 0.85 * s, 0.25 * s); g.add(e); }
    g.add(glowSprite(c, 1.8 * s, 0.35)); g.children[g.children.length - 1].position.y = 0.6 * s;
    const ph = i * 2.3;
    return { group: g, update: (t) => {
      for (let k = 0; k < pos.count; k++) {
        const y = base[k * 3 + 1];
        if (y < 0.35) { const w = Math.sin(t * 3 + ph + base[k * 3] * 9 + base[k * 3 + 2] * 7) * 0.04 * (0.35 - y) * 3; pos.setX(k, base[k * 3] * (1 + w)); pos.setZ(k, base[k * 3 + 2] * (1 + w)); }
      }
      pos.needsUpdate = true;
    } };
  }, { height: 1.1 * s });
}

// ------------------------------------------------------------------ sky things
function craterTexture(seed, tintHex) {
  return noiseTex('crater' + seed + tintHex, 256, (g, w) => {
    const r = rng(seed); g.fillStyle = tintHex; g.fillRect(0, 0, w, w);
    for (let k = 0; k < 160; k++) {
      const x = r() * w, y = r() * w, rad = Math.pow(r(), 3) * 26 + 2;
      const gr = g.createRadialGradient(x, y, rad * 0.2, x, y, rad);
      gr.addColorStop(0, 'rgba(0,0,0,0.28)'); gr.addColorStop(0.75, 'rgba(0,0,0,0.12)'); gr.addColorStop(0.9, 'rgba(255,255,255,0.12)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, rad, 0, 6.28); g.fill();
    }
  });
}

function moonThing(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 2.6;
  const c = color(spec.color, '#e8ecf5');
  const g = new THREE.Group(); g.position.copy(place(0));
  const mat = new THREE.MeshStandardMaterial({ map: craterTexture(4, '#c9ccd6'), color: c, emissive: c, emissiveIntensity: 0.55, roughness: 1 });
  g.add(new THREE.Mesh(new THREE.SphereGeometry(s, 40, 28), mat));
  g.add(glowSprite(c, s * 6, 0.35));
  const root = new THREE.Group(); root.add(g);
  return { root, members: [g], height: s, tintables: [mat] };
}

function sun(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 3;
  const c = color(spec.color, '#ffd27a');
  const g = new THREE.Group(); g.position.copy(place(0));
  g.add(new THREE.Mesh(new THREE.SphereGeometry(s, 32, 20), glowMat(c, 1)));
  const a = glowSprite(c, s * 5, 0.75), b = glowSprite('#fff4d6', s * 2.6, 0.9); g.add(a, b);
  const root = new THREE.Group(); root.add(g);
  return { root, members: [g], height: s, update: (t) => { a.material.rotation = t * 0.02; a.scale.setScalar(s * (5 + 0.4 * Math.sin(t * 0.7))); } };
}

function planetTexture(kind, base) {
  return noiseTex('planet-' + kind + base, 256, (g, w) => {
    const r = rng(kind.length * 31 + base.length);
    if (kind === 'earth') {
      g.fillStyle = '#1d4f9a'; g.fillRect(0, 0, w, w);
      for (let k = 0; k < 22; k++) { g.fillStyle = r() < 0.5 ? '#3f8a4a' : '#7a6b3f'; g.beginPath(); g.ellipse(r() * w, (0.15 + 0.7 * r()) * w, 10 + r() * 34, 6 + r() * 20, r() * 3, 0, 6.28); g.fill(); }
      g.fillStyle = 'rgba(255,255,255,0.35)'; for (let k = 0; k < 40; k++) { g.beginPath(); g.ellipse(r() * w, (0.1 + 0.8 * r()) * w, 6 + r() * 14, 5 + r() * 9, r() * 3, 0, 6.28); g.fill(); }
    } else {
      for (let y = 0; y < w; y += 4) { const k = 0.75 + 0.25 * Math.sin(y * 0.11 + r() * 2) + 0.1 * r(); g.fillStyle = `rgba(${Math.round(255 * k)},${Math.round(235 * k)},${Math.round(210 * k)},1)`; g.fillRect(0, y, w, 4); }
      g.globalCompositeOperation = 'multiply'; g.fillStyle = base; g.fillRect(0, 0, w, w);
    }
  });
}

function planet(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 2.2;
  const earth = has(spec.label, 'earth', 'world', 'כדור הארץ', 'ארץ') || (!spec.label && /blue/.test(spec.color || ''));
  const ringed = has(spec.label, 'saturn', 'ring', 'שבתאי', 'טבעת');
  const c = color(spec.color, earth ? '#4f8cff' : '#d8b48a');
  const g = new THREE.Group(); g.position.copy(place(0));
  const mat = new THREE.MeshStandardMaterial({ map: planetTexture(earth ? 'earth' : 'gas', '#' + c.getHexString()), emissive: '#ffffff', emissiveIntensity: 0.18, roughness: 0.9 });
  mat.emissiveMap = mat.map;
  const ball = new THREE.Mesh(new THREE.SphereGeometry(s, 40, 28), mat); ball.rotation.z = 0.35; g.add(ball);
  g.add(glowSprite(earth ? '#7fb7ff' : c, s * 3.2, 0.4));
  if (ringed) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(s * 1.35, s * 2.2, 64), new THREE.MeshBasicMaterial({ color: c.clone().offsetHSL(0, -0.1, 0.1), transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = Math.PI / 2.4; g.add(ring);
  }
  const root = new THREE.Group(); root.add(g);
  return { root, members: [g], height: s, update: (t) => { ball.rotation.y = t * 0.05; } };
}

function stars(spec, r, place) {
  const n = spec.count || 1;
  const c = color(spec.color, '#fff6d8');
  const s = (SIZE[spec.size] || 1);
  const shooting = spec.motion === 'fly';
  return manyOf(spec, r, place, (i, rr) => {
    const g = new THREE.Group();
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: sparkleTex(), color: c, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    sp.scale.setScalar((n > 1 ? 0.9 : 2.2) * s * (0.7 + 0.6 * rr()));
    g.add(sp, glowSprite(c, sp.scale.x * 1.8, 0.35));
    let tail = null;
    if (shooting) { tail = glowSprite(c, 1, 0.6); tail.scale.set(6 * s, 0.25 * s, 1); tail.position.x = -3 * s; g.add(tail); }
    const ph = rr() * 6.28;
    return { group: g, update: (t) => { sp.material.rotation = t * 0.3 + ph; sp.material.opacity = 0.7 + 0.3 * Math.sin(t * 2.5 + ph); } };
  }, { height: 0.5, randomYaw: false });
}

function rainbow(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 9;
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uOpacity: { value: 0.55 }, uR0: { value: s * 0.82 }, uR1: { value: s } },
    vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: `uniform float uOpacity, uR0, uR1; varying vec3 vP;
      vec3 hue(float h){ return clamp(abs(mod(h*6.0+vec3(0.,4.,2.),6.)-3.)-1.,0.,1.); }
      void main(){ float k = (length(vP.xy) - uR0) / (uR1 - uR0); float e = smoothstep(0.0,0.15,k)*smoothstep(1.0,0.85,k);
        gl_FragColor = vec4(hue(0.82 - k*0.82), uOpacity * e); }`,
  });
  mat.userData.fadeBase = 0.55;
  Object.defineProperty(mat, 'opacity', { get() { return mat.uniforms.uOpacity.value; }, set(v) { mat.uniforms.uOpacity.value = v; } });
  const g = new THREE.Mesh(new THREE.RingGeometry(s * 0.82, s, 96, 1, 0, Math.PI), mat);
  const p = place(0); g.position.set(p.x, -1, p.z);
  const root = new THREE.Group(); root.add(g);
  return { root, members: [], height: s };
}

// ------------------------------------------------------------------ buildings & passages
function windowMat(c = '#ffd98a') { return solid(c, { glow: 1.8, emissive: c }); }

function house(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 1.6;
  const wall = color(spec.color, '#9a7b62');
  return manyOf(spec, r, place, (i) => {
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    const wm = solid(wall, { glow: 0.12, roughness: 0.95 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1, 1.1), wm); body.position.y = 0.5;
    const roof = new THREE.Mesh(new THREE.ConeGeometry(1.15, 0.8, 4), solid('#4a2f2a', { glow: 0.1, flat: true })); roof.position.y = 1.4; roof.rotation.y = Math.PI / 4;
    const win = windowMat();
    const w1 = new THREE.Mesh(new THREE.PlaneGeometry(0.28, 0.26), win); w1.position.set(-0.35, 0.6, 0.56);
    const w2 = w1.clone(); w2.position.x = 0.35;
    const door = new THREE.Mesh(new THREE.PlaneGeometry(0.28, 0.5), solid('#3a2416', { glow: 0.2 })); door.position.set(0, 0.25, 0.56);
    const chim = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.5, 0.16), wm); chim.position.set(0.35, 1.55, -0.1);
    b.add(body, roof, w1, w2, door, chim);
    const light = new THREE.PointLight('#ffcf7a', 1.5, 6 * s, 1.5); light.position.set(0, 0.6 * s, 1.0 * s); g.add(light);
    const smoke = new Particles(26, { size: 0.5 * s, colorA: '#cfd6e0', twinkle: 0, opacity: 0.25, additive: false, seed: 13 });
    const life = new Float32Array(26).map((_, k) => k / 26); g.add(smoke);
    return { group: g, tint: [wm], update: (t, dt) => {
      for (let k = 0; k < 26; k++) { life[k] = (life[k] + dt * 0.12) % 1; const L = life[k]; smoke.setBase(k, (0.35 + Math.sin(L * 6 + k) * 0.15 + L * 0.6) * s, (1.8 + L * 2.2) * s, -0.1 * s); }
      smoke.touch(); smoke.tick(t);
    } };
  }, { height: 1.8 * s, randomYaw: false });
}

function castle(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 2.4;
  const stone = color(spec.color, '#8c8fa3');
  return manyOf(spec, r, place, () => {
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    const sm = solid(stone, { glow: 0.12, roughness: 1, flat: true });
    const roofM = solid('#3d3a6b', { glow: 0.2, flat: true });
    const keep = new THREE.Mesh(new THREE.BoxGeometry(1.4, 2.2, 1.4), sm); keep.position.y = 1.1; b.add(keep);
    const kr = new THREE.Mesh(new THREE.ConeGeometry(1.1, 1.2, 4), roofM); kr.position.y = 2.8; kr.rotation.y = Math.PI / 4; b.add(kr);
    for (const [x, z] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
      const tw = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.36, 2.6, 10), sm); tw.position.set(x * 1.25, 1.3, z * 1.25); b.add(tw);
      const tr = new THREE.Mesh(new THREE.ConeGeometry(0.45, 0.9, 10), roofM); tr.position.set(x * 1.25, 3.05, z * 1.25); b.add(tr);
      const w = new THREE.Mesh(new THREE.PlaneGeometry(0.1, 0.22), windowMat()); w.position.set(x * 1.25, 2.0, z * 1.25 + 0.34 * Math.sign(z)); if (z < 0) w.rotation.y = Math.PI; b.add(w);
    }
    for (const [x, z, w, d] of [[0, 1.25, 2.5, 0.25], [0, -1.25, 2.5, 0.25], [1.25, 0, 0.25, 2.5], [-1.25, 0, 0.25, 2.5]]) {
      const wl = new THREE.Mesh(new THREE.BoxGeometry(w, 1.1, d), sm); wl.position.set(x, 0.55, z); b.add(wl);
    }
    const gate = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.7), windowMat('#ffb86b')); gate.position.set(0, 0.35, 1.39); b.add(gate);
    g.add(glowSprite('#ffd27a', 5 * s, 0.12)); g.children[1].position.y = 1.5 * s;
    return { group: g, tint: [sm] };
  }, { height: 3.4 * s, randomYaw: false });
}

function portal(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 1.5;
  const c = color(spec.color, '#8f7bff');
  const door = has(spec.label, 'door', 'gate', 'דלת', 'שער');
  return manyOf(spec, r, place, () => {
    const g = new THREE.Group();
    const swirl = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      uniforms: { uTime: { value: 0 }, uColor: { value: c }, uOpacity: { value: 0.9 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: `uniform float uTime, uOpacity; uniform vec3 uColor; varying vec2 vUv;
        void main(){ vec2 p = vUv * 2.0 - 1.0; float r = length(p); float a = atan(p.y, p.x);
          float sw = sin(a * 5.0 + r * 12.0 - uTime * 2.5) * 0.5 + 0.5;
          float m = smoothstep(1.0, 0.2, r);
          gl_FragColor = vec4(uColor * (0.6 + sw) + vec3(1.0) * pow(1.0 - r, 4.0), m * uOpacity * (0.45 + 0.55 * sw)); }`,
    });
    swirl.userData.fadeBase = 0.9;
    Object.defineProperty(swirl, 'opacity', { get() { return swirl.uniforms.uOpacity.value; }, set(v) { swirl.uniforms.uOpacity.value = v; } });
    let frame;
    if (door) {
      frame = new THREE.Group(); const fm = solid('#c9b48a', { glow: 0.6 });
      for (const [x, y, w, h] of [[-0.55, 1, 0.12, 2], [0.55, 1, 0.12, 2], [0, 2.02, 1.22, 0.14]]) { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.14), fm); m.position.set(x, y, 0); frame.add(m); }
      const disc = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 1.95), swirl); disc.position.y = 1.0; frame.add(disc);
    } else {
      frame = new THREE.Group();
      const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.07, 10, 64), solid(c, { glow: 1.6 })); frame.add(ring);
      const disc = new THREE.Mesh(new THREE.CircleGeometry(0.97, 48), swirl); frame.add(disc);
      frame.position.y = 1.15;
    }
    frame.scale.setScalar(s); g.add(frame);
    g.add(glowSprite(c, 3.8 * s, 0.4)); g.children[1].position.y = 1.1 * s;
    const light = new THREE.PointLight(c, 2.5, 8 * s, 1.4); light.position.y = 1.1 * s; g.add(light);
    return { group: g, update: (t) => { swirl.uniforms.uTime.value = t; } };
  }, { height: 2.2 * s });
}

// ------------------------------------------------------------------ vehicles
function spaceship(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 1.3;
  const c = color(spec.color, '#dfe6f0');
  return manyOf(spec, r, place, (i) => {
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    const prof = [[0, 0], [0.18, 0.02], [0.26, 0.25], [0.28, 0.8], [0.22, 1.25], [0.12, 1.55], [0, 1.72]].map(([x, y]) => new THREE.Vector2(x, y));
    const hull = solid(c, { glow: 0.12, roughness: 0.35, metalness: 0.6 });
    const body = new THREE.Mesh(new THREE.LatheGeometry(prof, 24), hull);
    const finM = solid(c.clone().offsetHSL(0, 0.2, -0.25), { glow: 0.3, flat: true });
    for (let k = 0; k < 3; k++) {
      const fg = new THREE.BufferGeometry().setFromPoints([v3(0.22, 0.05, 0), v3(0.55, -0.1, 0), v3(0.25, 0.55, 0)]);
      fg.computeVertexNormals();
      const fin = new THREE.Mesh(fg, finM); fin.material.side = THREE.DoubleSide; fin.rotation.y = k * 2.094; b.add(fin);
    }
    const port = new THREE.Mesh(new THREE.CircleGeometry(0.1, 20), windowMat('#9fe3ff')); port.position.set(0, 1.0, 0.272); port.rotation.x = -0.05;
    const flame = glowSprite('#ffb05a', 0.9, 0.9); flame.position.y = -0.2;
    b.add(body, port, flame);
    b.rotation.x = Math.PI / 2 * 0.0;
    const exhaust = new Particles(50, { size: 0.35 * s, colorA: '#fff0b0', colorB: '#ff6a2a', twinkle: 0.3, seed: 21 });
    const life = new Float32Array(50).map((_, k) => k / 50);
    g.add(exhaust);
    const ph = i;
    return { group: g, tint: [hull], update: (t, dt, env) => {
      const thrust = env && env.moving ? 1 : 0.35;
      for (let k = 0; k < 50; k++) { life[k] = (life[k] + dt * 1.6) % 1; const L = life[k]; exhaust.setBase(k, Math.sin(k * 7.1 + t) * 0.06 * s * L * 3, (-0.25 - L * 1.4 * thrust) * s, Math.cos(k * 3.3 + t) * 0.06 * s * L * 3); }
      exhaust.touch(); exhaust.tick(t);
      flame.scale.setScalar((0.7 + 0.3 * Math.sin(t * 22 + ph)) * thrust * 1.2 * s);
    } };
  }, { height: 1.8 * s, randomYaw: false });
}

function boat(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 1.4;
  const c = color(spec.color, '#8a5a3a');
  return manyOf(spec, r, place, () => {
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    const hm = solid(c, { glow: 0.15 });
    const hullG = new THREE.CylinderGeometry(0.5, 0.5, 1.8, 16, 1, false, Math.PI / 2, Math.PI); hullG.rotateX(Math.PI / 2); hullG.scale(1, 0.55, 1);
    const hull = new THREE.Mesh(hullG, hm); hull.material.side = THREE.DoubleSide; hull.position.y = 0.28;
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.03, 1.8, 6), solid('#5a3a22')); mast.position.y = 1.15;
    const sail = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 1.2, 6, 6), solid('#f2ead8', { glow: 0.35, side: THREE.DoubleSide })); sail.position.set(0.0, 1.25, 0.05); sail.rotation.y = Math.PI / 2;
    const lamp = glowSprite('#ffd27a', 0.5, 0.9); lamp.position.set(0, 0.7, 0.85);
    b.add(hull, mast, sail, lamp);
    return { group: g, tint: [hm] };
  }, { height: 2.2 * s });
}

// ------------------------------------------------------------------ small magical objects
function floating(spec, r, place, build, opts = {}) {
  const s = (SIZE[spec.size] || 1) * (opts.scale || 1);
  const c = color(spec.color, opts.color || '#ffd56a');
  return manyOf(spec, r, place, (i, rr) => {
    const g = new THREE.Group(); const b = new THREE.Group(); b.scale.setScalar(s); g.add(b);
    const made = build(b, c, rr, i) || {};
    if (opts.glow !== 0) g.add(glowSprite(opts.glowColor || c, (opts.glowSize || 1.4) * s, opts.glow ?? 0.35));
    if (opts.sparkles) { const sp = flora.sparkles(opts.sparkles, 0.6 * s, c, i + 3); g.add(sp); made.sp = sp; }
    return { group: g, tint: made.tint, update: (t, dt) => { if (made.sp) made.sp.tick(t); if (made.update) made.update(t, dt); } };
  }, { height: (opts.height || 0.5) * s });
}

function cheeseTexture() {
  return noiseTex('cheese', 256, (g, w) => {
    const r = rng(77); g.fillStyle = '#f6c548'; g.fillRect(0, 0, w, w);
    for (let k = 0; k < 2500; k++) { g.fillStyle = `rgba(255,${200 + r() * 55 | 0},90,${0.15 * r()})`; g.fillRect(r() * w, r() * w, 2, 2); }
    for (let k = 0; k < 28; k++) {
      const x = r() * w, y = r() * w, rad = 4 + Math.pow(r(), 2) * 16;
      const gr = g.createRadialGradient(x - rad * 0.2, y - rad * 0.2, 0, x, y, rad);
      gr.addColorStop(0, '#b98516'); gr.addColorStop(0.8, '#d49b25'); gr.addColorStop(1, 'rgba(246,197,72,0)');
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, rad, 0, 6.28); g.fill();
    }
  });
}

function cheese(spec, r, place) {
  return floating(spec, r, place, (b, c) => {
    const shape = new THREE.Shape(); shape.moveTo(0, 0); shape.lineTo(0.9, 0); shape.lineTo(0, 0.55); shape.closePath();
    const geo = new THREE.ExtrudeGeometry(shape, { depth: 0.55, bevelEnabled: true, bevelSize: 0.03, bevelThickness: 0.03, bevelSegments: 3 });
    geo.center();
    const tex = cheeseTexture(); tex.repeat.set(1.6, 1.6);
    const mat = new THREE.MeshStandardMaterial({ map: tex, color: c, emissive: '#ffb020', emissiveIntensity: 0.32, roughness: 0.6 });
    const m = new THREE.Mesh(geo, mat); m.rotation.set(0.25, 0.6, 0); b.add(m);
    return { tint: [] };
  }, { color: '#ffe08a', glowColor: '#ffcf4a', glowSize: 2.6, glow: 0.5, sparkles: 26, scale: 1.0, height: 0.6 });
}

function food(spec, r, place) {
  const cake = has(spec.label, 'cake', 'עוגה'), bread = has(spec.label, 'bread', 'loaf', 'לחם');
  return floating(spec, r, place, (b, c) => {
    if (cake) {
      const m1 = solid(c, { glow: 0.3 }), cream = solid('#fff3e6', { glow: 0.35 });
      const t1 = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 0.25, 24), m1); const t2 = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.22, 24), m1); t2.position.y = 0.23;
      const ic = new THREE.Mesh(new THREE.CylinderGeometry(0.29, 0.29, 0.04, 24), cream); ic.position.y = 0.35;
      const candle = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.18, 6), solid('#ff9ad5', { glow: 0.5 })); candle.position.y = 0.46;
      const fl = glowSprite('#ffcf6a', 0.18, 1); fl.position.y = 0.6;
      b.add(t1, t2, ic, candle, fl); return { tint: [m1] };
    }
    if (bread) { const m = solid(c, { glow: 0.2 }); const l = new THREE.Mesh(new THREE.CapsuleGeometry(0.18, 0.4, 6, 12), m); l.rotation.z = Math.PI / 2; b.add(l); return { tint: [m] }; }
    const m = solid(c, { glow: 0.4, roughness: 0.4 });
    const ap = new THREE.Mesh(new THREE.SphereGeometry(0.25, 20, 16), m); ap.scale.y = 0.9;
    const st = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.016, 0.14, 5), solid('#5a3a22')); st.position.y = 0.26;
    const lf = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 6), solid('#4caf50', { glow: 0.3 })); lf.scale.set(1.6, 0.3, 0.8); lf.position.set(0.07, 0.29, 0);
    b.add(ap, st, lf); return { tint: [m] };
  }, { color: cake ? '#f7a8c4' : bread ? '#d9a35a' : '#e53935', glow: 0.25, scale: 1.0, height: 0.5 });
}

function gem(spec, r, place) {
  const coins = has(spec.label, 'coin', 'treasure', 'gold', 'מטבע', 'אוצר', 'זהב');
  return floating(spec, r, place, (b, c) => {
    if (coins) {
      const gm = solid('#ffcf4a', { glow: 0.6, metalness: 0.9, roughness: 0.25 });
      for (let k = 0; k < 9; k++) { const cn = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.025, 18), gm); cn.position.set(Math.sin(k * 2.4) * 0.15, k * 0.03, Math.cos(k * 2.4) * 0.15); cn.rotation.set(Math.sin(k) * 0.4, 0, Math.cos(k) * 0.4); b.add(cn); }
      return { tint: [] };
    }
    const m = solid(c, { glow: 0.9, metalness: 0.5, roughness: 0.1, flat: true });
    const top = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.26, 0.12, 8), m); top.position.y = 0.06;
    const bot = new THREE.Mesh(new THREE.ConeGeometry(0.26, 0.38, 8), m); bot.rotation.x = Math.PI; bot.position.y = -0.19;
    b.add(top, bot); return { tint: [m] };
  }, { color: coins ? '#ffcf4a' : '#4fd2ff', glow: 0.45, sparkles: 18, height: 0.4 });
}

function crown(spec, r, place) {
  return floating(spec, r, place, (b, c) => {
    const m = solid(c, { glow: 0.6, metalness: 0.9, roughness: 0.25 });
    const band = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.27, 0.16, 32, 1, true), m); band.material.side = THREE.DoubleSide; b.add(band);
    const gemM = solid('#e0115f', { glow: 1.2 });
    for (let k = 0; k < 7; k++) {
      const a = k / 7 * 6.283;
      const sp = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.2, 6), m); sp.position.set(Math.cos(a) * 0.29, 0.17, Math.sin(a) * 0.29); b.add(sp);
      const gm = new THREE.Mesh(new THREE.SphereGeometry(0.025, 6, 4), gemM); gm.position.set(Math.cos(a) * 0.31, 0.27, Math.sin(a) * 0.31); b.add(gm);
    }
    return { tint: [m] };
  }, { color: '#ffcf4a', glow: 0.45, sparkles: 14, scale: 1.2, height: 0.4 });
}

function key(spec, r, place) {
  return floating(spec, r, place, (b, c) => {
    const m = solid(c, { glow: 0.55, metalness: 0.9, roughness: 0.3 });
    const bow = new THREE.Mesh(new THREE.TorusGeometry(0.12, 0.03, 8, 24), m); bow.position.y = 0.32;
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.5, 8), m);
    const t1 = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.04, 0.03), m); t1.position.set(0.06, -0.18, 0);
    const t2 = t1.clone(); t2.position.y = -0.1; t2.scale.x = 0.7;
    b.add(bow, shaft, t1, t2); return { tint: [m] };
  }, { color: '#ffcf4a', glow: 0.35, sparkles: 10, scale: 1.3, height: 0.5 });
}

function book(spec, r, place) {
  return floating(spec, r, place, (b, c) => {
    const cover = solid(c, { glow: 0.3 }); const pages = solid('#fff6e0', { glow: 0.9, emissive: '#fff0c8' });
    for (const x of [1, -1]) {
      const half = new THREE.Group(); half.rotation.z = x * 0.28;
      const cv = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.02, 0.55), cover); cv.position.x = x * 0.21;
      const pg = new THREE.Mesh(new THREE.BoxGeometry(0.39, 0.05, 0.52), pages); pg.position.set(x * 0.2, 0.035, 0);
      half.add(cv, pg); b.add(half);
    }
    b.rotation.x = 0.5;
    return { tint: [cover] };
  }, { color: '#7a2f3a', glowColor: '#fff0c8', glow: 0.5, sparkles: 22, scale: 1.2, height: 0.4 });
}

function sword(spec, r, place) {
  const wand = has(spec.label, 'wand', 'staff', 'שרביט', 'מטה');
  return floating(spec, r, place, (b, c) => {
    if (wand) {
      const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.025, 0.9, 6), solid('#5a3a22', { glow: 0.2 }));
      const tip = glowSprite(c, 0.45, 1); tip.position.y = 0.5;
      b.add(stick, tip); b.rotation.z = 0.5; return { tint: [] };
    }
    const blade = solid(c, { glow: 1.0, metalness: 0.9, roughness: 0.15 });
    const bl = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.95, 0.015), blade); bl.position.y = 0.5;
    const tip = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.14, 4), blade); tip.position.y = 1.04; tip.rotation.y = Math.PI / 4; tip.scale.z = 0.3;
    const guard = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.04, 0.05), solid('#c9a24a', { glow: 0.4, metalness: 0.8 }));
    const hilt = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.22, 8), solid('#3a2416')); hilt.position.y = -0.12;
    const pom = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 6), solid('#c9a24a', { glow: 0.5, metalness: 0.8 })); pom.position.y = -0.25;
    b.add(bl, tip, guard, hilt, pom); b.rotation.z = 0.2; return { tint: [blade] };
  }, { color: wand ? '#ffe9a8' : '#bfe3ff', glow: 0.35, sparkles: 12, scale: 1.1, height: 1.0 });
}

function orb(spec, r, place) {
  return floating(spec, r, place, (b, c) => {
    const shell = new THREE.Mesh(new THREE.SphereGeometry(0.32, 32, 24), new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: 0.6, transparent: true, opacity: 0.55, roughness: 0.05, metalness: 0.1 }));
    const core = glowSprite('#ffffff', 0.35, 1);
    const swirl = flora.sparkles(30, 0.22, c, 9);
    b.add(shell, core, swirl);
    const light = new THREE.PointLight(c, 1.8, 5, 1.5); b.add(light);
    return { tint: [], update: (t) => { swirl.rotation.y = t * 0.9; swirl.rotation.x = t * 0.4; swirl.tick(t); core.scale.setScalar(0.3 + 0.08 * Math.sin(t * 3)); } };
  }, { color: '#9fd8ff', glowSize: 2.2, glow: 0.5, height: 0.4 });
}

function lantern(spec, r, place) {
  const paper = (spec.count || 1) > 1 || has(spec.label, 'sky lantern', 'paper', 'פנס נייר');
  return floating(spec, r, place, (b, c) => {
    if (paper) {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.12, 0.34, 12, 1, true), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
      b.add(m, glowSprite('#ffe6b0', 0.3, 1)); return { tint: [] };
    }
    const fm = solid('#2f2a24', { glow: 0.1, metalness: 0.6 });
    const top = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.12, 6), fm); top.position.y = 0.22;
    const bot = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.12, 0.04, 6), fm); bot.position.y = -0.18;
    const glass = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.36, 6, 1, true), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.35, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
    const flame = glowSprite('#fff1c2', 0.25, 1);
    b.add(top, bot, glass, flame);
    const light = new THREE.PointLight(c, 2, 6, 1.6); b.add(light);
    return { tint: [], update: (t) => { light.intensity = 2 * (0.85 + 0.15 * Math.sin(t * 9) * Math.sin(t * 5.3)); } };
  }, { color: '#ffb35a', glowSize: 1.4, glow: 0.45, height: 0.5 });
}

// ------------------------------------------------------------------ the unknown
function other(spec, r, place) {
  return floating(spec, r, place, (b, c, rr) => {
    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(0.3, 1), new THREE.MeshBasicMaterial({ color: c, wireframe: true, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false }));
    const inner = glowSprite(c, 0.7, 0.9);
    const rings = [0, 1].map((k) => { const m = new THREE.Mesh(new THREE.TorusGeometry(0.45 + k * 0.1, 0.008, 6, 64), glowMat(c, 0.7)); m.rotation.x = k ? 1.2 : 0.3; b.add(m); return m; });
    b.add(core, inner);
    return { tint: [], update: (t) => { core.rotation.y = t * 0.4; core.rotation.x = t * 0.25; rings[0].rotation.y = t * 0.6; rings[1].rotation.z = t * 0.5; } };
  }, { color: '#cbb8ff', glowSize: 2, glow: 0.4, sparkles: 16, height: 0.6 });
}

// The nameless spark the dreamer starts as.
export function wisp(spec, r, place) {
  const c = color(spec.color, '#d8ecff');
  const g = new THREE.Group(); if (place) g.position.copy(place(0));
  const core = glowSprite('#ffffff', 0.35, 1), halo = glowSprite(c, 1.6, 0.7), outer = glowSprite(c, 4.2, 0.18);
  const motes = new Particles(36, { size: 0.08, colorA: '#ffffff', colorB: c, twinkle: 0.8, seed: 41 });
  for (let i = 0; i < 36; i++) { const a = i * 2.39996, rr = 0.35 + (i % 7) * 0.08; motes.setBase(i, Math.cos(a) * rr, Math.sin(i * 1.7) * 0.3, Math.sin(a) * rr); }
  g.add(outer, halo, core, motes);
  const light = new THREE.PointLight(c, 2.2, 7, 1.5); g.add(light);
  const root = new THREE.Group(); root.add(g);
  return {
    root, members: [g], height: 0.4, core, halo,
    update: (t, dt, env) => {
      const v = env ? env.voice : 0;
      const breath = 0.5 + 0.5 * Math.sin(t * 1.3);
      core.scale.setScalar(0.32 + 0.06 * breath + 0.35 * v);
      halo.scale.setScalar(1.4 + 0.3 * breath + 1.2 * v);
      halo.material.opacity = 0.55 + 0.25 * breath + 0.2 * v;
      light.intensity = 1.8 + 1.2 * breath + 3 * v;
      motes.rotation.y = t * 0.35; motes.tick(t);
    },
  };
}

export const BUILDERS = {
  fairy, butterfly, fireflies, bird, fish,
  cat: quadruped('cat'), dog: quadruped('dog'), wolf: quadruped('wolf'), horse: quadruped('horse'),
  dragon, person, ghost,
  tree: (sp, r, pl, o) => flora.trees(sp, r, pl, o), flower: flora.flowers, mushroom: flora.mushrooms,
  rock: (sp, r, pl) => flora.rocks(sp, r, pl), crystal: flora.crystals, cloud: flora.clouds,
  fire: flora.fire, water: flora.water,
  moon: moonThing, sun, planet, star: stars, rainbow,
  house, castle, portal, spaceship, boat, lantern,
  cheese, food, gem, crown, key, book, sword, orb, other,
};

// Rough category for placement.
export const GROUND_KINDS = new Set(['tree', 'flower', 'mushroom', 'rock', 'crystal', 'fire', 'water', 'house', 'castle', 'portal', 'cat', 'dog', 'wolf', 'horse', 'person']);
export const SKY_KINDS = new Set(['moon', 'sun', 'planet', 'star', 'rainbow', 'cloud']);
export { softDot };
