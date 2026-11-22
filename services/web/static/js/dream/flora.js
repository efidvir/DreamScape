// Nature: trees, flowers, mushrooms, rocks, crystals, clouds, water, fire.
// Every builder takes (spec, r, place) and returns { root, members, height, update }.
import * as THREE from 'three';
import { color, solid, glowSprite, mistTex, Particles, SIZE, dreamify, sparkleTex } from './kit.js';
import { merge, moved, roughen, instanced } from './geo.js';

const has = (label, ...words) => words.some((w) => (label || '').toLowerCase().includes(w));

// ------------------------------------------------------------------ trees
const _treeGeo = {};
function treeGeometry(style) {
  if (_treeGeo[style]) return _treeGeo[style];
  let trunk, crown;
  if (style === 'conifer') {
    trunk = new THREE.CylinderGeometry(0.09, 0.16, 1.4, 6); trunk.translate(0, 0.7, 0);
    crown = merge([
      moved(new THREE.ConeGeometry(1.15, 1.9, 8), 0, 1.6, 0),
      moved(new THREE.ConeGeometry(0.9, 1.6, 8), 0, 2.5, 0, 0, 0.4),
      moved(new THREE.ConeGeometry(0.6, 1.3, 8), 0, 3.3, 0, 0, 0.8),
    ]);
  } else if (style === 'bare') {
    trunk = merge([
      moved(new THREE.CylinderGeometry(0.07, 0.18, 2.8, 6), 0, 1.4, 0),
      moved(new THREE.CylinderGeometry(0.03, 0.07, 1.4, 5), 0.45, 2.5, 0, 0, 0, -0.8),
      moved(new THREE.CylinderGeometry(0.03, 0.06, 1.2, 5), -0.4, 2.2, 0.1, 0.2, 0, 0.9),
      moved(new THREE.CylinderGeometry(0.02, 0.05, 1.0, 5), 0.1, 3.0, -0.35, -0.9, 0, 0.1),
    ]);
    crown = new THREE.SphereGeometry(0.01, 3, 2);
  } else {
    trunk = new THREE.CylinderGeometry(0.1, 0.2, 1.8, 7); trunk.translate(0, 0.9, 0);
    crown = merge([
      roughen(moved(new THREE.IcosahedronGeometry(1.0, 1), 0, 2.5, 0), 0.18, 3),
      roughen(moved(new THREE.IcosahedronGeometry(0.75, 1), 0.65, 2.15, 0.2), 0.2, 5),
      roughen(moved(new THREE.IcosahedronGeometry(0.7, 1), -0.6, 2.25, -0.15), 0.2, 7),
      roughen(moved(new THREE.IcosahedronGeometry(0.6, 1), 0.1, 3.15, -0.1), 0.2, 9),
    ]);
  }
  _treeGeo[style] = { trunk, crown };
  return _treeGeo[style];
}

export function treeStyle(label, setting) {
  if (has(label, 'pine', 'fir', 'spruce', 'conifer', 'christmas', 'אורן')) return 'conifer';
  if (has(label, 'dead', 'bare', 'spooky', 'leafless', 'twisted')) return 'bare';
  if (has(label, 'oak', 'apple', 'cherry', 'willow', 'birch', 'blossom', 'maple')) return 'broad';
  return setting === 'snow' || setting === 'mountains' ? 'conifer' : null;
}

export function trees(spec, r, place, opts = {}) {
  const n = spec.count || 1;
  const s = (SIZE[spec.size] || 1) * 1.15;
  const leaf = color(spec.color, opts.leaf || '#1f4a2c');
  const bark = color(opts.bark || '#3b2a20');
  const fixed = treeStyle(spec.label, opts.setting);
  const byStyle = { conifer: [], broad: [], bare: [] };
  for (let i = 0; i < n; i++) {
    const style = fixed || (r() < 0.55 ? 'conifer' : 'broad');
    const k = s * (0.75 + 0.5 * r());
    byStyle[style].push({ p: place(i), ry: r() * 6.28, s: k, rz: (r() - 0.5) * 0.06 });
  }
  const root = new THREE.Group();
  const trunkMat = solid(bark, { glow: 0.05, roughness: 0.95 });
  const leafMat = solid(leaf, { glow: opts.glow ?? 0.1, roughness: 0.85, flat: true });
  for (const [style, list] of Object.entries(byStyle)) {
    if (!list.length) continue;
    const g = treeGeometry(style);
    root.add(instanced(g.trunk, trunkMat, list));
    if (style !== 'bare') root.add(instanced(g.crown, leafMat, list));
  }
  return { root, members: [], height: 4 * s, tintables: [leafMat] };
}

// ------------------------------------------------------------------ flowers & mushrooms
export function flowers(spec, r, place) {
  const n = spec.count || 1, s = (SIZE[spec.size] || 1) * 0.9;
  const head = color(spec.color, '#ff8fd0');
  const stems = [], heads = [];
  const headGeo = merge([
    ...[0, 1, 2, 3, 4].map((i) => moved(new THREE.SphereGeometry(0.09, 6, 4), Math.cos(i * 1.256) * 0.1, 0, Math.sin(i * 1.256) * 0.1, 0, 0, 0, 1).scale(1, 0.45, 1)),
    moved(new THREE.SphereGeometry(0.06, 6, 4), 0, 0.02, 0),
  ]);
  for (let i = 0; i < n; i++) {
    const p = place(i), k = s * (0.7 + 0.6 * r()), h = 0.45 * k;
    stems.push({ p: p.clone(), s: k, sy: k });
    heads.push({ p: p.clone().add(new THREE.Vector3(0, h + 0.02, 0)), s: k, rx: (r() - 0.5) * 0.5, ry: r() * 6.28 });
  }
  const stemGeo = new THREE.CylinderGeometry(0.012, 0.018, 0.45, 4); stemGeo.translate(0, 0.225, 0);
  const root = new THREE.Group();
  const headMat = solid(head, { glow: 0.45 });
  root.add(instanced(stemGeo, solid('#3e7a3a', { glow: 0.08 }), stems));
  root.add(instanced(headGeo, headMat, heads));
  return { root, members: [], height: 0.5 * s, tintables: [headMat] };
}

export function mushrooms(spec, r, place) {
  const n = spec.count || 1, s = (SIZE[spec.size] || 1) * 0.8;
  const cap = color(spec.color, '#c94fff');
  const stemGeo = new THREE.CylinderGeometry(0.05, 0.07, 0.3, 7); stemGeo.translate(0, 0.15, 0);
  const capGeo = new THREE.SphereGeometry(0.2, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2); capGeo.scale(1, 0.6, 1); capGeo.translate(0, 0.29, 0);
  const list = [];
  for (let i = 0; i < n; i++) list.push({ p: place(i), s: s * (0.6 + 0.8 * r()), ry: r() * 6.28, rz: (r() - 0.5) * 0.3 });
  const root = new THREE.Group();
  const capMat = solid(cap, { glow: 0.85 });
  root.add(instanced(stemGeo, solid('#e8e0cc', { glow: 0.2 }), list));
  root.add(instanced(capGeo, capMat, list));
  return { root, members: [], height: 0.4 * s, tintables: [capMat] };
}

// ------------------------------------------------------------------ rocks & crystals
export function rocks(spec, r, place, opts = {}) {
  const n = spec.count || 1, s = SIZE[spec.size] || 1;
  const geo = roughen(new THREE.DodecahedronGeometry(0.5, 1), 0.22, 11);
  const list = [];
  for (let i = 0; i < n; i++) {
    const k = s * (0.5 + r()); const p = place(i); p.y += 0.18 * k;
    list.push({ p, s: k, sy: k * (0.55 + 0.4 * r()), rx: r() * 3, ry: r() * 3 });
  }
  const mat = solid(color(spec.color, opts.color || '#6d6a72'), { glow: 0.06, flat: true, roughness: 1 });
  const root = new THREE.Group();
  root.add(instanced(geo, mat, list));
  return { root, members: [], height: 0.6 * s, tintables: [mat] };
}

export function crystals(spec, r, place) {
  const n = spec.count || 1, s = SIZE[spec.size] || 1;
  const c = color(spec.color, '#8fdcff');
  const shard = () => { const g = new THREE.OctahedronGeometry(0.22, 0); g.scale(0.55, 2.2, 0.55); g.translate(0, 0.45, 0); return g; };
  const cluster = merge([
    shard(), moved(shard(), 0.12, 0, 0.05, 0.1, 0, -0.45, 0.7), moved(shard(), -0.12, 0, -0.04, -0.1, 0, 0.5, 0.65),
    moved(shard(), 0.02, 0, -0.14, -0.5, 0, 0.05, 0.55), moved(shard(), -0.05, 0, 0.13, 0.45, 0, -0.1, 0.5),
  ]);
  const root = new THREE.Group(); const members = [];
  const mat = solid(c, { glow: 1.1, roughness: 0.15, metalness: 0.2 });
  mat.opacity = 0.9;
  for (let i = 0; i < n; i++) {
    const m = new THREE.Group();
    const mesh = new THREE.Mesh(cluster, mat); mesh.scale.setScalar(s * (0.7 + 0.6 * r()));
    m.add(mesh, glowSprite(c, 1.8 * s, 0.35));
    m.position.copy(place(i)); m.rotation.y = r() * 6.28;
    root.add(m); members.push(m);
  }
  return { root, members, height: 0.9 * s, tintables: [mat], groundMembers: true };
}

// ------------------------------------------------------------------ clouds
export function clouds(spec, r, place) {
  const n = spec.count || 1, s = (SIZE[spec.size] || 1) * 2.2;
  const c = color(spec.color, '#b7c2d6');
  const root = new THREE.Group(); const members = [];
  for (let i = 0; i < n; i++) {
    const m = new THREE.Group();
    for (let j = 0; j < 9; j++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: mistTex(j % 4), color: c, transparent: true,
        opacity: 0.55, depthWrite: false, fog: true }));
      sp.position.set((r() - 0.5) * 2.6 * s, (r() - 0.5) * 0.5 * s, (r() - 0.5) * 0.8 * s);
      sp.scale.setScalar(s * (1.3 + r()));
      sp.material.rotation = r() * 6.28;
      m.add(sp);
    }
    m.position.copy(place(i)); root.add(m); members.push(m);
  }
  return { root, members, height: s, tintables: [] };
}

// ------------------------------------------------------------------ water
export function water(spec, r, place) {
  const s = (SIZE[spec.size] || 1) * 3.2;
  const c = color(spec.color, '#3d7fb3');
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uTime: { value: 0 }, uColor: { value: c }, uOpacity: { value: 0.85 } },
    vertexShader: `varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform float uTime; uniform vec3 uColor; uniform float uOpacity; varying vec2 vUv; varying vec3 vW;
      void main(){
        vec2 p = vUv - 0.5; float d = length(p) * 2.0;
        float rings = sin(d * 38.0 - uTime * 1.6) * 0.5 + 0.5;
        float waves = sin(vW.x * 2.1 + uTime * 0.9) * sin(vW.z * 1.7 - uTime * 0.7);
        vec3 col = uColor * (0.55 + 0.25 * rings) + vec3(0.6, 0.8, 1.0) * pow(max(waves, 0.0), 6.0) * 0.9;
        float edge = 1.0 - smoothstep(0.82, 1.0, d);
        gl_FragColor = vec4(col, uOpacity * edge);
      }`,
  });
  mat.userData.fadeBase = 0.85;
  Object.defineProperty(mat, 'opacity', { get() { return mat.uniforms.uOpacity.value; }, set(v) { mat.uniforms.uOpacity.value = v; } });
  const mesh = new THREE.Mesh(new THREE.CircleGeometry(s, 48), mat);
  mesh.rotation.x = -Math.PI / 2;
  const p = place(0); mesh.position.set(p.x, p.y + 0.03, p.z);
  const root = new THREE.Group(); root.add(mesh);
  return { root, members: [], height: 0.1, update: (t) => { mat.uniforms.uTime.value = t; }, tintables: [] };
}

// ------------------------------------------------------------------ fire
export function fire(spec, r, place) {
  const s = SIZE[spec.size] || 1;
  const c = color(spec.color, '#ff8a2a');
  const root = new THREE.Group();
  const m = new THREE.Group(); m.position.copy(place(0));
  const log = new THREE.CylinderGeometry(0.07, 0.07, 0.9, 6); log.rotateZ(Math.PI / 2);
  const logMat = solid('#3a2416', { glow: 0.15, emissive: '#ff5a1a' });
  for (let k = 0; k < 3; k++) { const l = new THREE.Mesh(log, logMat); l.rotation.y = k * 1.05; l.position.y = 0.06; l.scale.setScalar(s); m.add(l); }
  const flames = new Particles(70, { size: 0.32 * s, colorA: '#fff1a8', colorB: c, twinkle: 0.2, seed: 7 });
  const life = new Float32Array(70).map(() => r());
  m.add(flames);
  const light = new THREE.PointLight(c, 4 * s, 9 * s, 1.6); light.position.y = 0.6 * s; m.add(light);
  m.add(glowSprite(c, 2.6 * s, 0.45));
  root.add(m);
  const update = (t, dt) => {
    for (let i = 0; i < 70; i++) {
      life[i] += dt * (0.9 + (i % 5) * 0.12);
      if (life[i] > 1) life[i] -= 1;
      const L = life[i], a = i * 2.39996;
      flames.setBase(i, Math.cos(a) * 0.18 * s * (1 - L), L * 1.3 * s, Math.sin(a) * 0.18 * s * (1 - L));
    }
    flames.touch(); flames.tick(t);
    light.intensity = 4 * s * (0.75 + 0.35 * Math.sin(t * 13) * Math.sin(t * 7.3));
  };
  return { root, members: [], height: 1.0 * s, update, tintables: [] };
}

// Fallback sparkles used by gems, cheese and other magical objects.
export function sparkles(n, radius, c, seed = 3) {
  const p = new Particles(n, { size: 0.14, colorA: '#ffffff', colorB: c, twinkle: 1, seed, map: sparkleTex() });
  const r = (s => () => (s = (s * 16807) % 2147483647) / 2147483647)(seed * 7919 + 1);
  for (let i = 0; i < n; i++) {
    const a = r() * 6.283, b = Math.acos(2 * r() - 1), rr = radius * (0.6 + 0.6 * r());
    p.setBase(i, rr * Math.sin(b) * Math.cos(a), rr * Math.cos(b), rr * Math.sin(b) * Math.sin(a));
  }
  return p;
}

export { dreamify };
