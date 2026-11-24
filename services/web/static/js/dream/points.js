// Painted dreaming: every solid shape is re-drawn as thousands of brush strokes sampled
// from its surface, like Gaussian splats with a painter's hand: each one soft, elongated,
// turned along the surface and streaked with bristles, overlapping its neighbours so the
// colours blend. Strokes gather out of the mist when a thing appears, drift apart when the
// dream turns hazy, and melt into round bokeh by their distance from the focus.
import * as THREE from 'three';
import { dreamClock, noise3 } from './kit.js';

// Shared by every particle material; the renderer and world update them each frame.
export const dreamU = {
  uTime: dreamClock,
  uFocus: { value: 7 }, uAperture: { value: 0.6 }, uHaze: { value: 0.6 },
  uLight: { value: new THREE.Color(0.6, 0.65, 0.8) },
  uFogColor: { value: new THREE.Color(0, 0, 0) }, uFogDensity: { value: 0.05 },
  uViewH: { value: 720 }, uAspect: { value: 1.6 },
  uStroke: { value: 2.6 },          // how much longer than wide a brush stroke is
  uBristle: { value: 0.55 },        // how much its bristle streaks show
};

const VS = `
uniform float uTime, uFocus, uAperture, uHaze, uViewH, uFogDensity, uAspect, uStroke;
uniform float uSize, uForm, uWander, uTintAmt, uSelfLit, uFogScale, uDof;
uniform vec3 uLight, uFogColor, uTint;
attribute vec3 aScatter, aNormal, aColor, aGlow, aDir;
attribute float aSize, aSeed, aFade;
varying vec3 vCol; varying float vA; varying vec2 vRot; varying float vStretch, vSeed;
float dh(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float dn(vec3 x){ vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(dh(i), dh(i + vec3(1,0,0)), f.x), mix(dh(i + vec3(0,1,0)), dh(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(dh(i + vec3(0,0,1)), dh(i + vec3(1,0,1)), f.x), mix(dh(i + vec3(0,1,1)), dh(i + vec3(1,1,1)), f.x), f.y), f.z); }
void main(){
  // gather: each particle leaves the mist at its own moment
  float k = clamp((uForm - aSeed * 0.45) / 0.55, 0.0, 1.0); k = k * k * (3.0 - 2.0 * k);
  vec4 wp = modelMatrix * vec4(mix(aScatter, position, k), 1.0);
  // drift: the haze of the dream loosens every shape
  // most particles hold their shape; a few drift far, like dust lifting off
  float stray = 0.35 + 2.8 * aSeed * aSeed * aSeed * aSeed;
  float amp = (0.02 + 0.22 * uHaze * uWander) * stray + (1.0 - k) * 0.5;
  vec3 q = wp.xyz * 0.8 + vec3(uTime * 0.13, -uTime * 0.09, uTime * 0.07) + aSeed * 9.0;
  wp.xyz += (vec3(dn(q), dn(q + 11.3), dn(q + 27.1)) - 0.5) * 2.0 * amp;
  vec4 mv = viewMatrix * wp;
  float depth = max(-mv.z, 0.05);
  // depth of field, one particle at a time
  float coc = clamp(abs(depth - uFocus) * uAperture * uDof / max(depth, 1.0), 0.0, 1.2);
  float tw = 0.82 + 0.18 * sin(uTime * (0.6 + aSeed * 2.1) + aSeed * 37.0);
  float px = uSize * aSize * projectionMatrix[1][1] * 0.5 * uViewH / depth;
  gl_Position = projectionMatrix * mv;
  // the stroke: its direction on screen, and how long it looks (short when seen end-on,
  // round when out of focus, like bokeh)
  vec3 wd = mat3(modelMatrix) * aDir; float wl = length(wd);
  vec2 dir2 = vec2(1.0, 0.0); float stretch = 1.0;
  if (wl > 1e-4) {
    vec4 c2 = projectionMatrix * (viewMatrix * vec4(wp.xyz + wd / wl * 0.25, 1.0));
    vec2 sd = c2.xy / max(c2.w, 1e-3) - gl_Position.xy / max(gl_Position.w, 1e-3);
    sd.x *= uAspect;
    float L = length(sd);
    float face = clamp(L / max(0.25 * projectionMatrix[1][1] / depth, 1e-5), 0.0, 1.0);
    if (L > 1e-6) dir2 = sd / L;
    stretch = mix(1.0, uStroke * (0.7 + 0.6 * fract(aSeed * 3.7)), face * (1.0 - clamp(coc, 0.0, 1.0)));
  }
  vRot = dir2; vStretch = stretch; vSeed = aSeed;
  gl_PointSize = clamp(px * (1.0 + coc * 3.5) * tw * sqrt(stretch) * 1.2, 1.0, 96.0);
  vec3 n = normalize(mat3(modelMatrix) * aNormal + 1e-4);
  float shade = 0.5 + 0.5 * clamp(n.y * 0.65 + 0.35, 0.0, 1.0);
  // dream light: dark colours are lifted so every shape glows faintly out of the mist
  vec3 albedo = pow(max(aColor, vec3(0.0)), vec3(0.62)) * (0.78 + 0.44 * fract(aSeed * 7.13));
  float lum = dot(albedo, vec3(0.299, 0.587, 0.114));
  albedo = mix(albedo, uTint * (0.3 + lum * 1.1), uTintAmt);
  // a painting brings its own light: its colours are kept as painted
  vec3 own = aColor * (0.9 + 0.2 * fract(aSeed * 7.13));
  own = mix(own, uTint * (0.25 + dot(own, vec3(0.299, 0.587, 0.114)) * 1.2), uTintAmt);
  float fog = 1.0 - exp(-uFogDensity * uFogDensity * depth * depth * uFogScale);
  float nearFade = smoothstep(1.2, 4.2, depth);
  float sparkle = step(0.965, fract(aSeed * 13.7)) * (0.5 + 0.5 * sin(uTime * 3.1 + aSeed * 60.0));
#ifdef GLOW
  vCol = aGlow * 0.42 + albedo * (uLight * 0.4 + 0.08) + sparkle * 0.25;
  vA = (0.2 + 0.8 * k) / (1.0 + coc * 3.0) * (1.0 - fog) * tw * nearFade;
#else
  vec3 lit = albedo * (uLight * shade + 0.22 + sparkle * 0.9) + aGlow * 0.6;
  vCol = mix(mix(lit, own + aGlow * 0.3, uSelfLit), uFogColor, fog * 0.85);
  vA = (0.25 + 0.75 * k) / (1.0 + coc * 2.5) * (1.0 - fog * 0.3) * nearFade;
#endif
  vA *= 1.0 - aFade;
}`;

const FS = `
uniform float uOpacity, uBristle;
varying vec3 vCol; varying float vA; varying vec2 vRot; varying float vStretch, vSeed;
void main(){
  vec2 c = gl_PointCoord * 2.0 - 1.0; c.y = -c.y;
  vec2 s = vec2(dot(c, vRot), dot(c, vec2(-vRot.y, vRot.x)));   // along, across the stroke
  s.y *= vStretch;
  float d = dot(s, s);
  if (d > 1.0) discard;
  float a = exp(-d * 2.2);
  if (vStretch > 1.15) {
    float streak = 0.6 + 0.4 * sin(s.y * 11.0 + vSeed * 31.0);   // bristle lines along it
    float ragged = 1.0 - smoothstep(0.55, 1.0, abs(s.x) + 0.14 * sin(s.y * 9.0 + vSeed * 17.0));
    a *= mix(1.0, streak * ragged, uBristle * clamp(vStretch - 1.0, 0.0, 1.0));
  }
  // each stroke carries its own load of paint, a shade lighter or darker than its neighbours,
  // and its bristles leave fine light and dark lines along it
  float load = 0.93 + 0.14 * fract(vSeed * 91.7);
  float lines = vStretch > 1.15 ? 1.0 + 0.16 * uBristle * (sin(s.y * 11.0 + vSeed * 31.0) - 0.2) : 1.0;
  gl_FragColor = vec4(vCol * (0.93 + 0.1 * s.x) * load * lines, a * vA * uOpacity);   // paint thins out along it
}`;

export function pointsMaterial(glow, size, wander = 1) {
  const m = new THREE.ShaderMaterial({
    uniforms: {
      ...dreamU,
      uSize: { value: size }, uForm: { value: 0 }, uWander: { value: wander }, uOpacity: { value: 0 },
      uTint: { value: new THREE.Color(1, 1, 1) }, uTintAmt: { value: 0 },
      uSelfLit: { value: 0 }, uFogScale: { value: 1 }, uDof: { value: 1 },
    },
    defines: glow ? { GLOW: '' } : {},
    vertexShader: VS, fragmentShader: FS, transparent: true, depthWrite: false,
    blending: glow ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
  // clouds without per-stroke fading read 0 (a stale value at that slot could hide them)
  m.defaultAttributeValues = { ...m.defaultAttributeValues, aFade: [0] };
  m.userData.dreamPoints = m.uniforms;
  m.userData.wander0 = wander;
  return m;
}

// ------------------------------------------------------------------ texture lookups
const _pix = new WeakMap();
function pixels(tex) {
  if (!tex || !tex.image) return null;
  if (_pix.has(tex)) return _pix.get(tex);
  const img = tex.image; const w = img.width || 0, h = img.height || 0;
  if (!w || !h) return null;
  let data;
  try {
    const c = document.createElement('canvas'); c.width = Math.min(256, w); c.height = Math.min(256, h);
    const g = c.getContext('2d'); g.drawImage(img, 0, 0, c.width, c.height);
    data = { d: g.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height };
  } catch { data = null; }
  _pix.set(tex, data);
  return data;
}
const _lin = new THREE.Color();
function texel(tex, u, v, out) {
  const p = pixels(tex); if (!p) return 1;
  u = u * tex.repeat.x + tex.offset.x; v = v * tex.repeat.y + tex.offset.y;
  u -= Math.floor(u); v -= Math.floor(v);
  const x = Math.min(p.w - 1, Math.floor(u * p.w)), y = Math.min(p.h - 1, Math.floor((tex.flipY ? 1 - v : v) * p.h));
  const i = (y * p.w + x) * 4;
  _lin.setRGB(p.d[i] / 255, p.d[i + 1] / 255, p.d[i + 2] / 255, THREE.SRGBColorSpace);
  out.multiply(_lin);
  return p.d[i + 3] / 255;
}

// ------------------------------------------------------------------ surface sampling
function triangleTable(geom, weightFn, matrix) {
  const pos = geom.attributes.position, idx = geom.index;
  const tris = idx ? idx.count / 3 : pos.count / 3;
  const cdf = new Float32Array(tris); let total = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), m = new THREE.Vector3();
  for (let t = 0; t < tris; t++) {
    const i0 = idx ? idx.getX(t * 3) : t * 3, i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1, i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(pos, i0); b.fromBufferAttribute(pos, i1); c.fromBufferAttribute(pos, i2);
    if (matrix) { a.applyMatrix4(matrix); b.applyMatrix4(matrix); c.applyMatrix4(matrix); }
    let area = b.clone().sub(a).cross(c.clone().sub(a)).length() * 0.5;
    if (weightFn) { m.copy(a).add(b).add(c).multiplyScalar(1 / 3); area *= weightFn(m.x, m.z); }
    total += area; cdf[t] = total;
  }
  return { cdf, total, tris };
}

function pickTriangle(table, r) {
  const x = r * table.total; let lo = 0, hi = table.tris - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (table.cdf[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}

// A painter's stroke at a point: around the form (sideways on a body, across a face),
// bent by a slow flow so neighbouring strokes swirl together; tops and the ground follow
// the flow alone. 'up' strokes stand upright (grass, bark).
const _UP = new THREE.Vector3(0, 1, 0), _fl = new THREE.Vector3();
function strokeDir(p, n, mode, flow, swirl, out) {
  _fl.set(noise3(p.x * flow + 3.1, p.y * flow, p.z * flow) - 0.5,
    (noise3(p.x * flow, p.y * flow + 7.7, p.z * flow) - 0.5) * 0.6,
    noise3(p.x * flow, p.y * flow, p.z * flow + 1.9) - 0.5);
  if (mode === 'up') out.set(0, 1, 0);
  else { out.crossVectors(n, _UP); if (out.lengthSq() < 0.05) out.set(0, 0, 0); else out.normalize(); }
  out.addScaledVector(_fl, swirl);
  out.addScaledVector(n, -out.dot(n));                 // lie on the surface
  if (out.lengthSq() < 1e-6) out.set(1, 0, 0);
  return out.normalize();
}

// Sample `n` particles from a mesh's surface (in the mesh's local space; instanced meshes
// get every instance baked in). Colours come from the material, vertex colours and textures.
function sampleMesh(mesh, n, opts) {
  const geom = mesh.geometry, mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const pos = geom.attributes.position, nor = geom.attributes.normal, uv = geom.attributes.uv, col = geom.attributes.color;
  const idx = geom.index;
  const instances = mesh.isInstancedMesh ? mesh.count : 1;
  const mats = []; const tmp = new THREE.Matrix4();
  for (let k = 0; k < instances; k++) { if (mesh.isInstancedMesh) { mesh.getMatrixAt(k, tmp); mats.push(tmp.clone()); } else mats.push(null); }
  const table = triangleTable(geom, opts.weight && !mesh.isInstancedMesh ? opts.weight : null, null);
  const basic = mat.isMeshBasicMaterial;
  const additive = mat.blending === THREE.AdditiveBlending;
  const base = mat.color ? mat.color.clone() : new THREE.Color(1, 1, 1);
  const emissive = mat.emissive ? mat.emissive.clone().multiplyScalar(mat.emissiveIntensity ?? 1) : new THREE.Color(0, 0, 0);
  const P = [], N = [], A = [], G = [], D = [];
  const dd = new THREE.Vector3();
  const va = new THREE.Vector3(), vb = new THREE.Vector3(), vc = new THREE.Vector3(), p = new THREE.Vector3(), nn = new THREE.Vector3();
  const ca = new THREE.Color(), cg = new THREE.Color(), t3 = new THREE.Color();
  const nm = new THREE.Matrix3();
  let tries = 0, rnd = opts.rand;
  while (P.length / 3 < n && tries++ < n * 5) {
    const t = pickTriangle(table, rnd());
    const i0 = idx ? idx.getX(t * 3) : t * 3, i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1, i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    let u = rnd(), v = rnd(); if (u + v > 1) { u = 1 - u; v = 1 - v; } const w = 1 - u - v;
    va.fromBufferAttribute(pos, i0); vb.fromBufferAttribute(pos, i1); vc.fromBufferAttribute(pos, i2);
    p.set(0, 0, 0).addScaledVector(va, w).addScaledVector(vb, u).addScaledVector(vc, v);
    if (nor) { nn.set(0, 0, 0).addScaledVector(va.fromBufferAttribute(nor, i0), w).addScaledVector(vb.fromBufferAttribute(nor, i1), u).addScaledVector(vc.fromBufferAttribute(nor, i2), v).normalize(); }
    else nn.set(0, 1, 0);
    ca.copy(base);
    if (col && mat.vertexColors) { t3.setRGB(col.getX(i0) * w + col.getX(i1) * u + col.getX(i2) * v, col.getY(i0) * w + col.getY(i1) * u + col.getY(i2) * v, col.getZ(i0) * w + col.getZ(i1) * u + col.getZ(i2) * v); ca.multiply(t3); }
    let alpha = 1;
    let tu = 0, tv = 0;
    if (uv) { tu = uv.getX(i0) * w + uv.getX(i1) * u + uv.getX(i2) * v; tv = uv.getY(i0) * w + uv.getY(i1) * u + uv.getY(i2) * v; }
    if (mat.map && uv) alpha = texel(mat.map, tu, tv, ca);
    if (alpha < 0.08 || rnd() > alpha + 0.15) continue;       // follow the texture's shape (wings, sails)
    cg.copy(emissive);
    if (mat.emissiveMap && uv) texel(mat.emissiveMap, tu, tv, cg);
    if (basic) { if (additive) { cg.copy(ca); ca.setRGB(0, 0, 0); } else cg.copy(ca).multiplyScalar(0.55); }
    if (mat.transparent && mat.opacity < 1) {             // faint materials stay faint
      if (additive) { cg.multiplyScalar(mat.opacity); ca.multiplyScalar(mat.opacity); }
      else if (rnd() > mat.opacity + 0.1) continue;
    }
    const m = mats[Math.floor(rnd() * instances) % instances];
    if (m) { p.applyMatrix4(m); nm.getNormalMatrix(m); nn.applyMatrix3(nm).normalize(); }
    strokeDir(p, nn, opts.strokes, opts.flow, opts.swirl, dd);
    if (opts.thickness) p.addScaledVector(nn, (rnd() - 0.35) * opts.thickness);
    P.push(p.x, p.y, p.z); N.push(nn.x, nn.y, nn.z); A.push(ca.r, ca.g, ca.b); G.push(cg.r, cg.g, cg.b);
    D.push(dd.x, dd.y, dd.z);
  }
  return { P, N, A, G, D, additive: additive || (luminance(emissive) > 0.3 && luminance(emissive) > luminance(base) * 0.75) };
}

const luminance = (c) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

function surfaceArea(mesh, scale2) {
  const geom = mesh.geometry, pos = geom.attributes.position, idx = geom.index;
  const tris = idx ? idx.count / 3 : pos.count / 3; let area = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  for (let t = 0; t < tris; t++) {
    a.fromBufferAttribute(pos, idx ? idx.getX(t * 3) : t * 3); b.fromBufferAttribute(pos, idx ? idx.getX(t * 3 + 1) : t * 3 + 1); c.fromBufferAttribute(pos, idx ? idx.getX(t * 3 + 2) : t * 3 + 2);
    area += b.sub(a).cross(c.sub(a)).length() * 0.5;
  }
  if (mesh.isInstancedMesh) {
    const m = new THREE.Matrix4(), s = new THREE.Vector3(), q = new THREE.Quaternion(), v = new THREE.Vector3();
    let sum = 0; for (let k = 0; k < mesh.count; k++) { mesh.getMatrixAt(k, m); m.decompose(v, q, s); sum += Math.abs(s.x * s.y + s.y * s.z + s.x * s.z) / 3; }
    return area * sum * scale2;
  }
  return area * scale2;
}

function buildPoints(mesh, opts) {
  const ws = new THREE.Vector3(); mesh.getWorldScale(ws);
  const s = (Math.abs(ws.x) + Math.abs(ws.y) + Math.abs(ws.z)) / 3 || 1;
  const area = Math.max(1e-4, surfaceArea(mesh, s * s));
  const cfg = mesh.userData.points || {};
  const n = Math.round(Math.min(cfg.count ?? opts.max, Math.max(cfg.min ?? opts.min, area * (cfg.density ?? opts.density))));
  if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
  const bs = mesh.geometry.boundingSphere ? mesh.geometry.boundingSphere.radius : 1;
  const thickness = cfg.thickness ?? Math.min(bs * 0.12, 0.25 / s);
  // tall, thin shapes (trunks, stems, grass) take upright strokes
  if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
  const bb = mesh.geometry.boundingBox, bw = Math.max(bb.max.x - bb.min.x, bb.max.z - bb.min.z);
  const tall = (bb.max.y - bb.min.y) > 2.5 * bw && !mesh.isInstancedMesh;
  const sample = sampleMesh(mesh, n, { rand: opts.rand, weight: cfg.weight, thickness,
    strokes: cfg.strokes ?? opts.strokes ?? (tall ? 'up' : 'form'), flow: (cfg.flow ?? opts.flow ?? 0.55) * s, swirl: cfg.swirl ?? opts.swirl ?? 1.2 });
  const count = sample.P.length / 3; if (!count) return null;
  // Organic warp: bend straight edges and flat faces into soft, irregular forms.
  const warp = cfg.warp ?? opts.warp ?? 0.16;
  if (warp > 0) {
    const f = 1.1 / bs, A = warp * bs, P = sample.P;
    for (let i = 0; i < count; i++) {
      const x = P[i * 3] * f, y = P[i * 3 + 1] * f, z = P[i * 3 + 2] * f;
      P[i * 3] += (noise3(x + 3.1, y, z) - 0.5) * 2 * A;
      P[i * 3 + 1] += (noise3(x, y + 7.7, z) - 0.5) * 2 * A * 0.7;
      P[i * 3 + 2] += (noise3(x, y, z + 1.9) - 0.5) * 2 * A;
    }
  }
  const g = new THREE.BufferGeometry();
  const scatter = new Float32Array(count * 3), size = new Float32Array(count), seed = new Float32Array(count);
  const R = bs + 0.6 / s;
  for (let i = 0; i < count; i++) {
    const r = opts.rand, th = r() * 6.283, ph = Math.acos(2 * r() - 1), d = R * (1.2 + 2.2 * r());
    scatter[i * 3] = sample.P[i * 3] + Math.sin(ph) * Math.cos(th) * d;
    scatter[i * 3 + 1] = sample.P[i * 3 + 1] + Math.cos(ph) * d * 0.6 + R * 0.5;
    scatter[i * 3 + 2] = sample.P[i * 3 + 2] + Math.sin(ph) * Math.sin(th) * d;
    size[i] = 0.55 + 0.9 * r(); seed[i] = r();
    if (cfg.sizeByDistance) { const x = sample.P[i * 3], z = sample.P[i * 3 + 2]; size[i] *= 1 + Math.hypot(x, z) / cfg.sizeByDistance; }
  }
  g.setAttribute('position', new THREE.Float32BufferAttribute(sample.P, 3));
  g.setAttribute('aNormal', new THREE.Float32BufferAttribute(sample.N, 3));
  g.setAttribute('aColor', new THREE.Float32BufferAttribute(sample.A, 3));
  g.setAttribute('aGlow', new THREE.Float32BufferAttribute(sample.G, 3));
  g.setAttribute('aDir', new THREE.Float32BufferAttribute(sample.D, 3));
  g.setAttribute('aScatter', new THREE.BufferAttribute(scatter, 3));
  g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
  g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
  // world-space particle size from how densely the surface is covered
  const pointSize = THREE.MathUtils.clamp((cfg.size ?? 1.35) * Math.sqrt(area / count), opts.minSize, cfg.maxSize ?? opts.maxSize);
  const pts = new THREE.Points(g, pointsMaterial(sample.additive, pointSize, cfg.wander ?? opts.wander));
  pts.frustumCulled = false;
  pts.renderOrder = sample.additive ? 2 : 1;
  return pts;
}

// Replace the look of every solid mesh under `root` with a particle cloud. The meshes
// stay in place (invisible) so all existing animation keeps moving their particles.
export function pointify(root, opts = {}) {
  const o = { density: 700, min: 24, max: 14000, minSize: 0.012, maxSize: 0.22, wander: 1, ...opts };
  if (!o.rand) { let s = 1234567; o.rand = () => (s = (s * 16807) % 2147483647) / 2147483647; }
  root.updateMatrixWorld(true);
  const meshes = [];
  root.traverse((m) => {
    if (!(m.isMesh || m.isInstancedMesh) || m.userData.keepMesh) return;
    const mat = Array.isArray(m.material) ? m.material[0] : m.material;
    if (!mat || mat.isShaderMaterial || mat.isPointsMaterial) return;
    meshes.push(m);
  });
  for (const m of meshes) {
    const pts = buildPoints(m, o);
    if (!pts) continue;
    m.add(pts);
    const hidden = new THREE.MeshBasicMaterial({ visible: false });
    m.userData.solidMaterial = m.material;
    m.material = hidden;
  }
  return root;
}

// A generated picture (subject on black) becomes a particle cloud with a little relief.
export function imageCloud(img, size, count = 14000) {
  const W = 128, c = document.createElement('canvas'); c.width = W; c.height = W;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0, W, W);
  const d = g.getImageData(0, 0, W, W).data;
  const lum = (x, y) => { x = Math.max(0, Math.min(W - 1, x)); y = Math.max(0, Math.min(W - 1, y)); const k = (y * W + x) * 4; return 0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2]; };
  const P = [], A = [], G = [], N = [], S = [], E = [], D = [];
  const col = new THREE.Color();
  let s0 = 987654321; const rnd = () => (s0 = (s0 * 16807) % 2147483647) / 2147483647;
  for (let tries = 0; P.length / 3 < count && tries < count * 12; tries++) {
    const u = rnd(), v = rnd();
    const i = ((Math.floor(v * W) * W) + Math.floor(u * W)) * 4;
    const r = d[i] / 255, gg = d[i + 1] / 255, b = d[i + 2] / 255;
    const bright = Math.max(r, gg, b);
    const edge = 1 - THREE.MathUtils.smoothstep(Math.hypot(u - 0.5, v - 0.5), 0.36, 0.5);
    const keep = THREE.MathUtils.smoothstep(bright, 0.06, 0.24) * edge;   // drop the black background
    if (rnd() > keep) continue;
    const lum = 0.299 * r + 0.587 * gg + 0.114 * b;
    P.push((u - 0.5) * size, (0.5 - v) * size, (lum - 0.45) * size * 0.22 + (rnd() - 0.5) * size * 0.05);
    col.setRGB(r, gg, b, THREE.SRGBColorSpace);
    A.push(col.r, col.g, col.b); G.push(col.r * 0.22, col.g * 0.22, col.b * 0.22); N.push(0, 0, 1);
    // a painter follows the contours: the stroke runs across the brightness gradient
    const px = Math.floor(u * W), py = Math.floor(v * W);
    const gx = lum(px + 1, py) - lum(px - 1, py), gy = lum(px, py + 1) - lum(px, py - 1);
    if (Math.hypot(gx, gy) > 6) D.push(-gy, -gx, 0); else { const a = noise3(u * 3, v * 3, 0.5) * 6.283; D.push(Math.cos(a), Math.sin(a), 0); }
    const th = rnd() * 6.283, rr = size * (0.6 + rnd());
    S.push((u - 0.5) * size + Math.cos(th) * rr, (0.5 - v) * size + rr * 0.6, Math.sin(th) * rr);
    E.push(rnd());
  }
  const n = P.length / 3; if (!n) return null;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  geo.setAttribute('aColor', new THREE.Float32BufferAttribute(A, 3));
  geo.setAttribute('aGlow', new THREE.Float32BufferAttribute(G, 3));
  geo.setAttribute('aNormal', new THREE.Float32BufferAttribute(N, 3));
  geo.setAttribute('aDir', new THREE.Float32BufferAttribute(D, 3));
  geo.setAttribute('aScatter', new THREE.Float32BufferAttribute(S, 3));
  geo.setAttribute('aSize', new THREE.Float32BufferAttribute(E.map((e) => 0.6 + 0.8 * e), 1));
  geo.setAttribute('aSeed', new THREE.Float32BufferAttribute(E, 1));
  const pts = new THREE.Points(geo, pointsMaterial(false, size * 1.5 / Math.sqrt(n) * 1.4, 0.7));
  pts.frustumCulled = false; pts.renderOrder = 3;
  return pts;
}

// ------------------------------------------------------------------ paintings
// The imagery service paints a thing or a place and packs it in one picture, twice as tall
// as the painting (see services/imagery/app/relief.py): the colours on top; below them, how
// near each pixel is (red), the brush angle and the detail there (green: high and low four
// bits), and how much of the picture belongs to the thing (blue).
function readPainting(img) {
  const W = img.naturalWidth || img.width, H = (img.naturalHeight || img.height) >> 1;
  if (!W || H < 8) return null;
  const c = document.createElement('canvas'); c.width = W; c.height = H * 2;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0);
  return { W, H, d: g.getImageData(0, 0, W, H * 2).data, off: W * H * 4 };
}

// Sample strokes the way a painter lays them: a loose underpainting of big strokes over the
// whole shape, then smaller strokes where the picture is busy.
function brushwork(P, rnd, under, detail) {
  const { W, H, d, off } = P, out = [[], []];
  [under, detail].forEach((pass, k) => {
    const s = pass.stride;
    for (let gy = 0; gy < H; gy += s) for (let gx = 0; gx < W; gx += s) {
      const x = Math.min(W - 1, Math.floor(gx + rnd() * s)), y = Math.min(H - 1, Math.floor(gy + rnd() * s));
      const i = (y * W + x) * 4, j = off + i, mask = d[j + 2] / 255;
      if (mask < 0.04) continue;
      const g = d[j + 1], busy = (g & 15) / 15;
      if (rnd() > pass.keep(mask, busy)) continue;
      out[k].push({ u: (x + 0.5) / W, v: (y + 0.5) / H, i, near: d[j] / 255, mask, busy,
        ang: ((g >> 4) + 0.5) / 16 * Math.PI + (rnd() - 0.5) * 0.3, size: pass.size(busy) });
    }
  });
  return out;
}

function strokeGeometry(list, fill) {
  const n = list.length, f32 = (k) => new Float32Array(n * k);
  const A = { position: f32(3), aColor: f32(3), aGlow: f32(3), aNormal: f32(3), aDir: f32(3), aScatter: f32(3),
    aSize: f32(1), aSeed: f32(1), aFade: f32(1) };
  list.forEach((s, k) => fill(s, k, A));
  const geo = new THREE.BufferGeometry();
  for (const [name, arr] of Object.entries(A)) geo.setAttribute(name, new THREE.BufferAttribute(arr, arr.length / n));
  return geo;
}

function paintMaterial(size, wander, o) {
  const m = pointsMaterial(false, size, wander);
  const u = m.userData.dreamPoints;
  u.uSelfLit.value = o.selfLit; u.uFogScale.value = o.fogScale ?? 1; u.uDof.value = o.dof ?? 1;
  m.userData.paint = o;
  return m;
}

// A painted thing: its picture laid down in strokes, each one standing as near as the
// painting says, so the thing has relief. Edge strokes thin out into fragments.
export function reliefCloud(img, size, opts = {}) {
  const P = readPainting(img); if (!P) return null;
  let s0 = opts.seed ?? 424242; const rnd = () => (s0 = (s0 * 16807) % 2147483647) / 2147483647;
  const sA = Math.max(2, Math.round(P.W / 100)), sB = Math.max(1, Math.round(P.W / 240));
  const [under, fine] = brushwork(P, rnd,
    { stride: sA, keep: (m) => Math.pow(m, 0.7), size: () => sA * 2.6 },
    { stride: sB, keep: (m, b) => (m > 0.3 ? 0.16 + 0.84 * b : 0), size: (b) => sB * (2.0 + 2.2 * (1 - b)) });
  const budget = opts.count ?? 26000;
  while (under.length + fine.length > budget && fine.length) fine.splice(Math.floor(rnd() * fine.length), 1);
  const byNear = (a, b) => a.near - b.near;                     // far strokes first, near ones over them
  // the far side: a darker shell behind the painted face, so the thing has body when the
  // view comes round it a little
  const back = under.filter((t) => t.mask > 0.5).map((t) => ({ ...t, back: true }));
  const list = [...back.sort((a, b) => b.near - a.near), ...under.sort(byNear), ...fine.sort(byNear)];
  if (!list.length) return null;
  const aspect = P.H / P.W, depth = opts.depth ?? 0.3, col = new THREE.Color(), d = P.d;
  const geo = strokeGeometry(list, (s, k, A) => {
    const loose = (1 - s.mask) * size * 0.05;                   // the edges come apart a little
    const x = (s.u - 0.5) * size + (rnd() - 0.5) * loose, y = (0.5 - s.v) * size * aspect + (rnd() - 0.5) * loose;
    // (the far side mirrors the near one about the edge's depth, a little flatter)
    const z = s.back ? -(0.5 + 0.7 * s.near) * size * depth : (s.near - 0.5) * size * depth + (rnd() - 0.5) * size * 0.012;
    A.position.set([x, y, z], k * 3);
    col.setRGB(d[s.i] / 255, d[s.i + 1] / 255, d[s.i + 2] / 255, THREE.SRGBColorSpace);
    if (s.back) col.multiplyScalar(0.55);
    A.aColor.set([col.r, col.g, col.b], k * 3); A.aGlow.set([col.r * 0.08, col.g * 0.08, col.b * 0.08], k * 3);
    A.aNormal.set([0, 0, s.back ? -1 : 1], k * 3);
    A.aDir.set([Math.cos(s.ang), Math.sin(s.ang), 0], k * 3);
    const th = rnd() * 6.283, rr = size * (0.5 + rnd());
    A.aScatter.set([x + Math.cos(th) * rr, y + rr * 0.5, z + Math.sin(th) * rr], k * 3);
    A.aSize[k] = s.size; A.aSeed[k] = rnd();
    A.aFade[k] = 1 - Math.pow(s.mask, 0.8);
  });
  const pts = new THREE.Points(geo, paintMaterial(size / P.W, 0.5, { selfLit: opts.selfLit ?? 0.8 }));
  pts.frustumCulled = false; pts.renderOrder = 3; pts.userData.relief = true;
  return pts;
}

// The same painting again for another member of a group (shares the strokes).
export function reliefTwin(pts) {
  const m = pts.material, u = m.userData.dreamPoints;
  const twin = new THREE.Points(pts.geometry, paintMaterial(u.uSize.value, m.userData.wander0, m.userData.paint));
  twin.frustumCulled = false; twin.renderOrder = pts.renderOrder; twin.userData.relief = true;
  return twin;
}

// A painted place: a wide painting stood up around the dreamer like the inside of a bowl,
// every stroke as far away as the painting says (sky farthest), the ground rows below the
// horizon. Strokes turn along the bowl the way they run in the painting.
export function paintedSurround(img, opts = {}) {
  const P = readPainting(img); if (!P) return null;
  let s0 = opts.seed ?? 777; const rnd = () => (s0 = (s0 * 16807) % 2147483647) / 2147483647;
  const span = opts.span ?? Math.PI, top = opts.top ?? 0.87, bottom = opts.bottom ?? -0.52;
  const nearR = opts.near ?? 26, farR = opts.far ?? 75;
  const C = opts.center ?? new THREE.Vector3(0, 1.6, 6.5);
  const sA = Math.max(3, Math.round(P.W / 128)), sB = Math.max(2, Math.round(P.W / 256));
  const [under, fine] = brushwork(P, rnd,
    { stride: sA, keep: () => 1, size: () => sA * 2.1 },
    { stride: sB, keep: (m, b) => 0.08 + 0.8 * b, size: (b) => sB * (1.7 + 1.9 * (1 - b)) });
  for (const s of [...under, ...fine]) s.r = farR + (nearR - farR) * Math.pow(s.near, 1.4);
  const byFar = (a, b) => b.r - a.r;
  const list = [...under.sort(byFar), ...fine.sort(byFar)];
  if (!list.length) return null;
  const col = new THREE.Color(), d = P.d, sm = THREE.MathUtils.smoothstep;
  const geo = strokeGeometry(list, (s, k, A) => {
    const phi = (s.u - 0.5) * span, th = top + (bottom - top) * s.v, r = s.r;
    const cp = Math.cos(phi), sp = Math.sin(phi), ct = Math.cos(th), st = Math.sin(th);
    const dir = [sp * ct, st, -cp * ct];
    A.position.set([C.x + dir[0] * r, C.y + dir[1] * r, C.z + dir[2] * r], k * 3);
    col.setRGB(d[s.i] / 255, d[s.i + 1] / 255, d[s.i + 2] / 255, THREE.SRGBColorSpace);
    A.aColor.set([col.r, col.g, col.b], k * 3); A.aGlow.set([col.r * 0.05, col.g * 0.05, col.b * 0.05], k * 3);
    A.aNormal.set([-dir[0], -dir[1], -dir[2]], k * 3);
    const ca = Math.cos(s.ang), sa = Math.sin(s.ang);           // across the bowl, and up it
    A.aDir.set([ca * cp - sa * sp * st, sa * ct, ca * sp + sa * cp * st], k * 3);
    const lift = r * 0.12;
    A.aScatter.set([C.x + dir[0] * r + (rnd() - 0.5) * lift, C.y + dir[1] * r + rnd() * lift, C.z + dir[2] * r + (rnd() - 0.5) * lift], k * 3);
    A.aSize[k] = s.size * r; A.aSeed[k] = rnd();
    // the painting's edges melt into the sky around it
    A.aFade[k] = 1 - sm(Math.min(s.u, 1 - s.u), 0, 0.1) * sm(s.v, 0, 0.16) * sm(1 - s.v, 0, 0.06);
  });
  const pts = new THREE.Points(geo, paintMaterial(span / P.W, 0.22, { selfLit: opts.selfLit ?? 0.9, fogScale: 0.08, dof: 0.12 }));
  pts.frustumCulled = false; pts.renderOrder = -6;
  // its colours, for the sky, the fog and the light around it
  const avg = (v0, v1) => {
    const c = new THREE.Color(), acc = [0, 0, 0]; let n = 0;
    for (let y = Math.floor(v0 * P.H); y < Math.floor(v1 * P.H); y += 2) for (let x = 0; x < P.W; x += 4) {
      const i = (y * P.W + x) * 4; c.setRGB(d[i] / 255, d[i + 1] / 255, d[i + 2] / 255, THREE.SRGBColorSpace);
      acc[0] += c.r; acc[1] += c.g; acc[2] += c.b; n++;
    }
    return new THREE.Color(acc[0] / n, acc[1] / n, acc[2] / n);
  };
  return { pts, sky: avg(0, 0.14), horizon: avg(0.42, 0.6), ground: avg(0.86, 1) };
}
