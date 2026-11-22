// Small geometry utilities: merge simple geometries so a whole forest is a few draw calls.
import * as THREE from 'three';

export function merge(geoms) {
  const parts = geoms.map((g) => (g.index ? g.toNonIndexed() : g));
  let n = 0; for (const g of parts) n += g.attributes.position.count;
  const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let o = 0; let hasColor = false;
  for (const g of parts) {
    const c = g.attributes.position.count;
    pos.set(g.attributes.position.array, o * 3);
    if (!g.attributes.normal) g.computeVertexNormals();
    nor.set(g.attributes.normal.array, o * 3);
    if (g.attributes.color) { col.set(g.attributes.color.array, o * 3); hasColor = true; } else col.fill(1, o * 3, (o + c) * 3);
    o += c;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  if (hasColor) out.setAttribute('color', new THREE.BufferAttribute(col, 3));
  for (const g of parts) g.dispose();
  return out;
}

// Paint a uniform vertex colour onto a geometry (so merged parts keep their own colours).
export function tint(geom, c) {
  const g = geom.index ? geom.toNonIndexed() : geom;
  const n = g.attributes.position.count, arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b; }
  g.setAttribute('color', new THREE.BufferAttribute(arr, 3));
  return g;
}

export function moved(geom, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0, s = 1) {
  const m = new THREE.Matrix4().compose(new THREE.Vector3(x, y, z),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)), new THREE.Vector3(s, s, s));
  geom.applyMatrix4(m);
  return geom;
}

// Push vertices along their normals with noise: rocks, peaks, craters.
export function roughen(geom, amount, seed = 1, freq = 1.7) {
  const p = geom.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = Math.sin(v.x * freq * 3.1 + seed) * Math.cos(v.y * freq * 2.3 - seed) * Math.sin(v.z * freq * 2.7 + seed * 0.5);
    v.multiplyScalar(1 + amount * n);
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geom.computeVertexNormals();
  return geom;
}

export function instanced(geom, mat, transforms) {
  const mesh = new THREE.InstancedMesh(geom, mat, Math.max(1, transforms.length));
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3();
  transforms.forEach((t, i) => {
    e.set(t.rx || 0, t.ry || 0, t.rz || 0); q.setFromEuler(e);
    s.set(t.sx ?? t.s ?? 1, t.sy ?? t.s ?? 1, t.sz ?? t.s ?? 1);
    m.compose(t.p, q, s); mesh.setMatrixAt(i, m);
  });
  mesh.count = transforms.length;
  mesh.instanceMatrix.needsUpdate = true;
  mesh.frustumCulled = false;
  return mesh;
}
