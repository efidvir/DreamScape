// The lens of the dream: camera drift, focus, bloom, depth-of-field and a final
// dream pass (wobble, chromatic fringe, edge haze, grain, flashes, ripples, warps).
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { snoise } from './kit.js';
import { dreamU } from './points.js';

const DreamShader = {
  uniforms: {
    tDiffuse: { value: null }, uTime: { value: 0 }, uHaze: { value: 0.8 }, uFlash: { value: 0 },
    uFlashColor: { value: new THREE.Color('#ffffff') }, uRipple: { value: 0 }, uWarp: { value: 0 },
    uDarken: { value: 0 }, uAspect: { value: 1.6 },
  },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uTime, uHaze, uFlash, uRipple, uWarp, uDarken, uAspect; uniform vec3 uFlashColor;
    varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec2 uv = vUv;
      vec2 c = uv - 0.5; c.x *= uAspect; float r = length(c);
      vec2 dir = normalize(c + 1e-5) * vec2(1.0 / uAspect, 1.0);
      uv += vec2(sin(uv.y * 7.0 + uTime * 0.6), cos(uv.x * 6.0 - uTime * 0.5)) * 0.0022 * uHaze;
      if (uRipple > 0.001) {
        float front = (1.0 - uRipple) * 1.1;
        float w = sin((r - front) * 45.0) * exp(-abs(r - front) * 9.0) * uRipple;
        uv += dir * w * 0.012;
      }
      float ca = (0.0012 + 0.005 * uHaze) * r * r * 2.0 + uWarp * 0.012;
      vec3 col = vec3(texture2D(tDiffuse, uv + dir * ca).r, texture2D(tDiffuse, uv).g, texture2D(tDiffuse, uv - dir * ca).b);
      if (uWarp > 0.001) {
        vec3 acc = col;
        for (int i = 1; i < 10; i++) { float s = 1.0 - float(i) * 0.02 * uWarp; acc += texture2D(tDiffuse, 0.5 + (uv - 0.5) * s).rgb; }
        col = acc / 10.0;
      }
      float edge = smoothstep(0.32, 0.95, r) * (0.3 + 0.7 * uHaze);
      if (edge > 0.01) {
        float k = 0.013 * edge; vec3 b = vec3(0.0);
        b += texture2D(tDiffuse, uv + vec2(k, 0.0)).rgb; b += texture2D(tDiffuse, uv - vec2(k, 0.0)).rgb;
        b += texture2D(tDiffuse, uv + vec2(0.0, k)).rgb; b += texture2D(tDiffuse, uv - vec2(0.0, k)).rgb;
        b += texture2D(tDiffuse, uv + vec2(k, k) * 0.7).rgb; b += texture2D(tDiffuse, uv - vec2(k, k) * 0.7).rgb;
        col = mix(col, b / 6.0, edge * 0.85);
      }
      float l = dot(col, vec3(0.299, 0.587, 0.114));
      col = mix(col, vec3(l), 0.15 * uHaze + 0.55 * uDarken);
      col *= 1.0 - 0.45 * uDarken;
      col *= mix(1.0, smoothstep(1.08, 0.22, r), 0.78);
      col += vec3(0.012, 0.014, 0.022) * uHaze;
      col += (hash(uv * vec2(1931.0, 1087.0) + fract(uTime)) - 0.5) * (0.012 + 0.014 * uHaze);
      col = mix(col, uFlashColor, clamp(uFlash, 0.0, 1.0));
      gl_FragColor = vec4(col, 1.0);
    }`,
};

export class DreamRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    // The discrete GPU belongs to the storyteller and the voice; on a laptop with two GPUs the
    // soft particle dream runs fine on the integrated one.
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'low-power' });
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 1);   // soft particles need no more
    this.renderer.setPixelRatio(this.pixelRatio);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.1, 1200);
    this.camera.position.set(0, 1.7, 7.2);

    this.home = new THREE.Vector3(0, 1.25, 0);
    this.attention = this.home.clone(); this.look = this.home.clone();
    this.attendUntil = 0;
    this.fx = { pulse: 0, flash: 0, ripple: 0, warp: 0, darken: 0, bloom: 0 };
    this.flashColor = new THREE.Color('#ffffff');
    this.fogBoost = 1; this.haze = 0.85; this.world = null;

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.7, 0.75, 0.2);
    this.composer.addPass(this.bloom);
    this.dreamPass = new ShaderPass(DreamShader);
    this.composer.addPass(this.dreamPass);
    this.composer.addPass(new OutputPass());

    this.timer = new THREE.Timer();
    this.maxDt = 0.1;               // tests on slow software GL raise this to keep real time
    this.elapsed = 0;
    this.frames = 0; this.slowFor = 0;
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
    this.composer.setPixelRatio(this.pixelRatio);
    this.composer.setSize(w, h);
    this.dreamPass.uniforms.uAspect.value = w / h;
    dreamU.uViewH.value = h * this.pixelRatio;
  }

  // ------------------------------------------------------------------ cues from the world
  attend(pos, animate = true) {
    this.attention.copy(pos);
    if (!animate) this.look.copy(this.home).lerp(pos, 0.35);
    this.attendUntil = this.elapsed + 8;
  }
  pulseClarity() { this.fx.pulse = 1; }
  flash(c, amount = 0.8) { this.flashColor.copy(c); this.fx.flash = Math.max(this.fx.flash, amount); }
  ripple() { this.fx.ripple = 1; }
  warp() { this.fx.warp = 1; }
  darken() { this.fx.darken = 1; }
  bloomBurst(k = 2.5) { this.fx.bloom = Math.max(this.fx.bloom, k); }

  // ------------------------------------------------------------------ loop
  start(onFrame) {
    let last = 0;
    this.renderer.setAnimationLoop((now) => {
      // While the storyteller writes its answer, half the frames are enough: the GPU is its.
      const gap = this.world && this.world.thinking ? 1000 / 30 - 2 : 0;
      if (now - last < gap) return;
      last = now;
      this.timer.update(now);
      const dt = Math.min(this.maxDt, this.timer.getDelta());
      this.elapsed += dt;
      const t = this.elapsed;
      this.step(t, dt);
      onFrame(t, dt);
      this.composer.render(dt);
      this.watchSpeed(dt, gap / 1000);
    });
  }

  step(t, dt) {
    const fx = this.fx;
    fx.pulse = Math.max(0, fx.pulse - dt / 7);
    fx.flash = Math.max(0, fx.flash - dt * 1.2);
    fx.ripple = Math.max(0, fx.ripple - dt / 2.2);
    fx.warp = Math.max(0, fx.warp - dt / 2.6);
    fx.darken = Math.max(0, fx.darken - dt / 9);
    fx.bloom = Math.max(0, fx.bloom - dt * 0.6);

    // How dreamy the image is: the story's clarity, a slow breath, and a sharpening after news.
    const clarity = this.world ? this.world.clarityNow : 0.2;
    const breath = 0.5 + 0.5 * Math.sin(t * 0.19) * 0.7 + 0.15 * snoise(t * 0.11, 4);
    const pulse = Math.sin(Math.min(1, (1 - fx.pulse) * 1.3) * Math.PI * 0.5) ;
    const sharpen = fx.pulse > 0 ? (1 - pulse) * 0.6 : 0;
    this.haze = THREE.MathUtils.clamp((1 - clarity) * (0.7 + 0.45 * breath) - sharpen, 0.06, 1);
    this.fogBoost = 1 + fx.warp * 1.6 + fx.darken * 0.8;

    // Camera: slow, breathing orbit with a little hand-held sway.
    const sway = Math.sin(t * 0.07) * 0.42 + snoise(t * 0.05, 9) * 0.12;
    const dist = 7.2 + Math.sin(t * 0.11) * 0.5 - fx.warp * 2.2;
    this.camera.position.set(Math.sin(sway) * dist, 1.65 + Math.sin(t * 0.13) * 0.28 + snoise(t * 0.2, 2) * 0.05, Math.cos(sway) * dist);
    const goal = t < this.attendUntil ? this.home.clone().lerp(this.attention, 0.55) : this.home;
    this.look.lerp(goal, Math.min(1, dt * 0.8));
    this.camera.lookAt(this.look);
    this.camera.fov = 55 + fx.warp * 14 * Math.sin(fx.warp * Math.PI);
    this.camera.updateProjectionMatrix();

    // Lens.
    const focusPoint = t < this.attendUntil ? this.attention : this.home;
    const focus = this.camera.position.distanceTo(focusPoint);
    dreamU.uFocus.value += (focus - dreamU.uFocus.value) * Math.min(1, dt * 2);
    dreamU.uAperture.value = 0.35 + 1.5 * this.haze;
    dreamU.uHaze.value = this.haze;
    this.bloom.strength = 0.5 + 0.35 * this.haze + fx.bloom;
    // Bright places (day, snow, clouds) would bloom everywhere: raise the bar with the fog's brightness.
    const fc = this.scene.fog ? this.scene.fog.color : null;
    const fogLum = fc ? 0.2126 * fc.r + 0.7152 * fc.g + 0.0722 * fc.b : 0;
    this.bloom.threshold = 0.2 + 1.4 * fogLum;
    this.bloom.radius = 0.75;
    const d = this.dreamPass.uniforms;
    d.uTime.value = t; d.uHaze.value = this.haze; d.uFlash.value = fx.flash * fx.flash;
    d.uFlashColor.value.copy(this.flashColor); d.uRipple.value = fx.ripple; d.uWarp.value = Math.sin(fx.warp * Math.PI);
    d.uDarken.value = Math.sin(fx.darken * Math.PI * 0.5);
  }

  // Keep it smooth on weaker GPUs: lower resolution, then drop depth-of-field.
  watchSpeed(dt, gap = 0) {
    this.frames++;
    const slow = dt > Math.max(1 / 32, gap * 1.5);
    if (slow) this.slowFor += dt; else this.slowFor = Math.max(0, this.slowFor - dt * 0.5);
    if (this.slowFor > 4 && this.frames > 240) {
      this.slowFor = 0;
      if (this.pixelRatio > 0.75) { this.pixelRatio = Math.max(0.75, this.pixelRatio - 0.25); this.resize(); }
    }
  }
}
