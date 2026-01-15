"use client";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { fromUrl } from "geotiff";

type LonLat = [number, number];
type XY = [number, number];

type GeoJSONGeometry =
  | { type: "Polygon"; coordinates: LonLat[][] }
  | { type: "MultiPolygon"; coordinates: LonLat[][][] };

type GeoJSONFeature = {
  type: "Feature";
  properties?: Record<string, any>;
  geometry: GeoJSONGeometry;
};

type GeoJSONFC = { type: "FeatureCollection"; features: GeoJSONFeature[] };

type Dem = {
  width: number;
  height: number;
  nodata?: number;
  data: Float32Array;
  originLon: number;
  originLat: number;
  resLon: number;
  resLat: number;
};

type Bounds = { west: number; south: number; east: number; north: number };
type Rect = { x: number; y: number; w: number; h: number };

function clamp(x: number, a: number, b: number) {
  return Math.min(b, Math.max(a, x));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  const text = await r.text();
  if (!text.trim()) throw new Error(`${url}: empty response`);
  return JSON.parse(text) as T;
}

async function loadDemGeoTiff(url: string): Promise<Dem> {
  const tiff = await fromUrl(url);
  const image = await tiff.getImage();

  const width = image.getWidth();
  const height = image.getHeight();
  const rasters = (await image.readRasters({ interleave: true })) as any;

  const data = new Float32Array(rasters.length);
  for (let i = 0; i < rasters.length; i++) data[i] = rasters[i];

  const [originLon, originLat] = image.getOrigin();
  const [resLon, resLat] = image.getResolution();

  const fileDir: any = (image as any).fileDirectory;
  const nodataRaw = fileDir?.GDAL_NODATA;
  const nodata = nodataRaw != null ? Number(nodataRaw) : undefined;

  return { width, height, data, originLon, originLat, resLon, resLat, nodata };
}

function sampleDemAtLonLat(dem: Dem, lon: number, lat: number) {
  const px = (lon - dem.originLon) / dem.resLon;
  const py = (lat - dem.originLat) / dem.resLat;
  const x = Math.floor(px);
  const y = Math.floor(py);
  if (x < 0 || y < 0 || x >= dem.width || y >= dem.height) return null;
  const v = dem.data[y * dem.width + x];
  if (dem.nodata != null && v === dem.nodata) return null;
  if (v <= -32000) return null;
  return v;
}

function projectLonLatToXY(lon: number, lat: number, bounds: Bounds, inner: Rect) {
  const x =
    inner.x + ((lon - bounds.west) / (bounds.east - bounds.west)) * inner.w;
  const y =
    inner.y + ((bounds.north - lat) / (bounds.north - bounds.south)) * inner.h;
  return [x, y] as const;
}

function getBoundsFromAOI(fc: GeoJSONFC): Bounds {
  let west = Infinity,
    east = -Infinity,
    south = Infinity,
    north = -Infinity;

  const visit = (lon: number, lat: number) => {
    west = Math.min(west, lon);
    east = Math.max(east, lon);
    south = Math.min(south, lat);
    north = Math.max(north, lat);
  };

  for (const f of fc.features) {
    const g: any = f.geometry;
    if (!g) continue;

    if (g.type === "Polygon") {
      const rings = g.coordinates as LonLat[][];
      for (const ring of rings) for (const [lon, lat] of ring) visit(lon, lat);
    } else if (g.type === "MultiPolygon") {
      const polys = g.coordinates as LonLat[][][];
      for (const poly of polys)
        for (const ring of poly)
          for (const [lon, lat] of ring) visit(lon, lat);
    }
  }

  if (!Number.isFinite(west)) {
    return { west: -74.18, south: 4.46, east: -73.96, north: 4.71 };
  }

  return { west, south, east, north };
}

function laplacianSmooth2D(pts: XY[], passes: number, alpha = 0.5): XY[] {
  const n = pts.length;
  if (passes <= 0 || n < 3) return pts;

  let curr = pts;
  const isClosed =
    n > 3 && Math.hypot(curr[0][0] - curr[n - 1][0], curr[0][1] - curr[n - 1][1]) < 1e-3;

  for (let p = 0; p < passes; p++) {
    const next: XY[] = curr.map((v) => [v[0], v[1]]);

    if (isClosed) {
      for (let i = 0; i < n; i++) {
        const prev = curr[(i - 1 + n) % n];
        const self = curr[i];
        const nxt = curr[(i + 1) % n];
        const avgx = 0.5 * (prev[0] + nxt[0]);
        const avgy = 0.5 * (prev[1] + nxt[1]);
        next[i][0] = (1 - alpha) * self[0] + alpha * avgx;
        next[i][1] = (1 - alpha) * self[1] + alpha * avgy;
      }
      next[n - 1][0] = next[0][0];
      next[n - 1][1] = next[0][1];
    } else {
      next[0][0] = curr[0][0];
      next[0][1] = curr[0][1];
      next[n - 1][0] = curr[n - 1][0];
      next[n - 1][1] = curr[n - 1][1];

      for (let i = 1; i < n - 1; i++) {
        const prev = curr[i - 1];
        const self = curr[i];
        const nxt = curr[i + 1];
        const avgx = 0.5 * (prev[0] + nxt[0]);
        const avgy = 0.5 * (prev[1] + nxt[1]);
        next[i][0] = (1 - alpha) * self[0] + alpha * avgx;
        next[i][1] = (1 - alpha) * self[1] + alpha * avgy;
      }
    }

    curr = next;
  }

  return curr;
}

function buildAOIClipPath(
  aoi: GeoJSONFC,
  bounds: Bounds,
  inner: Rect,
  smoothPasses: number,
  clipScale: number
): Path2D | null {
  const path = new Path2D();

  const addRing = (ring: LonLat[]) => {
    if (ring.length < 3) return;

    let pts: XY[] = ring.map(([lon, lat]) => projectLonLatToXY(lon, lat, bounds, inner) as XY);

    const f = pts[0];
    const l = pts[pts.length - 1];
    if (Math.hypot(f[0] - l[0], f[1] - l[1]) > 1e-3) pts = [...pts, [f[0], f[1]]];

    if (smoothPasses > 0) pts = laplacianSmooth2D(pts, smoothPasses, 0.5);

    const s = clamp(clipScale, 0.5, 3.0);
    if (Math.abs(s - 1) > 1e-6) {
      let cx = 0,
        cy = 0;
      for (const [x, y] of pts) {
        cx += x;
        cy += y;
      }
      cx /= pts.length;
      cy /= pts.length;
      pts = pts.map(([x, y]) => [cx + (x - cx) * s, cy + (y - cy) * s]);
    }

    path.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) path.lineTo(pts[i][0], pts[i][1]);
    path.closePath();
  };

  let any = false;
  for (const feat of aoi.features) {
    const g: any = feat.geometry;
    if (!g) continue;

    if (g.type === "Polygon") {
      any = true;
      for (const ring of g.coordinates as LonLat[][]) addRing(ring);
    } else if (g.type === "MultiPolygon") {
      any = true;
      for (const poly of g.coordinates as LonLat[][][])
        for (const ring of poly) addRing(ring);
    }
  }

  return any ? path : null;
}

function rotatePoint(x: number, y: number, cx: number, cy: number, angRad: number): [number, number] {
  const dx = x - cx;
  const dy = y - cy;
  const c = Math.cos(angRad);
  const s = Math.sin(angRad);
  return [cx + dx * c - dy * s, cy + dx * s + dy * c];
}

/**
 * Build overlay RGBA texture (CPU once).
 * WebGL then handles bleed/noise/reveal every frame.
 */
function buildOverlayRGBA(
  dem: Dem,
  bounds: Bounds,
  inner: Rect,
  aoiClipPath: Path2D | null,
  opts: {
    levels: number;
    noiseAmp: number;
    noiseFreq: number;
    alpha: number;
    paletteStops: [number, number, number][];
    minMaxSampleStep: number;
  }
) {
  const w = Math.max(1, Math.floor(inner.w));
  const h = Math.max(1, Math.floor(inner.h));

  const tmp = document.createElement("canvas");
  tmp.width = 1;
  tmp.height = 1;
  const tmpCtx = tmp.getContext("2d");

  let minE = Infinity,
    maxE = -Infinity;
  const step = Math.max(2, Math.floor(opts.minMaxSampleStep));

  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const px = inner.x + x;
      const py = inner.y + y;

      if (aoiClipPath && tmpCtx) {
        if (!(tmpCtx as any).isPointInPath(aoiClipPath, px, py, "evenodd")) continue;
      }

      const lon = bounds.west + (x / w) * (bounds.east - bounds.west);
      const lat = bounds.north - (y / h) * (bounds.north - bounds.south);
      const e = sampleDemAtLonLat(dem, lon, lat);
      if (e == null) continue;

      minE = Math.min(minE, e);
      maxE = Math.max(maxE, e);
    }
  }

  if (!Number.isFinite(minE) || !Number.isFinite(maxE) || maxE <= minE) {
    minE = 2400;
    maxE = 3600;
  }

  // tiny deterministic noise helper
  const hash2 = (x: number, y: number) => {
    const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
    return s - Math.floor(s);
  };

  const valueNoise = (x: number, y: number) => {
    const x0 = Math.floor(x),
      y0 = Math.floor(y);
    const x1 = x0 + 1,
      y1 = y0 + 1;
    const sx = x - x0,
      sy = y - y0;
    const u = sx * sx * (3 - 2 * sx);
    const v = sy * sy * (3 - 2 * sy);
    const n00 = hash2(x0, y0);
    const n10 = hash2(x1, y0);
    const n01 = hash2(x0, y1);
    const n11 = hash2(x1, y1);
    const nx0 = lerp(n00, n10, u);
    const nx1 = lerp(n01, n11, u);
    const nxy = lerp(nx0, nx1, v);
    return nxy * 2 - 1;
  };

  const fbm = (x: number, y: number, oct = 5) => {
    let amp = 0.5;
    let freq = 1.0;
    let sum = 0;
    for (let i = 0; i < oct; i++) {
      sum += amp * valueNoise(x * freq, y * freq);
      freq *= 2.0;
      amp *= 0.5;
    }
    return clamp(sum, -1, 1);
  };

  const LEVELS = Math.max(0, Math.floor(opts.levels)); // 0 => no posterization
  const data = new Uint8Array(w * h * 4);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const px = inner.x + x;
      const py = inner.y + y;
      const idx = (y * w + x) * 4;

      if (aoiClipPath && tmpCtx) {
        if (!(tmpCtx as any).isPointInPath(aoiClipPath, px, py, "evenodd")) {
          data[idx + 3] = 0;
          continue;
        }
      }

      const lon = bounds.west + (x / w) * (bounds.east - bounds.west);
      const lat = bounds.north - (y / h) * (bounds.north - bounds.south);
      const e = sampleDemAtLonLat(dem, lon, lat);

      if (e == null) {
        data[idx + 3] = 0;
        continue;
      }

      let t = (e - minE) / (maxE - minE);
      const n = fbm(x * opts.noiseFreq, y * opts.noiseFreq, 5);
      t = clamp(t + n * opts.noiseAmp, 0, 1);

      if (LEVELS > 1) t = Math.round(t * LEVELS) / LEVELS;

      // palette sample (linear between stops)
      const stops = opts.paletteStops;
      const nStops = stops.length;
      const pos = clamp(t, 0, 1) * (nStops - 1);
      const i0 = Math.floor(pos);
      const f = pos - i0;
      const a = stops[Math.max(0, Math.min(nStops - 1, i0))];
      const b = stops[Math.max(0, Math.min(nStops - 1, i0 + 1))];

      const r = lerp(a[0], b[0], f);
      const g = lerp(a[1], b[1], f);
      const bb = lerp(a[2], b[2], f);

      data[idx + 0] = r & 255;
      data[idx + 1] = g & 255;
      data[idx + 2] = bb & 255;
      data[idx + 3] = Math.round(255 * clamp(opts.alpha, 0, 1));
    }
  }

  return { w, h, data };
}

// ---------- WebGL helpers ----------
function createShader(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  type: number,
  src: string
) {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`Shader compile failed: ${log}`);
  }
  return sh;
}

function createProgram(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  vsSrc: string,
  fsSrc: string
) {
  const vs = createShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const p = gl.createProgram()!;
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(`Program link failed: ${log}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return p;
}

function createTextureRGBA(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  w: number,
  h: number,
  data: ArrayBufferView | null,
  opts?: { linear?: boolean; clamp?: boolean }
) {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, opts?.linear ? gl.LINEAR : gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, opts?.linear ? gl.LINEAR : gl.NEAREST);
  gl.texParameteri(
    gl.TEXTURE_2D,
    gl.TEXTURE_WRAP_S,
    opts?.clamp === false ? gl.REPEAT : gl.CLAMP_TO_EDGE
  );
  gl.texParameteri(
    gl.TEXTURE_2D,
    gl.TEXTURE_WRAP_T,
    opts?.clamp === false ? gl.REPEAT : gl.CLAMP_TO_EDGE
  );
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data as any);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

function updateTextureFromImage(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  tex: WebGLTexture,
  img: HTMLImageElement,
  linear = true
) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear ? gl.LINEAR : gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function createFramebuffer(gl: WebGLRenderingContext | WebGL2RenderingContext, tex: WebGLTexture) {
  const fb = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fb;
}

function rasterizeAOIMask(inner: Rect, aoiPathPlate: Path2D | null, w: number, h: number) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) return null;

  ctx.clearRect(0, 0, w, h);

  // Path2D coordinates are in PLATE space; convert plate->mapTex pixel space:
  // mapTex pixel (x,y) corresponds to plate (inner.x + x, inner.y + y)
  // So we translate by -inner.x/-inner.y.
  if (aoiPathPlate) {
    ctx.save();
    ctx.translate(-inner.x, -inner.y);
    ctx.fillStyle = "rgba(255,255,255,1)";
    (ctx as any).fill(aoiPathPlate, "evenodd");
    ctx.restore();
  } else {
    ctx.fillStyle = "rgba(255,255,255,1)";
    ctx.fillRect(0, 0, w, h);
  }

  const img = ctx.getImageData(0, 0, w, h);

  // Put mask in alpha, keep RGB = 255 (simple)
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const a = img.data[i * 4 + 3];
    out[i * 4 + 0] = 255;
    out[i * 4 + 1] = 255;
    out[i * 4 + 2] = 255;
    out[i * 4 + 3] = a;
  }
  return { w, h, data: out };
}

// ---------- GLSL ----------
const VS_FULL_SCREEN = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

 // mask update shader: decay + diffusion + "paint" blob at pointer (with noisy edge)
const FS_MASK_UPDATE = `
precision mediump float;
varying vec2 vUv;

uniform sampler2D uPrevMask;
uniform vec2 uResolution;   // plate px
uniform vec2 uPointerPx;    // plate px
uniform float uTime;

uniform float uRadiusPx;    // blob radius
uniform float uFeatherPx;   // soft edge thickness
uniform float uDecay;       // 0.90..0.995
uniform float uDiffusion;   // 0..1 (how much blur gets mixed in)
uniform float uNoiseAmp;    // 0..1
uniform float uNoiseFreq;   // ~0.8..3.0 (in uv space)

float hash21(vec2 p){
  p = fract(p * vec2(123.34, 345.45));
  p += dot(p, p + 34.345);
  return fract(p.x * p.y);
}

float noise(vec2 p){
  vec2 i = floor(p);
  vec2 f = fract(p);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  vec2 u = f*f*(3.0-2.0*f);
  return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
}

void main(){
  vec2 px = vUv * uResolution;
  float prev = texture2D(uPrevMask, vUv).r;

  // 4-neighbor blur (cheap diffusion)
  vec2 texel = 1.0 / uResolution;
  float n0 = texture2D(uPrevMask, vUv + vec2(texel.x, 0.0)).r;
  float n1 = texture2D(uPrevMask, vUv - vec2(texel.x, 0.0)).r;
  float n2 = texture2D(uPrevMask, vUv + vec2(0.0, texel.y)).r;
  float n3 = texture2D(uPrevMask, vUv - vec2(0.0, texel.y)).r;
  float blur = (prev + n0 + n1 + n2 + n3) / 5.0;
  float base = mix(prev, blur, clamp(uDiffusion, 0.0, 1.0));
  base *= clamp(uDecay, 0.0, 1.0);

  // target blob
  float d = distance(px, uPointerPx);

  // noisy edge (blobby)
  float nn = noise(vUv * uNoiseFreq + vec2(uTime * 0.05, -uTime * 0.03));
  float wobble = (nn - 0.5) * 2.0 * uNoiseAmp * (uRadiusPx * 0.25);
  float r = uRadiusPx + wobble;
  float f = max(1.0, uFeatherPx);
  float t = 1.0 - smoothstep(r - f, r, d);

  // sharper core so it "paints" into the sim
  t = pow(t, 1.35);

  float outv = max(base, t);
  gl_FragColor = vec4(outv, outv, outv, 1.0);
}
`;

 // composite shader: hillshade base + overlay with ink bleed, revealed by mask + glow/shadow controls
const FS_COMPOSITE = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 vUv;

uniform vec2 uPlateSizePx;
uniform vec4 uInnerRectPx;   // x,y,w,h in plate px
uniform float uRotRad;
uniform vec2 uMapSizePx;

uniform sampler2D uHillshade;
uniform sampler2D uOverlay;
uniform sampler2D uAOIMask;
uniform sampler2D uRevealMask;

uniform float uHillAlpha;
uniform float uOverlayAlpha;
uniform float uBleedStrength;
uniform float uBleedRadiusPx;
uniform float uRevealSoftness;

// --- glow/shadow controls ---
uniform float uGlowStrength;    // ~0..2
uniform float uGlowRadiusPx;    // blur radius in map pixels, e.g. 2..12
uniform float uGlowThreshold;   // 0..1, where bloom starts, e.g. 0.55
uniform float uGlowGamma;       // >=1, bloom curve, e.g. 1.6
uniform vec3 uGlowTint;         // e.g. vec3(0.85,0.95,1.0) for cool glow

uniform float uShadowStrength;  // 0..1, e.g. 0.35
uniform float uShadowGamma;     // >=1, e.g. 1.8

vec2 rotateAbout(vec2 p, vec2 c, float a){
  float s = sin(a), co = cos(a);
  p -= c;
  p = vec2(p.x * co - p.y * s, p.x * s + p.y * co);
  return p + c;
}

float lum(vec3 c){ return dot(c, vec3(0.3333333)); }

void main(){
  vec2 platePx = vUv * uPlateSizePx;
  float x0 = uInnerRectPx.x;
  float y0 = uInnerRectPx.y;
  float w = uInnerRectPx.z;
  float h = uInnerRectPx.w;

  // Slightly "paper" (not pure white)
  vec3 paperBase = vec3(0.975, 0.972, 0.965);

  // Outside inner frame = paper only
  if (platePx.x < x0 || platePx.y < y0 || platePx.x > x0 + w || platePx.y > y0 + h) {
    gl_FragColor = vec4(paperBase, 1.0);
    return;
  }

  // Plate -> unrotated map UV
  vec2 c = vec2(x0 + 0.5*w, y0 + 0.5*h);
  vec2 unrot = rotateAbout(platePx, c, -uRotRad);
  vec2 mapUv = (unrot - vec2(x0, y0)) / vec2(w, h);

  // AOI clip
  float aoi = texture2D(uAOIMask, mapUv).a;
  if (aoi < 0.01) {
    gl_FragColor = vec4(paperBase, 1.0);
    return;
  }

  // Reveal mask in plate space (your existing sim)
  float m = texture2D(uRevealMask, vUv).r;
  float softness = clamp(uRevealSoftness, 0.0001, 1.0);
  float reveal = smoothstep(0.0, softness, m);

  // --- Hillshade base ---
  vec3 hill = texture2D(uHillshade, mapUv).rgb;
  float hl = lum(hill);

  // Give hillshade a slightly more photographic contrast curve
  // (tweak exponents to taste)
  hl = pow(clamp(hl, 0.0, 1.0), 0.85);
  vec3 base = mix(paperBase, vec3(hl), clamp(uHillAlpha, 0.0, 1.0));

  // --- NEW: Bloom/glow from blurred hillshade ---
  vec2 texel = 1.0 / uMapSizePx;
  vec2 o = texel * max(0.0, uGlowRadiusPx);

  vec3 h0 = texture2D(uHillshade, mapUv).rgb;
  vec3 h1 = texture2D(uHillshade, mapUv + vec2( o.x, 0.0)).rgb;
  vec3 h2 = texture2D(uHillshade, mapUv + vec2(-o.x, 0.0)).rgb;
  vec3 h3 = texture2D(uHillshade, mapUv + vec2(0.0, o.y)).rgb;
  vec3 h4 = texture2D(uHillshade, mapUv + vec2(0.0, -o.y)).rgb;
  vec3 h5 = texture2D(uHillshade, mapUv + vec2( o.x, o.y)).rgb;
  vec3 h6 = texture2D(uHillshade, mapUv + vec2(-o.x, o.y)).rgb;
  vec3 h7 = texture2D(uHillshade, mapUv + vec2( o.x, -o.y)).rgb;
  vec3 h8 = texture2D(uHillshade, mapUv + vec2(-o.x, -o.y)).rgb;

  float hb = lum((h0+h1+h2+h3+h4+h5+h6+h7+h8)/9.0);

  // Bloom only from highlights
  float g = smoothstep(clamp(uGlowThreshold, 0.0, 1.0), 1.0, hb);
  g = pow(g, max(0.0001, uGlowGamma));
  vec3 glow = uGlowTint * (g * uGlowStrength);

  // --- NEW: Shadow deepen ---
  float sh = pow(clamp(1.0 - hb, 0.0, 1.0), max(0.0001, uShadowGamma)) * uShadowStrength;

  vec3 col = base;
  col *= (1.0 - sh); // deepen shadows

  // --- Overlay + bleed (your existing ink feel) ---
  vec4 ov = texture2D(uOverlay, mapUv);
  vec3 ovRGB = ov.rgb;
  float ovA = ov.a * uOverlayAlpha;

  // 9-tap bleed on overlay
  vec2 o2 = texel * max(0.0, uBleedRadiusPx);

  vec4 s0 = texture2D(uOverlay, mapUv + vec2( o2.x, 0.0));
  vec4 s1 = texture2D(uOverlay, mapUv + vec2(-o2.x, 0.0));
  vec4 s2 = texture2D(uOverlay, mapUv + vec2(0.0, o2.y));
  vec4 s3 = texture2D(uOverlay, mapUv + vec2(0.0, -o2.y));
  vec4 s4 = texture2D(uOverlay, mapUv + vec2( o2.x, o2.y));
  vec4 s5 = texture2D(uOverlay, mapUv + vec2(-o2.x, o2.y));
  vec4 s6 = texture2D(uOverlay, mapUv + vec2( o2.x, -o2.y));
  vec4 s7 = texture2D(uOverlay, mapUv + vec2(-o2.x, -o2.y));

  vec4 b = (ov + s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7) / 9.0;
  vec3 bleed = b.rgb;
  float bleedA = b.a * uBleedStrength;

  // Bleed multiply (pigment seeping)
  col = mix(col, col * (0.5 + 0.5 * bleed), bleedA * reveal);

  // Main overlay (revealed)
  col = mix(col, ovRGB, ovA * reveal);

  // --- NEW: Apply glow as SCREEN blend, gated by reveal (or remove *reveal to make it always-on)
  vec3 glowScr = clamp(glow, 0.0, 1.0) * reveal;
  col = 1.0 - (1.0 - col) * (1.0 - glowScr);

  gl_FragColor = vec4(col, 1.0);
}
`;

export default function BogotaHillshadeAOIWebGL() {
  // assets
  const DEM_URL = "/bogota/bogota_raw_srtm.tif";
  const HILLSHADE_PNG_URL = "/bogota/hillshade.png";
  const AOI_URL = "/bogota/aoi_localidades.geojson";

  // rotation knob
  const ROT_DEG = 120;
  const ROT_RAD = (ROT_DEG * Math.PI) / 180;

  // AOI clip shaping
  const CLIP_VIEW_PAD_FRAC = 0.08;
  const CLIP_SCALE = 1.08;
  const CLIP_SMOOTH_PASSES = 8;

  // plate geometry
  const plateSize = 1100;
  const marginInner = 95;

  const inner: Rect = useMemo(
    () => ({
      x: marginInner,
      y: marginInner,
      w: plateSize - 2 * marginInner,
      h: plateSize - 2 * marginInner - 90,
    }),
    []
  );

  // visual knobs
  const HILL_ALPHA = 0.30;

  // overlay generation knobs (CPU once)
  const OVERLAY_LEVELS = 0; // 0 = no posterization
  const OVERLAY_NOISE_AMP = 0.0; // or tiny like 0.03 if you want gentle variation
  const OVERLAY_NOISE_FREQ = 3.0;
  const OVERLAY_PIXEL_ALPHA = 0.99;

  const OVERLAY_PALETTE_RGB: [number, number, number][] = [
    [0x00, 0x00, 0x00],
    [0x13, 0x00, 0x33],
    // [0x33, 0x9a, 0xff],
    [0x33, 0x47, 0xff],
    [0x0e, 0x1a, 0x81],
    [0xff, 0x00, 0x00],
    [0xff, 0xff, 0xff],
  ];

  // GPU overlay ink feel
  const OVERLAY_ALPHA = 2.0;
  const BLEED_STRENGTH = 20.0;
  const BLEED_RADIUS_PX = 30000000.0;

  // --- glow/shadow knobs ---
  const GLOW_STRENGTH = 1.0;
  const GLOW_RADIUS_PX = 90.0;
  const GLOW_THRESHOLD = 0.55;
  const GLOW_GAMMA = 10000.6;
  const GLOW_TINT: [number, number, number] = [0.0, 0.0, 100.0]; // cool-ish
  const SHADOW_STRENGTH = 0.35;
  const SHADOW_GAMMA = 1.8;

  // reveal sim knobs (GPU, per frame)
  const REVEAL_RADIUS_PX = 190;
  const REVEAL_FEATHER_PX = 135;
  const REVEAL_DECAY = 0.96; // closer to 1 => longer trails
  const REVEAL_DIFFUSION = 900000.0; // higher => more "watery"
  const REVEAL_NOISE_AMP = 2.5; // blob edge wobble (set to 0 to remove pointy noise)
  const REVEAL_NOISE_FREQ = 2.1;
  const REVEAL_SOFTNESS = 500000000000.0; // how quickly mask turns into reveal

  // pointer spring knobs (CPU)
  const SPRING_K = 80; // stiffness
  const SPRING_D = 16; // damping

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [dem, setDem] = useState<Dem | null>(null);
  const [aoi, setAoi] = useState<GeoJSONFC | null>(null);
  const [hillshadeImg, setHillshadeImg] = useState<HTMLImageElement | null>(null);
  const [status, setStatus] = useState<string>("loading");
  const [hover, setHover] = useState<{ lon: number; lat: number; elev: number | null } | null>(
    null
  );

  // bounds from AOI
  const bounds: Bounds = useMemo(() => {
    if (!aoi) return { west: -74.17658, south: 4.46084, east: -73.95979, north: 4.70682 };
    const b = getBoundsFromAOI(aoi);
    const lonR = b.east - b.west;
    const latR = b.north - b.south;
    const pad = clamp(CLIP_VIEW_PAD_FRAC, 0, 0.6);
    return {
      west: b.west - lonR * pad,
      east: b.east + lonR * pad,
      south: b.south - latR * pad,
      north: b.north + latR * pad,
    };
  }, [aoi]);

  const aoiClipPath = useMemo(() => {
    if (!aoi) return null;
    return buildAOIClipPath(aoi, bounds, inner, CLIP_SMOOTH_PASSES, CLIP_SCALE);
  }, [aoi, bounds, inner]);

  // load DEM + AOI
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        setStatus("loading DEM + AOI");
        const [demLoaded, aoiLoaded] = await Promise.all([
          loadDemGeoTiff(DEM_URL),
          fetchJson<GeoJSONFC>(AOI_URL),
        ]);
        if (cancelled) return;
        setDem(demLoaded);
        setAoi(aoiLoaded);
        setStatus("loaded");
      } catch (e: any) {
        console.error(e);
        if (!cancelled) setStatus(`error: ${e?.message ?? String(e)}`);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // load hillshade image
  useEffect(() => {
    const img = new Image();
    img.src = HILLSHADE_PNG_URL;
    img.onload = () => setHillshadeImg(img);
    img.onerror = () => setHillshadeImg(null);
  }, []);

  // WebGL runtime
  const glRef = useRef<{
    gl: WebGLRenderingContext | WebGL2RenderingContext;
    isWebGL2: boolean;

    // programs
    progMask: WebGLProgram;
    progComp: WebGLProgram;

    // quad
    vbo: WebGLBuffer;
    aPosMask: number;
    aPosComp: number;

    // textures
    texHill: WebGLTexture; // map texture (inner res)
    texOverlay: WebGLTexture; // map texture (inner res)
    texAOIMask: WebGLTexture; // map texture (inner res)

    texMaskA: WebGLTexture; // plate res
    texMaskB: WebGLTexture; // plate res

    fbMaskA: WebGLFramebuffer;
    fbMaskB: WebGLFramebuffer;

    // sizes
    platePx: number; // actual backing px (plateSize*dpr)
    dpr: number;

    // uniforms locations (mask)
    uPrevMask: WebGLUniformLocation | null;
    uRes: WebGLUniformLocation | null;
    uPointerPx: WebGLUniformLocation | null;
    uTime: WebGLUniformLocation | null;
    uRadiusPx: WebGLUniformLocation | null;
    uFeatherPx: WebGLUniformLocation | null;
    uDecay: WebGLUniformLocation | null;
    uDiffusion: WebGLUniformLocation | null;
    uNoiseAmp: WebGLUniformLocation | null;
    uNoiseFreq: WebGLUniformLocation | null;

    // uniforms locations (composite)
    uPlateSizePx: WebGLUniformLocation | null;
    uInnerRectPx: WebGLUniformLocation | null;
    uRotRad: WebGLUniformLocation | null;
    uMapSizePx: WebGLUniformLocation | null;
    uHillAlpha: WebGLUniformLocation | null;
    uOverlayAlpha: WebGLUniformLocation | null;
    uBleedStrength: WebGLUniformLocation | null;
    uBleedRadiusPx: WebGLUniformLocation | null;
    uRevealSoftness: WebGLUniformLocation | null;

    uHillTex: WebGLUniformLocation | null;
    uOverlayTex: WebGLUniformLocation | null;
    uAOIMaskTex: WebGLUniformLocation | null;
    uRevealMaskTex: WebGLUniformLocation | null;

    uGlowStrength: WebGLUniformLocation | null;
    uGlowRadiusPx: WebGLUniformLocation | null;
    uGlowThreshold: WebGLUniformLocation | null;
    uGlowGamma: WebGLUniformLocation | null;
    uGlowTint: WebGLUniformLocation | null;

    uShadowStrength: WebGLUniformLocation | null;
    uShadowGamma: WebGLUniformLocation | null;

    // pingpong
    ping: 0 | 1;
  } | null>(null);

  const pointerTargetRef = useRef<{ x: number; y: number; inside: boolean }>({
    x: -9999,
    y: -9999,
    inside: false,
  });

  const pointerSpringRef = useRef<{ x: number; y: number; vx: number; vy: number }>({
    x: -9999,
    y: -9999,
    vx: 0,
    vy: 0,
  });

  const rafRef = useRef<number | null>(null);
  const t0Ref = useRef<number>(performance.now());

  // init & run WebGL when assets are ready
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !dem || !aoiClipPath || !hillshadeImg) return;

    let destroyed = false;

    const init = () => {
      const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      canvas.width = Math.floor(plateSize * dpr);
      canvas.height = Math.floor(plateSize * dpr);
      canvas.style.width = `${plateSize}px`;
      canvas.style.height = `${plateSize}px`;

      const gl2 = canvas.getContext("webgl2", {
        premultipliedAlpha: false,
        antialias: true,
      }) as WebGL2RenderingContext | null;

      const gl = (gl2 ||
        (canvas.getContext("webgl", {
          premultipliedAlpha: false,
          antialias: true,
        }) as WebGLRenderingContext | null))!;

      if (!gl) throw new Error("WebGL not available");
      const isWebGL2 = !!gl2;
      console.log("[WebGL] Context:", isWebGL2 ? "WebGL2" : "WebGL1");

      // programs
      const progMask = createProgram(gl, VS_FULL_SCREEN, FS_MASK_UPDATE);
      const progComp = createProgram(gl, VS_FULL_SCREEN, FS_COMPOSITE);

      // full-screen quad
      const vbo = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, +1, -1, -1, +1, +1, +1]), gl.STATIC_DRAW);

      const aPosMask = gl.getAttribLocation(progMask, "aPos");
      const aPosComp = gl.getAttribLocation(progComp, "aPos");

      // map textures resolution = inner w/h in backing px (so it stays crisp)
      const mapW = Math.max(1, Math.floor(inner.w * dpr));
      const mapH = Math.max(1, Math.floor(inner.h * dpr));

      // hillshade texture
      const texHill = createTextureRGBA(gl, 1, 1, new Uint8Array([255, 255, 255, 255]), {
        linear: true,
        clamp: true,
      });
      updateTextureFromImage(gl, texHill, hillshadeImg, true);

      // AOI mask texture (alpha in inner space)
      const aoiMask = rasterizeAOIMask(inner, aoiClipPath, mapW, mapH);
      if (!aoiMask) throw new Error("Failed to rasterize AOI mask");
      const texAOIMask = createTextureRGBA(gl, mapW, mapH, aoiMask.data, { linear: true, clamp: true });

      // overlay texture (RGBA in inner space)
      const overlayRGBA = buildOverlayRGBA(dem, bounds, inner, aoiClipPath, {
        levels: OVERLAY_LEVELS,
        noiseAmp: OVERLAY_NOISE_AMP,
        noiseFreq: OVERLAY_NOISE_FREQ,
        alpha: OVERLAY_PIXEL_ALPHA,
        paletteStops: OVERLAY_PALETTE_RGB,
        minMaxSampleStep: 6,
      });

      // upload overlay (note: CPU built at inner.w/h in CSS px; resample to mapW/mapH via canvas)
      // simplest: draw to canvas at overlayRGBA.w/h then scale to mapW/mapH before tex upload.
      const ovCanvas = document.createElement("canvas");
      ovCanvas.width = overlayRGBA.w;
      ovCanvas.height = overlayRGBA.h;

      const ovCtx = ovCanvas.getContext("2d");
      if (!ovCtx) throw new Error("overlay ctx fail");

      const imgData = new ImageData(new Uint8ClampedArray(overlayRGBA.data.buffer), overlayRGBA.w, overlayRGBA.h);
      ovCtx.putImageData(imgData, 0, 0);

      const ovScaled = document.createElement("canvas");
      ovScaled.width = mapW;
      ovScaled.height = mapH;

      const ovS = ovScaled.getContext("2d");
      if (!ovS) throw new Error("overlay scale ctx fail");

      ovS.imageSmoothingEnabled = true;
      ovS.drawImage(ovCanvas, 0, 0, mapW, mapH);

      const ovScaledData = ovS.getImageData(0, 0, mapW, mapH).data;
      const texOverlay = createTextureRGBA(gl, mapW, mapH, new Uint8Array(ovScaledData.buffer), {
        linear: true,
        clamp: true,
      });

      // reveal mask ping-pong textures at plate backing resolution
      const platePx = Math.floor(plateSize * dpr);
      const zero = new Uint8Array(platePx * platePx * 4);

      const texMaskA = createTextureRGBA(gl, platePx, platePx, zero, { linear: true, clamp: true });
      const texMaskB = createTextureRGBA(gl, platePx, platePx, zero, { linear: true, clamp: true });

      const fbMaskA = createFramebuffer(gl, texMaskA);
      const fbMaskB = createFramebuffer(gl, texMaskB);

      // uniform locs
      gl.useProgram(progMask);
      const uPrevMask = gl.getUniformLocation(progMask, "uPrevMask");
      const uRes = gl.getUniformLocation(progMask, "uResolution");
      const uPointerPx = gl.getUniformLocation(progMask, "uPointerPx");
      const uTime = gl.getUniformLocation(progMask, "uTime");
      const uRadiusPx = gl.getUniformLocation(progMask, "uRadiusPx");
      const uFeatherPx = gl.getUniformLocation(progMask, "uFeatherPx");
      const uDecay = gl.getUniformLocation(progMask, "uDecay");
      const uDiffusion = gl.getUniformLocation(progMask, "uDiffusion");
      const uNoiseAmp = gl.getUniformLocation(progMask, "uNoiseAmp");
      const uNoiseFreq = gl.getUniformLocation(progMask, "uNoiseFreq");

      gl.useProgram(progComp);
      const uPlateSizePx = gl.getUniformLocation(progComp, "uPlateSizePx");
      const uInnerRectPx = gl.getUniformLocation(progComp, "uInnerRectPx");
      const uRotRad = gl.getUniformLocation(progComp, "uRotRad");
      const uMapSizePx = gl.getUniformLocation(progComp, "uMapSizePx");
      const uHillAlpha = gl.getUniformLocation(progComp, "uHillAlpha");
      const uOverlayAlpha = gl.getUniformLocation(progComp, "uOverlayAlpha");
      const uBleedStrength = gl.getUniformLocation(progComp, "uBleedStrength");
      const uBleedRadiusPx = gl.getUniformLocation(progComp, "uBleedRadiusPx");
      const uRevealSoftness = gl.getUniformLocation(progComp, "uRevealSoftness");

      const uHillTex = gl.getUniformLocation(progComp, "uHillshade");
      const uOverlayTex = gl.getUniformLocation(progComp, "uOverlay");
      const uAOIMaskTex = gl.getUniformLocation(progComp, "uAOIMask");
      const uRevealMaskTex = gl.getUniformLocation(progComp, "uRevealMask");

      const uGlowStrength = gl.getUniformLocation(progComp, "uGlowStrength");
      const uGlowRadiusPx = gl.getUniformLocation(progComp, "uGlowRadiusPx");
      const uGlowThreshold = gl.getUniformLocation(progComp, "uGlowThreshold");
      const uGlowGamma = gl.getUniformLocation(progComp, "uGlowGamma");
      const uGlowTint = gl.getUniformLocation(progComp, "uGlowTint");

      const uShadowStrength = gl.getUniformLocation(progComp, "uShadowStrength");
      const uShadowGamma = gl.getUniformLocation(progComp, "uShadowGamma");

      glRef.current = {
        gl,
        isWebGL2,

        progMask,
        progComp,

        vbo,
        aPosMask,
        aPosComp,

        texHill,
        texOverlay,
        texAOIMask,

        texMaskA,
        texMaskB,
        fbMaskA,
        fbMaskB,

        platePx,
        dpr,

        uPrevMask,
        uRes,
        uPointerPx,
        uTime,
        uRadiusPx,
        uFeatherPx,
        uDecay,
        uDiffusion,
        uNoiseAmp,
        uNoiseFreq,

        uPlateSizePx,
        uInnerRectPx,
        uRotRad,
        uMapSizePx,
        uHillAlpha,
        uOverlayAlpha,
        uBleedStrength,
        uBleedRadiusPx,
        uRevealSoftness,

        uHillTex,
        uOverlayTex,
        uAOIMaskTex,
        uRevealMaskTex,

        uGlowStrength,
        uGlowRadiusPx,
        uGlowThreshold,
        uGlowGamma,
        uGlowTint,

        uShadowStrength,
        uShadowGamma,

        ping: 0,
      };

      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);

      setStatus("webgl ready");
    };

    const drawQuad = (
      gl: WebGLRenderingContext | WebGL2RenderingContext,
      aPosLoc: number,
      vbo: WebGLBuffer
    ) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.enableVertexAttribArray(aPosLoc);
      gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.disableVertexAttribArray(aPosLoc);
    };

    let firstFrame = true;

    const step = (now: number) => {
      if (destroyed) return;

      const st = glRef.current;
      if (!st) return;

      const gl = st.gl;

      const dt = Math.min(0.05, (now - (t0Ref.current || now)) / 1000);
      t0Ref.current = now;

      const time = now / 1000;

      // spring pointer towards target (lag feel)
      const tgt = pointerTargetRef.current;
      const p = pointerSpringRef.current;

      if (!tgt.inside) {
        // if not inside, drift away but keep decay trails bob
        p.vx *= 0.92;
        p.vy *= 0.92;
      } else {
        const ax = SPRING_K * (tgt.x - p.x) - SPRING_D * p.vx;
        const ay = SPRING_K * (tgt.y - p.y) - SPRING_D * p.vy;
        p.vx += ax * dt;
        p.vy += ay * dt;
      }

      p.x += p.vx * dt;
      p.y += p.vy * dt;

      // if pointer not initialized, snap it (only when inside)
      const isSentinel = p.x < -1000 || p.y < -1000;
      if ((isSentinel || !Number.isFinite(p.x) || !Number.isFinite(p.y)) && tgt.inside) {
        p.x = tgt.x;
        p.y = tgt.y;
        p.vx = 0;
        p.vy = 0;
      }

      // -------- pass 1: update reveal mask (ping-pong) --------
      const srcMask = st.ping === 0 ? st.texMaskA : st.texMaskB;
      const dstFB = st.ping === 0 ? st.fbMaskB : st.fbMaskA;

      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFB);
      gl.viewport(0, 0, st.platePx, st.platePx);

      gl.useProgram(st.progMask);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcMask);
      gl.uniform1i(st.uPrevMask, 0);

      gl.uniform2f(st.uRes, st.platePx, st.platePx);

      // Only paint blob when hovering inside; otherwise send off-screen to let decay work
      if (tgt.inside) {
        const pxX = p.x * st.dpr;
        const pxY = st.platePx - p.y * st.dpr; // flip to bottom-left origin
        gl.uniform2f(st.uPointerPx, pxX, pxY);
      } else {
        // Off-screen position so no new blob is painted, existing one decays naturally
        gl.uniform2f(st.uPointerPx, -10000, -10000);
      }

      gl.uniform1f(st.uTime, time);
      gl.uniform1f(st.uRadiusPx, REVEAL_RADIUS_PX * st.dpr);
      gl.uniform1f(st.uFeatherPx, REVEAL_FEATHER_PX * st.dpr);
      gl.uniform1f(st.uDecay, REVEAL_DECAY);
      gl.uniform1f(st.uDiffusion, REVEAL_DIFFUSION);
      gl.uniform1f(st.uNoiseAmp, REVEAL_NOISE_AMP);
      gl.uniform1f(st.uNoiseFreq, REVEAL_NOISE_FREQ);

      drawQuad(gl, st.aPosMask, st.vbo);

      st.ping = st.ping === 0 ? 1 : 0;

      // -------- pass 2: composite to screen --------
      const curMask = st.ping === 0 ? st.texMaskA : st.texMaskB;

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, st.platePx, st.platePx);

      gl.useProgram(st.progComp);

      gl.uniform2f(st.uPlateSizePx, st.platePx, st.platePx);
      gl.uniform4f(
        st.uInnerRectPx,
        inner.x * st.dpr,
        inner.y * st.dpr,
        inner.w * st.dpr,
        inner.h * st.dpr
      );

      gl.uniform1f(st.uRotRad, ROT_RAD);
      gl.uniform2f(st.uMapSizePx, Math.floor(inner.w * st.dpr), Math.floor(inner.h * st.dpr));

      gl.uniform1f(st.uHillAlpha, HILL_ALPHA);
      gl.uniform1f(st.uOverlayAlpha, OVERLAY_ALPHA);
      gl.uniform1f(st.uBleedStrength, BLEED_STRENGTH);
      gl.uniform1f(st.uBleedRadiusPx, BLEED_RADIUS_PX * st.dpr);

      gl.uniform1f(st.uRevealSoftness, REVEAL_SOFTNESS);

      gl.uniform1f(st.uGlowStrength, GLOW_STRENGTH);
      gl.uniform1f(st.uGlowRadiusPx, GLOW_RADIUS_PX * st.dpr);
      gl.uniform1f(st.uGlowThreshold, GLOW_THRESHOLD);
      gl.uniform1f(st.uGlowGamma, GLOW_GAMMA);
      gl.uniform3f(st.uGlowTint, GLOW_TINT[0], GLOW_TINT[1], GLOW_TINT[2]);

      gl.uniform1f(st.uShadowStrength, SHADOW_STRENGTH);
      gl.uniform1f(st.uShadowGamma, SHADOW_GAMMA);

      // bind textures
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, st.texHill);
      gl.uniform1i(st.uHillTex, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, st.texOverlay);
      gl.uniform1i(st.uOverlayTex, 1);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, st.texAOIMask);
      gl.uniform1i(st.uAOIMaskTex, 2);

      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, curMask);
      gl.uniform1i(st.uRevealMaskTex, 3);

      drawQuad(gl, st.aPosComp, st.vbo);

      rafRef.current = requestAnimationFrame(step);
    };

    try {
      setStatus("init webgl");
      init();
      rafRef.current = requestAnimationFrame(step);
    } catch (e: any) {
      console.error(e);
      setStatus(`error: ${e?.message ?? String(e)}`);
    }

    return () => {
      destroyed = true;

      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;

      const st = glRef.current;
      glRef.current = null;

      if (!st) return;

      const gl = st.gl;
      gl.deleteTexture(st.texHill);
      gl.deleteTexture(st.texOverlay);
      gl.deleteTexture(st.texAOIMask);
      gl.deleteTexture(st.texMaskA);
      gl.deleteTexture(st.texMaskB);
      gl.deleteFramebuffer(st.fbMaskA);
      gl.deleteFramebuffer(st.fbMaskB);
      gl.deleteBuffer(st.vbo);
      gl.deleteProgram(st.progMask);
      gl.deleteProgram(st.progComp);
    };
  }, [dem, aoiClipPath, hillshadeImg, bounds, inner, ROT_RAD]);

  // mouse -> pointer target + hover sampling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !dem) return;

    // Update hover based on spring position every frame
    const updateHover = () => {
      const p = pointerSpringRef.current;
      const mx = p.x;
      const my = p.y;

      // Convert 5rem padding to plate coordinates
      const rect = canvas.getBoundingClientRect();
      const remPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      const paddingPlate = (remPx * 5 * plateSize) / rect.width;

      const inFrame =
        mx >= inner.x - paddingPlate &&
        my >= inner.y - paddingPlate &&
        mx <= inner.x + inner.w + paddingPlate &&
        my <= inner.y + inner.h + paddingPlate;

      if (!inFrame || !pointerTargetRef.current.inside) {
        setHover(null);
        return;
      }

      const cx = inner.x + inner.w / 2;
      const cy = inner.y + inner.h / 2;

      const [ux, uy] = rotatePoint(mx, my, cx, cy, -ROT_RAD);

      let inside = true;
      if (aoiClipPath) {
        const ctx2d = canvas.getContext("2d");
        if (ctx2d && !(ctx2d as any).isPointInPath(aoiClipPath, ux, uy, "evenodd")) inside = false;
      }

      if (!inside) {
        setHover(null);
        return;
      }

      const lon = bounds.west + ((ux - inner.x) / inner.w) * (bounds.east - bounds.west);
      const lat = bounds.north - ((uy - inner.y) / inner.h) * (bounds.north - bounds.south);
      const elev = sampleDemAtLonLat(dem, lon, lat);

      setHover({ lon, lat, elev: elev != null ? Math.round(elev) : null });
    };

    // Set up hover update interval
    const hoverInterval = setInterval(updateHover, 16);

    const onMove = (ev: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();

      // Convert from rendered CSS pixels -> plate coordinates (0..plateSize)
      const sx = plateSize / rect.width;
      const sy = plateSize / rect.height;

      const mx = (ev.clientX - rect.left) * sx;
      const my = (ev.clientY - rect.top) * sy;

      // Convert 5rem padding to plate coordinates
      const remPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      const paddingPlate = (remPx * 5 * plateSize) / rect.width;

      const inFrame =
        mx >= inner.x - paddingPlate &&
        my >= inner.y - paddingPlate &&
        mx <= inner.x + inner.w + paddingPlate &&
        my <= inner.y + inner.h + paddingPlate;

      // Just update the target, hover will follow the spring
      pointerTargetRef.current = { x: mx, y: my, inside: inFrame };
    };

    const onLeave = () => {
      pointerTargetRef.current = { x: -9999, y: -9999, inside: false };
      setHover(null);
    };

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    return () => {
      clearInterval(hoverInterval);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  }, [dem, bounds, inner, aoiClipPath, ROT_RAD]);

  return (
    <div className="w-full min-h-screen bg-neutral-100">
      <div className="max-w-[1300px] mx-auto px-6 py-6">
        <div className="inline-block shadow-2xl bg-white rounded-2xl p-4 border border-neutral-200">
          <canvas ref={canvasRef} className="block" />
          <div
            className="pt-3 text-sm text-neutral-700"
            style={{ fontFamily: '"IBM Plex Mono", ui-monospace' }}
          >
            <div>Status: {status}</div>
            {hover && (
              <div className="pt-1">
                Lon {hover.lon.toFixed(5)} · Lat {hover.lat.toFixed(5)} · Elev{" "}
                {hover.elev ?? "—"} m
              </div>
            )}
          </div>
          <div
            className="pt-2 text-xs text-neutral-500"
            style={{ fontFamily: '"IBM Plex Mono", ui-monospace' }}
          >
            ROT_DEG = {ROT_DEG} · reveal sim: decay {REVEAL_DECAY} · diffusion {REVEAL_DIFFUSION}
          </div>
        </div>
      </div>
    </div>
  );
}
