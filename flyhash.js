/**
 * flyhash.js — locality-sensitive hashing modelled on the Drosophila
 * olfactory circuit (Dasgupta, Stevens & Navlakha, Science 2017).
 *
 * Three steps, mirroring the mushroom body:
 *   1. normalise   — divisive normalisation in the antennal lobe
 *   2. expand      — sparse random projection onto many Kenyon cells
 *   3. winner-take-all — the APL neuron silences all but the top ~5%
 *
 * The result is a sparse binary tag. Similarity between two tags is just
 * the number of bits they share, which is a popcount over a bitset.
 *
 * No dependencies. No training. Works in Node and in the browser.
 */

/* ------------------------------------------------------------------ *
 * Deterministic PRNG (mulberry32)
 *
 * The projection MUST be reproducible. If it changes, every tag you have
 * already stored becomes meaningless. Seed is therefore explicit and the
 * generator is written out here rather than taken from Math.random.
 * ------------------------------------------------------------------ */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------------ *
 * Bitset helpers
 * ------------------------------------------------------------------ */

/** Population count of a 32-bit word (SWAR). */
function popcount32(v) {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(v, 0x01010101) >>> 24);
}

/** Number of bits set in both tags. Both must be the same length. */
export function overlap(a, b) {
  let n = 0;
  for (let i = 0; i < a.length; i++) n += popcount32(a[i] & b[i]);
  return n;
}

/** Cosine over binary tags: overlap / sqrt(|a| · |b|). */
export function similarity(a, b) {
  let ca = 0;
  let cb = 0;
  for (let i = 0; i < a.length; i++) {
    ca += popcount32(a[i]);
    cb += popcount32(b[i]);
  }
  if (ca === 0 || cb === 0) return 0;
  return overlap(a, b) / Math.sqrt(ca * cb);
}

/* ------------------------------------------------------------------ *
 * Quickselect — find the k-th largest activation without a full sort.
 * O(n) average instead of O(n log n). At 2048 cells per hash and one hash
 * per record this is the hot path, so it is worth not sorting.
 * ------------------------------------------------------------------ */
function kthLargest(values, k) {
  const a = Float64Array.from(values);
  let lo = 0;
  let hi = a.length - 1;
  const target = k - 1; // 0-indexed position in descending order

  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (a[i] > pivot) i++;
      while (a[j] < pivot) j--;
      if (i <= j) {
        const t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    }
    if (target <= j) hi = j;
    else if (target >= i) lo = i;
    else break;
  }
  return a[target];
}

/* ------------------------------------------------------------------ *
 * FlyHash
 * ------------------------------------------------------------------ */

export class FlyHash {
  /**
   * @param {object}  opts
   * @param {number}  opts.inputDim        size of the input vector
   * @param {number} [opts.cells=2048]     Kenyon cells; the expansion target
   * @param {number} [opts.sparsity=0.05]  fraction of cells left switched on
   * @param {number} [opts.fanIn=6]        input dims sampled per cell. 6 is
   *                                       what a real Kenyon cell gets, and it
   *                                       measured better than 12 at every
   *                                       cell count and sparsity tested — a
   *                                       wider fan-in lights too many cells
   *                                       on sparse text vectors, so genuinely
   *                                       different records collide.
   * @param {number} [opts.seed=20260913]  PRNG seed — changing it invalidates
   *                                       every tag you have already stored
   */
  constructor({
    inputDim,
    cells = 2048,
    sparsity = 0.05,
    fanIn = 6,
    seed = 20260913,
  } = {}) {
    if (!Number.isInteger(inputDim) || inputDim < 2) {
      throw new Error('FlyHash: inputDim must be an integer >= 2');
    }
    if (cells % 32 !== 0) {
      throw new Error('FlyHash: cells must be a multiple of 32');
    }
    if (fanIn > inputDim) fanIn = inputDim;

    this.inputDim = inputDim;
    this.cells = cells;
    this.sparsity = sparsity;
    this.fanIn = fanIn;
    this.seed = seed;

    /** bits set per tag */
    this.k = Math.max(1, Math.round(cells * sparsity));
    /** 32-bit words per tag */
    this.words = cells / 32;

    this.projection = this.#buildProjection();
    this.#scratch = new Float64Array(cells);
  }

  #scratch;

  /**
   * Sparse binary projection matrix, flattened.
   * Row c occupies [c * fanIn, (c + 1) * fanIn) and holds the input indices
   * that feed cell c. In the fly each Kenyon cell samples about six projection
   * neurons at random; this is the same idea with a tunable fan-in.
   */
  #buildProjection() {
    const rand = mulberry32(this.seed);
    const proj = new Int32Array(this.cells * this.fanIn);
    const seen = new Set();

    for (let c = 0; c < this.cells; c++) {
      seen.clear();
      let written = 0;
      const base = c * this.fanIn;
      // Sample without replacement so a cell never double-counts one input.
      while (written < this.fanIn) {
        const idx = (rand() * this.inputDim) | 0;
        if (seen.has(idx)) continue;
        seen.add(idx);
        proj[base + written] = idx;
        written++;
      }
    }
    return proj;
  }

  /**
   * Raw Kenyon cell activations for an input vector.
   * Exposed because the demo UI draws them; most callers want hash().
   * @param {ArrayLike<number>} vec
   * @returns {Float64Array} length = cells
   */
  activate(vec) {
    if (vec.length !== this.inputDim) {
      throw new Error(
        `FlyHash: expected vector of length ${this.inputDim}, got ${vec.length}`
      );
    }

    // Divisive normalisation. The antennal lobe rescales by mean activity so
    // that concentration (how loud the input is) does not change identity
    // (what the input is). Without this, long records beat short ones on
    // every query.
    let sum = 0;
    for (let i = 0; i < vec.length; i++) sum += vec[i];
    const mean = sum / vec.length;
    const scale = mean > 0 ? 1 / mean : 1;

    const out = this.#scratch;
    const { fanIn, projection } = this;

    for (let c = 0; c < this.cells; c++) {
      const base = c * fanIn;
      let acc = 0;
      for (let f = 0; f < fanIn; f++) acc += vec[projection[base + f]];
      out[c] = acc * scale;
    }
    return out;
  }

  /**
   * Hash a vector into a sparse binary tag.
   * @param {ArrayLike<number>} vec
   * @returns {Uint32Array} bitset of length `words`
   */
  hash(vec) {
    const act = this.activate(vec);
    const threshold = kthLargest(act, this.k);

    const tag = new Uint32Array(this.words);
    let set = 0;

    // Strictly-greater pass first, then fill the remainder from the ties.
    //
    // The `> 0` guard is load-bearing. A short input — a two-word search box
    // query — only reaches a few dozen cells, fewer than k. Padding the tag up
    // to k with silent cells picks them in index order, so every short input
    // ends up sharing the same block of low-index bits. That is a constant
    // background pattern masquerading as similarity, and it made unrelated
    // one-word queries score 0.28 against each other. A tag may therefore have
    // fewer than k bits, which is why similarity normalises by the counts
    // actually present rather than by k.
    for (let c = 0; c < this.cells && set < this.k; c++) {
      if (act[c] > threshold && act[c] > 0) {
        tag[c >>> 5] |= 1 << (c & 31);
        set++;
      }
    }
    if (threshold > 0) {
      for (let c = 0; c < this.cells && set < this.k; c++) {
        if (act[c] === threshold) {
          tag[c >>> 5] |= 1 << (c & 31);
          set++;
        }
      }
    }
    return tag;
  }

  /** Bits set in a tag. */
  static count(tag) {
    let n = 0;
    for (let i = 0; i < tag.length; i++) n += popcount32(tag[i]);
    return n;
  }

  /** Indices of the winning cells. Useful for visualisation and debugging. */
  winners(vec) {
    const tag = this.hash(vec);
    const out = [];
    for (let c = 0; c < this.cells; c++) {
      if (tag[c >>> 5] & (1 << (c & 31))) out.push(c);
    }
    return out;
  }

  /**
   * Similarity of two tags, 0..1.
   *
   * Cosine over the binary vectors: overlap / sqrt(|a| · |b|). When both tags
   * are full this is exactly overlap / k, the simple version. When one side is
   * a short query with fewer bits, it scales by what is actually there instead
   * of pretending the missing bits were misses.
   */
  compare(a, b) {
    const ca = FlyHash.count(a);
    const cb = FlyHash.count(b);
    if (ca === 0 || cb === 0) return 0;
    return overlap(a, b) / Math.sqrt(ca * cb);
  }

  /* ---------------- persistence ---------------- */

  /**
   * Tag → base64, for a Supabase `text` column.
   * 2048 cells becomes 344 characters. Storing the tag rather than a float
   * vector is why this needs no vector extension.
   */
  static encode(tag) {
    const bytes = new Uint8Array(tag.buffer, tag.byteOffset, tag.byteLength);
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
    let s = '';
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s);
  }

  /** base64 → tag. */
  static decode(b64) {
    let bytes;
    if (typeof Buffer !== 'undefined') {
      bytes = Uint8Array.from(Buffer.from(b64, 'base64'));
    } else {
      const bin = atob(b64);
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    }
    return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }

  /**
   * The config is the model. Store this next to your data — if any field
   * changes, existing tags must be regenerated.
   */
  config() {
    return {
      inputDim: this.inputDim,
      cells: this.cells,
      sparsity: this.sparsity,
      fanIn: this.fanIn,
      seed: this.seed,
      k: this.k,
    };
  }

  /** Stable fingerprint of the config, to detect drift at startup. */
  fingerprint() {
    const s = JSON.stringify(this.config());
    let h = 2166136261 >>> 0;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h.toString(16).padStart(8, '0');
  }
}

export default FlyHash;
