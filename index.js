/**
 * index.js — a searchable index of FlyHash tags.
 *
 * All tags live in one contiguous Uint32Array. Search is a linear scan of
 * ANDs and popcounts, which sounds naive and is in fact the right answer at
 * this scale: 100k records × 64 words is ~6M word operations, a few
 * milliseconds, with no index structure to keep in sync and no cache misses
 * from chasing pointers.
 *
 * If you outgrow it, the next step is an inverted list keyed by cell index,
 * not a fancier metric.
 */

import { FlyHash } from './flyhash.js';
import { vectorize, vectorizeFields, DIM } from './featurize.js';

function popcount32(v) {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(v, 0x01010101) >>> 24);
}

export class FlyIndex {
  /**
   * @param {object} [opts]
   * @param {FlyHash} [opts.hasher]   supply your own to control seed/params
   * @param {object}  [opts.weights]  per-field weights for vectorizeFields
   * @param {number}  [opts.capacity=1024] initial rows, grows automatically
   */
  constructor({ hasher, weights = {}, capacity = 1024, featureOpts = {} } = {}) {
    this.hasher = hasher ?? new FlyHash({ inputDim: DIM });
    this.weights = weights;
    this.featureOpts = { dim: this.hasher.inputDim, ...featureOpts };

    this.words = this.hasher.words;
    this.ids = [];
    this.meta = [];
    // id -> row. With indexOf, reindexing N records is O(N^2); that is
    // invisible with integer ids and very visible with uuid strings.
    this.slots = new Map();
    this.tags = new Uint32Array(capacity * this.words);
    this.counts = new Uint16Array(capacity);   // bits set per row
    this.size = 0;
  }

  #grow() {
    const next = new Uint32Array(this.tags.length * 2);
    next.set(this.tags);
    this.tags = next;
    const nextCounts = new Uint16Array(this.counts.length * 2);
    nextCounts.set(this.counts);
    this.counts = nextCounts;
  }

  /** Vector for a record: a string, or an object of fields. */
  #vector(record) {
    return typeof record === 'string'
      ? vectorize(record, this.featureOpts)
      : vectorizeFields(record, this.weights, this.featureOpts);
  }

  /**
   * Add or replace a record.
   * @param {string|number} id
   * @param {string|object} record  text, or {field: value} to be weighted
   * @param {object} [meta]         anything you want back from search()
   * @returns {Uint32Array} the tag, so you can persist it
   */
  add(id, record, meta = null) {
    const tag = this.hasher.hash(this.#vector(record));
    return this.addTag(id, tag, meta);
  }

  /** Add a tag you already have — the path used when loading from the DB. */
  addTag(id, tag, meta = null) {
    const existing = this.slots.get(id);
    const slot = existing === undefined ? this.size : existing;

    if (slot === this.size) {
      if ((this.size + 1) * this.words > this.tags.length) this.#grow();
      this.ids.push(id);
      this.meta.push(meta);
      this.slots.set(id, slot);
      this.size++;
    } else {
      this.meta[slot] = meta;
    }

    this.tags.set(tag, slot * this.words);
    this.counts[slot] = FlyHash.count(tag);
    return tag;
  }

  /**
   * Nearest records by tag overlap.
   *
   * @param {string|object} query
   * @param {object} [opts]
   * @param {number} [opts.limit=10]
   * @param {number} [opts.minScore=0]  drop anything below this (0..1)
   * @returns {Array<{id, score, meta}>} sorted, best first
   */
  search(query, { limit = 10, minScore = 0 } = {}) {
    const qTag = this.hasher.hash(this.#vector(query));
    return this.searchTag(qTag, { limit, minScore });
  }

  searchTag(qTag, { limit = 10, minScore = 0 } = {}) {
    const { tags, counts, words, size } = this;

    let qCount = 0;
    for (let w = 0; w < words; w++) qCount += popcount32(qTag[w]);
    if (qCount === 0) return [];

    // Bounded top-k. The obvious version pushes an object per row and sorts
    // the lot, which allocates 100k objects to return 10 — that dominated the
    // runtime and was ~8x slower than the scan it was wrapping.
    const topRows = new Int32Array(limit).fill(-1);
    const topScores = new Float64Array(limit);
    let filled = 0;
    let floor = minScore;

    for (let row = 0; row < size; row++) {
      const base = row * words;
      let n = 0;
      for (let w = 0; w < words; w++) n += popcount32(tags[base + w] & qTag[w]);
      if (n === 0) continue;

      const score = n / Math.sqrt(qCount * counts[row]);
      if (score < floor) continue;
      if (filled === limit && score <= topScores[limit - 1]) continue;

      let pos = filled < limit ? filled : limit - 1;
      while (pos > 0 && topScores[pos - 1] < score) {
        topScores[pos] = topScores[pos - 1];
        topRows[pos] = topRows[pos - 1];
        pos--;
      }
      topScores[pos] = score;
      topRows[pos] = row;
      if (filled < limit) filled++;
      if (filled === limit) floor = Math.max(floor, topScores[limit - 1]);
    }

    const out = [];
    for (let i = 0; i < filled; i++) {
      const row = topRows[i];
      out.push({ id: this.ids[row], score: topScores[i], meta: this.meta[row] });
    }
    return out;
  }

  /**
   * Empirical noise floor: the overlap two unrelated records get by chance.
   *
   * This matters more than it sounds. Tags are sparse but not orthogonal, so
   * random pairs score somewhere above zero — with the default parameters,
   * around 0.05 to 0.15. A score of 0.1 therefore means "nothing", not "a
   * weak match". Run this once against your real data and set minScore above
   * the p99 it reports.
   *
   * @param {number} [samples=2000] random pairs to compare
   */
  calibrate(samples = 2000) {
    if (this.size < 2) throw new Error('FlyIndex: need at least 2 records to calibrate');
    const { tags, counts, words, size } = this;
    const scores = [];

    for (let s = 0; s < samples; s++) {
      const a = (Math.random() * size) | 0;
      let b = (Math.random() * size) | 0;
      if (a === b) b = (b + 1) % size;
      let n = 0;
      for (let w = 0; w < words; w++) {
        n += popcount32(tags[a * words + w] & tags[b * words + w]);
      }
      const denom = Math.sqrt(counts[a] * counts[b]);
      scores.push(denom > 0 ? n / denom : 0);
    }

    scores.sort((x, y) => x - y);
    const at = (q) => scores[Math.min(scores.length - 1, Math.floor(q * scores.length))];
    return {
      mean: scores.reduce((x, y) => x + y, 0) / scores.length,
      p50: at(0.5),
      p99: at(0.99),
      max: scores[scores.length - 1],
      suggestedMinScore: Math.min(0.95, Math.round(at(0.99) * 100) / 100 + 0.05),
    };
  }

  /** Rows as base64 strings, ready for an upsert. */
  export() {
    const out = [];
    for (let row = 0; row < this.size; row++) {
      const tag = this.tags.subarray(row * this.words, (row + 1) * this.words);
      out.push({ id: this.ids[row], tag: FlyHash.encode(tag) });
    }
    return out;
  }

  /** Rehydrate from stored base64 tags — no re-vectorisation, so it's instant. */
  load(rows, metaOf = () => null) {
    for (const row of rows) {
      this.addTag(row.id, FlyHash.decode(row.tag), metaOf(row));
    }
    return this;
  }
}

export { FlyHash, vectorize, vectorizeFields, DIM };
export default FlyIndex;
