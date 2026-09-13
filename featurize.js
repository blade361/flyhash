/**
 * featurize.js — turns record text into the fixed-length vector FlyHash eats.
 *
 * This is the part that decides what "similar" means. FlyHash preserves
 * whatever similarity is already in this vector; it does not invent any.
 * So the tokenizer is where you spend your thinking, not the hash.
 *
 * Defaults here are tuned for short business records: company names, contact
 * notes, service tickets. Word tokens catch shared vocabulary; character
 * 4-grams catch typos, inflection and German compounds, which matters when
 * "Containerdienst" and "Container-Dienst" should land together.
 */

const DEFAULT_DIM = 4096;

/**
 * Words that carry no signal in a record. Kept deliberately short — an
 * aggressive stoplist costs more recall than it buys precision at this scale.
 */
const STOP = new Set([
  // en
  'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was', 'has',
  'have', 'not', 'but', 'you', 'our', 'their', 'his', 'her', 'its',
  // de
  'der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'einen', 'den', 'dem',
  'des', 'mit', 'von', 'für', 'auf', 'nicht', 'wir', 'sie', 'auch',
  // es
  'los', 'las', 'una', 'unos', 'unas', 'con', 'por', 'para', 'que', 'del',
  'como', 'más', 'pero', 'sus',
]);

/** FNV-1a, 32-bit. Deterministic across processes, unlike object key order. */
function fnv1a(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

/**
 * Lowercase, strip accents, collapse everything that isn't a letter or digit.
 * Accent folding means "Riópar" and "Riopar" hash to the same token.
 */
export function normalise(text) {
  return String(text)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9äöüß\s]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function tokens(text) {
  return normalise(text)
    .split(' ')
    .filter((t) => t.length > 1 && !STOP.has(t));
}

/**
 * Character n-grams inside word boundaries. The boundary markers stop
 * "ontainerd" style grams from spanning two unrelated words.
 */
function charGrams(word, n) {
  const padded = `^${word}$`;
  const out = [];
  for (let i = 0; i + n <= padded.length; i++) out.push(padded.slice(i, i + n));
  return out;
}

/**
 * Text → sparse-ish float vector via the hashing trick.
 *
 * @param {string} text
 * @param {object} [opts]
 * @param {number} [opts.dim=4096]      must match FlyHash inputDim
 * @param {number} [opts.gram=4]        character n-gram size, 0 to disable
 * @param {number} [opts.gramWeight=0.5] grams count for less than whole words
 * @returns {Float64Array}
 */
export function vectorize(text, opts = {}) {
  const { dim = DEFAULT_DIM, gram = 4, gramWeight = 0.5 } = opts;
  const vec = new Float64Array(dim);

  for (const tok of tokens(text)) {
    vec[fnv1a(tok) % dim] += 1;
    if (gram > 0 && tok.length >= gram) {
      for (const g of charGrams(tok, gram)) {
        vec[fnv1a(g) % dim] += gramWeight;
      }
    }
  }

  // Sublinear scaling: a word repeated ten times is not ten times as relevant.
  for (let i = 0; i < dim; i++) {
    if (vec[i] > 0) vec[i] = 1 + Math.log(vec[i]);
  }

  // L2 normalise so record length doesn't dominate. FlyHash also does its own
  // divisive normalisation, but doing it here keeps field weighting honest.
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < dim; i++) vec[i] /= norm;

  return vec;
}

/**
 * Weighted multi-field vectorisation — the version you actually want for CRM
 * rows, where a match on company name should count for more than a match
 * buried in a notes field.
 *
 * @example
 *   vectorizeFields({ name: 'Orban Klinger', notes: '...' }, { name: 3, notes: 1 })
 */
export function vectorizeFields(fields, weights = {}, opts = {}) {
  const { dim = DEFAULT_DIM } = opts;
  const vec = new Float64Array(dim);

  for (const [field, value] of Object.entries(fields)) {
    if (value == null || value === '') continue;
    const w = weights[field] ?? 1;
    if (w === 0) continue;
    const part = vectorize(String(value), { ...opts, dim });
    for (let i = 0; i < dim; i++) vec[i] += part[i] * w;
  }

  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < dim; i++) vec[i] /= norm;

  return vec;
}

export const DIM = DEFAULT_DIM;
