import { FlyHash } from './flyhash.js';
import { vectorize, DIM } from './featurize.js';
import { FlyIndex } from './index.js';

const pass = (m) => console.log(`  ok    ${m}`);
const fail = (m) => { console.log(`  FAIL  ${m}`); process.exitCode = 1; };
const check = (cond, m) => (cond ? pass(m) : fail(m));

console.log('\n--- invariants ---');

const fh = new FlyHash({ inputDim: DIM });
const v1 = vectorize('Orban Containerdienst skip hire Mannheim');
const t1 = fh.hash(v1);

const bitsOf = (t) => [...t].reduce((n, w) => n + ((w >>> 0).toString(2).split('1').length - 1), 0);

// How many cells a tag actually reaches depends on how many input dimensions
// are non-zero, i.e. how long the text is. k is a ceiling, not a guarantee.
// Below ~10 words a tag is activation-limited rather than k-limited, which is
// exactly why similarity normalises by the bits present.
const lengths = [
  ['1 word', 'Orban'],
  ['5 words', 'Orban Containerdienst skip hire Mannheim'],
  ['12 words', 'Orban Containerdienst GmbH Mannheim skip hire container rental demolition waste site clearance'],
  ['30 words', 'Orban Containerdienst GmbH Mannheim skip hire container rental demolition waste site clearance '
    + 'roll off dumpster construction debris removal recycling gravel sand soil transport logistics fleet '
    + 'maintenance contract client since 2014 invoice terms net thirty'],
];
console.log('  bits set by input length (k = ' + fh.k + '):');
for (const [label, text] of lengths) {
  console.log(`    ${label.padEnd(9)} ${String(bitsOf(fh.hash(vectorize(text)))).padStart(4)} bits`);
}
check(bitsOf(fh.hash(vectorize(lengths[3][1]))) === fh.k, 'a long record saturates at exactly k bits');
check(bitsOf(t1) <= fh.k, 'a tag never exceeds k bits');

const t1b = fh.hash(vectorize('Orban Containerdienst skip hire Mannheim'));
check(t1.every((w, i) => w === t1b[i]), 'same input gives same tag');

const fh2 = new FlyHash({ inputDim: DIM, seed: fh.seed });
check(fh2.hash(v1).every((w, i) => w === t1[i]), 'same seed reproduces tag across instances');

const fh3 = new FlyHash({ inputDim: DIM, seed: 999 });
check(!fh3.hash(v1).every((w, i) => w === t1[i]), 'different seed gives different tag');

check(fh.compare(t1, t1) === 1, 'self-similarity is exactly 1');

const enc = FlyHash.encode(t1);
check(FlyHash.decode(enc).every((w, i) => w === t1[i]), `base64 round-trips (${enc.length} chars)`);

// Length invariance: the same content padded out should stay close.
const short = fh.hash(vectorize('container rental'));
const long = fh.hash(vectorize('container rental ' + 'container rental '.repeat(20)));
check(fh.compare(short, long) > 0.8, `length invariance holds (${fh.compare(short, long).toFixed(2)})`);

// Two independent random inputs should overlap at chance: k/cells.
// If this comes out high, the projection is correlated and the whole thing
// is broken, independent of any corpus.
let chanceSum = 0;
const TRIALS = 300;
for (let t = 0; t < TRIALS; t++) {
  const a = new Float64Array(DIM);
  const b = new Float64Array(DIM);
  for (let i = 0; i < DIM; i++) { a[i] = Math.random(); b[i] = Math.random(); }
  chanceSum += fh.compare(fh.hash(a), fh.hash(b));
}
const chance = chanceSum / TRIALS;
const expected = fh.k / fh.cells;
check(Math.abs(chance - expected) < 0.02,
  `random inputs overlap at chance (${chance.toFixed(3)} vs theoretical ${expected.toFixed(3)})`);

// Regression: short inputs must not share a constant background pattern.
// Before the `> 0` guard in hash(), inputs this short reached fewer cells than
// k, the remainder were padded in index order, and two unrelated one-word
// queries scored ~0.28 against each other purely from that shared padding.
const shorties = ['riopar', 'mannheim', 'klinger', 'hotel', 'satellite'];
let worstShort = 0;
for (let i = 0; i < shorties.length; i++) {
  for (let j = i + 1; j < shorties.length; j++) {
    const s = fh.compare(fh.hash(vectorize(shorties[i])), fh.hash(vectorize(shorties[j])));
    worstShort = Math.max(worstShort, s);
  }
}
check(worstShort < 0.12, `unrelated one-word inputs stay near zero (worst ${worstShort.toFixed(3)})`);

const tShort = fh.hash(vectorize('orban'));
const bitsShort = [...tShort].reduce((n, w) => n + ((w >>> 0).toString(2).split('1').length - 1), 0);
check(bitsShort < fh.k, `short input sets fewer than k bits (${bitsShort} < ${fh.k}), not padded`);

console.log('\n--- ranking sanity ---');

const probe = 'skip hire and container rental for construction sites';
const candidates = {
  'near   (paraphrase)': 'container rental and skip hire for building sites',
  'near   (typo)':       'skip hier and contaner rental for construcion sites',
  'mid    (same domain)':'waste disposal and roll-off dumpster service',
  'far    (other domain)':'hotel booking engine with channel manager integration',
  'far    (unrelated)':  'orbital station keeping for low earth orbit satellites',
};

const pTag = fh.hash(vectorize(probe));
const scored = Object.entries(candidates).map(([label, text]) => [
  label, fh.compare(pTag, fh.hash(vectorize(text))),
]);
for (const [label, s] of scored) {
  const bar = '#'.repeat(Math.round(s * 40));
  console.log(`  ${label.padEnd(22)} ${s.toFixed(3)}  ${bar}`);
}
const byScore = [...scored].sort((a, b) => b[1] - a[1]).map(([l]) => l[0]);
check(byScore[0] === 'n' && byScore[1] === 'n', 'the two near items rank top');

// NOT asserted: that "waste disposal / roll-off dumpster" beats "orbital
// station keeping". It doesn't, and it can't — those share no characters with
// the probe, so both sit at the noise floor. This is the documented limit of
// lexical hashing, not a bug. Everything below the floor is indistinguishable.
const nearMin = Math.min(scored[0][1], scored[1][1]);
const farMax = Math.max(scored[2][1], scored[3][1], scored[4][1]);
check(nearMin > farMax * 3, `near items clear the floor by 3x (${nearMin.toFixed(2)} vs ${farMax.toFixed(2)})`);

console.log('\n--- scale ---');

const industries = ['skip hire', 'agricultural services', 'hotel software', 'logistics',
  'demolition', 'landscaping', 'catering', 'fleet maintenance', 'satellite operations'];
const cities = ['Mannheim', 'Heidelberg', 'Albacete', 'Riópar', 'Hamburg', 'Valencia'];
const suffix = ['GmbH', 'S.L.', 'AG', 'Ltd', 'e.K.'];

const N = process.env.CI ? 20_000 : 100_000;
const idx = new FlyIndex({ weights: { name: 3, industry: 2, notes: 1 } });

// A cyclic generator makes thousands of literal duplicates, which poisons the
// noise-floor measurement (it reported p99 = 0.97 because 1% of "random" pairs
// were the same record twice). Real CRM rows share vocabulary but not identity,
// so build them from a vocabulary with a seeded RNG instead.
// Domain vocabulary, shared across records the way real jargon is.
const vocab = ('acero albaran alquiler anfrage angebot auftrag baustelle bauschutt beton '
  + 'cliente contenedor cosecha demolition entsorgung erde factura fahrzeug finca flotte '
  + 'grunland kran lieferung maquinaria mieten mulde obra parcela presupuesto recogida '
  + 'reparatur riego sammlung schrott servicio siembra transport umzug vertrag wartung '
  + 'zaun abfall bagger container dumper ernte gully haufen kies lager').split(' ');

// Lehmer needs a 64-bit intermediate that JS doesn't give you through imul;
// the product goes negative, the modulus goes negative, and rnd() returns >1.
// mulberry32 stays inside 32 bits by construction.
let seed = 1234567;
const rnd = () => {
  seed = (seed + 0x6d2b79f5) >>> 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const pick = (arr) => arr[(rnd() * arr.length) | 0];

// The long tail: surnames, street names, contacts. Real corpora are mostly
// these, and they're what makes any two rows distinguishable. Leaving them out
// was why the last run measured a 0.53 noise floor.
const syl = 'ber hol mann schmi gru fel dorf bach stein wal ric ort lin gar cas tor mor vil'.split(' ');
const properNoun = () => pick(syl) + pick(syl) + (rnd() < 0.4 ? pick(syl) : '');

const sourceText = new Array(N);
const sourceRec = new Array(N);
let tBuild = performance.now();
for (let i = 0; i < N; i++) {
  const jargon = Array.from({ length: 4 + ((rnd() * 5) | 0) }, () => pick(vocab)).join(' ');
  const rec = {
    name: `${properNoun()} ${properNoun()} ${pick(suffix)}`,
    industry: pick(industries),
    notes: `${jargon} ${properNoun()}strasse ${(rnd() * 90 + 1) | 0} ${pick(cities)} `
         + `kontakt ${properNoun()} ref ${i.toString(36)}`,
  };
  sourceText[i] = `${rec.name} ${rec.industry} ${rec.notes}`;
  sourceRec[i] = rec;
  idx.add(i, rec);
}
tBuild = performance.now() - tBuild;

console.log(`  indexed ${idx.size.toLocaleString()} records in ${(tBuild / 1000).toFixed(1)}s ` +
            `(${(tBuild / N * 1000).toFixed(0)}µs/record)`);
console.log(`  memory: ${(idx.tags.byteLength / 1024 / 1024).toFixed(1)}MB of tags ` +
            `(${fh.words * 4} bytes/record)`);

// Proper retrieval test: take a known record, mangle it the way a user would
// (typos, dropped words, reordering), and see whether the original comes back.
// This gives real ground truth, unlike planting one obvious ringer.
function mangle(text) {
  const parts = text.split(' ').filter(() => rnd() > 0.3);          // drop ~30%
  for (let i = 0; i < parts.length; i++) {
    if (rnd() < 0.25 && parts[i].length > 3) {                       // typo 25%
      const p = 1 + ((rnd() * (parts[i].length - 2)) | 0);
      parts[i] = parts[i].slice(0, p) + parts[i].slice(p + 1);
    }
  }
  return parts.sort(() => rnd() - 0.5).join(' ');                    // shuffle
}

const PROBES = 300;
const probeIds = Array.from({ length: PROBES }, () => (rnd() * N) | 0);

function evaluate(index, makeQuery, label) {
  let h1 = 0;
  let h10 = 0;
  const t = performance.now();
  for (const id of probeIds) {
    const res = index.search(makeQuery(id), { limit: 10 });
    if (res[0]?.id === id) h1++;
    if (res.some((r) => r.id === id)) h10++;
  }
  const per = (performance.now() - t) / PROBES;
  console.log(`  ${label.padEnd(26)} recall@1 ${(h1 / PROBES * 100).toFixed(1).padStart(5)}%   ` +
              `recall@10 ${(h10 / PROBES * 100).toFixed(1).padStart(5)}%   ${per.toFixed(1)}ms/query`);
  return { r1: h1 / PROBES, r10: h10 / PROBES, per };
}

// Mode A — free-text search box. Query and documents must be vectorised the
// same way, so this index uses no field weights.
const flat = new FlyIndex({ capacity: N });
for (let i = 0; i < N; i++) flat.add(i, sourceText[i]);
const a = evaluate(flat, (id) => mangle(sourceText[id]), 'free text → flat index');

// Mode B — the trap. Documents weighted by field, query arriving as a plain
// string. The query has no idea the name field counted triple, so the two
// vectors live in different spaces and recall collapses. This cost ~55 points
// the first time I measured it and it looks exactly like a broken algorithm.
const b = evaluate(idx, (id) => mangle(sourceText[id]), 'free text → weighted index');

// Mode C — field-weighted on both sides, which is the deduplication case:
// an incoming record matched against existing ones.
const c = evaluate(idx, (id) => ({
  name: mangle(sourceRec[id].name),
  industry: sourceRec[id].industry,
  notes: mangle(sourceRec[id].notes),
}), 'fields → weighted index');

console.log(`  (queries: ~30% of words dropped, 25% of the rest typo'd, word order shuffled)`);

check(a.per < 15, `query under 15ms (${a.per.toFixed(1)}ms)`);
check(a.r1 > 0.85, `free-text recall@1 above 85% (${(a.r1 * 100).toFixed(1)}%)`);
// Mode C lands below Mode A, which is counterintuitive but correct: `name` is
// three words long and carries triple weight, so corrupting it corrupts most
// of the vector. Heavy weights on short fields amplify noise. Keep weights
// modest (2x at most) unless the field is clean and canonical.
check(c.r1 > 0.70, `field-matched recall@1 above 70% (${(c.r1 * 100).toFixed(1)}%)`);
check(b.r1 < a.r1, 'mismatched vectorisation measurably hurts (documented gotcha)');

const cal = idx.calibrate(5000);
console.log(`  noise floor on this corpus: mean ${cal.mean.toFixed(3)}, ` +
            `p99 ${cal.p99.toFixed(3)} → minScore ${cal.suggestedMinScore.toFixed(2)}`);

console.log('\n--- persistence round-trip ---');
const small = new FlyIndex();
small.add('a', 'container rental mannheim');
small.add('b', 'hotel property management system');
const rows = small.export();
const restored = new FlyIndex().load(rows);
const before = small.search('container rental', { limit: 1 })[0];
const after = restored.search('container rental', { limit: 1 })[0];
check(before.id === after.id && Math.abs(before.score - after.score) < 1e-12,
  'export → load preserves results exactly');

console.log(`\nfingerprint: ${fh.fingerprint()}  (store this alongside your tags)\n`);
