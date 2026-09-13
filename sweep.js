import { FlyHash } from './flyhash.js';
import { DIM } from './featurize.js';
import { FlyIndex } from './index.js';

let seed = 1234567;
const rnd = () => {
  seed = (seed + 0x6d2b79f5) >>> 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const pick = (a) => a[(rnd() * a.length) | 0];

const vocab = ('acero albaran alquiler anfrage angebot auftrag baustelle bauschutt beton cliente '
  + 'contenedor cosecha demolition entsorgung erde factura fahrzeug finca flotte grunland kran '
  + 'lieferung maquinaria mieten mulde obra parcela presupuesto recogida reparatur riego sammlung '
  + 'schrott servicio siembra transport umzug vertrag wartung zaun abfall bagger container dumper '
  + 'ernte gully haufen kies lager').split(' ');
const syl = 'ber hol mann schmi gru fel dorf bach stein wal ric ort lin gar cas tor mor vil'.split(' ');
const cities = ['Mannheim', 'Heidelberg', 'Albacete', 'Riopar', 'Hamburg', 'Valencia'];
const industries = ['skip hire', 'agricultural services', 'hotel software', 'logistics',
  'demolition', 'landscaping', 'catering', 'fleet maintenance', 'satellite operations'];
const suffix = ['GmbH', 'SL', 'AG', 'Ltd', 'eK'];
const noun = () => pick(syl) + pick(syl) + (rnd() < 0.4 ? pick(syl) : '');

const N = 20_000;
const corpus = [];
for (let i = 0; i < N; i++) {
  const jargon = Array.from({ length: 4 + ((rnd() * 5) | 0) }, () => pick(vocab)).join(' ');
  corpus.push(`${noun()} ${noun()} ${pick(suffix)} ${pick(industries)} ${jargon} `
    + `${noun()}strasse ${(rnd() * 90 + 1) | 0} ${pick(cities)} kontakt ${noun()} ref ${i.toString(36)}`);
}

function mangle(text, dropRate, typoRate) {
  const parts = text.split(' ').filter(() => rnd() > dropRate);
  for (let i = 0; i < parts.length; i++) {
    if (rnd() < typoRate && parts[i].length > 3) {
      const p = 1 + ((rnd() * (parts[i].length - 2)) | 0);
      parts[i] = parts[i].slice(0, p) + parts[i].slice(p + 1);
    }
  }
  return parts.sort(() => rnd() - 0.5).join(' ');
}

const PROBES = 300;
const probeIds = Array.from({ length: PROBES }, () => (rnd() * N) | 0);
const probes = probeIds.map((id) => ({ id, q: mangle(corpus[id], 0.3, 0.25) }));

console.log('\n cells  sparsity  fanIn   bits  bytes/rec  recall@1  recall@10  build   query');
console.log(' ' + '-'.repeat(76));

const results = [];
for (const cells of [2048, 8192, 16384]) {
  for (const sparsity of [0.05, 0.02, 0.01]) {
    for (const fanIn of [6, 12]) {
      const hasher = new FlyHash({ inputDim: DIM, cells, sparsity, fanIn });
      const idx = new FlyIndex({ hasher, capacity: N });

      const tb = performance.now();
      for (let i = 0; i < N; i++) idx.add(i, corpus[i]);
      const build = performance.now() - tb;

      let h1 = 0;
      let h10 = 0;
      const tq = performance.now();
      for (const p of probes) {
        const r = idx.search(p.q, { limit: 10 });
        if (r[0]?.id === p.id) h1++;
        if (r.some((x) => x.id === p.id)) h10++;
      }
      const query = (performance.now() - tq) / PROBES;

      const row = {
        cells, sparsity, fanIn, bits: hasher.k, bytes: hasher.words * 4,
        r1: h1 / PROBES, r10: h10 / PROBES, build, query,
      };
      results.push(row);
      console.log(
        ` ${String(cells).padStart(5)}  ${sparsity.toFixed(2).padStart(8)}  ` +
        `${String(fanIn).padStart(5)}  ${String(hasher.k).padStart(5)}  ` +
        `${String(hasher.words * 4).padStart(9)}  ` +
        `${(row.r1 * 100).toFixed(1).padStart(7)}%  ${(row.r10 * 100).toFixed(1).padStart(8)}%  ` +
        `${(build / N * 1000).toFixed(0).padStart(4)}µs  ${query.toFixed(1).padStart(5)}ms`
      );
    }
  }
}

const best = [...results].sort((a, b) => b.r1 - a.r1)[0];
console.log(`\n best recall@1: cells=${best.cells} sparsity=${best.sparsity} fanIn=${best.fanIn}`
  + ` → ${(best.r1 * 100).toFixed(1)}% @ ${best.bytes} bytes/record\n`);
