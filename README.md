# FlyHash

Fuzzy record matching modelled on the fruit fly olfactory circuit. No training
step, no embedding API, no vector database. 256 bytes per record.

Based on Dasgupta, Stevens & Navlakha, *A neural algorithm for a fundamental
computing problem*, Science 2017.

[Live demo](https://claude.ai/code/artifact/ee1628cc-8d29-436e-9c25-71096107dfe4) — runs the real implementation in the browser.

## Install

No dependencies, so installing straight from GitHub works fine:

```bash
npm install github:blade361/flyhash
```

Or copy `flyhash.js`, `featurize.js` and `index.js` into your project. They're
plain ESM with no build step.

## What it does

Turns text into a sparse binary tag. Similar text lights up overlapping bits,
so "how similar are these" becomes "how many bits do they share" — a bitwise
AND and a popcount.

```js
import { FlyIndex } from './index.js';

const index = new FlyIndex();
index.add(1, 'Orban Containerdienst GmbH, Mannheim, skip hire');
index.add(2, 'Klinger Agrarservice, Heidelberg');

index.search('containerdienst manheim');
// [ { id: 1, score: 0.62, meta: null }, ... ]
```

## Measured

100,000 synthetic CRM records, Node 22, single thread. Queries had ~30% of
words dropped, 25% of the remainder given a typo, and word order shuffled.

| | |
|---|---|
| Index size | 256 bytes/record (32 MB total) |
| Indexing | ~215 µs/record |
| Query | 11–14 ms over 100k records |
| recall@1 | 89.3% |
| recall@10 | 96.3% |

Run `node bench.js` to reproduce, `node sweep.js` for the parameter grid.

## Parameters

Defaults came from the sweep in `sweep.js`, not from taste.

| Option | Default | Notes |
|---|---|---|
| `cells` | 2048 | More cells discriminate better and cost linearly. 8192 bought ~5 points of recall@1 at 4x the storage. |
| `sparsity` | 0.05 | Fraction of cells left on. Dropping to 0.01 cost 30+ points. |
| `fanIn` | 6 | Inputs sampled per cell. Beat 12 at every cell count tested, and it's what a real Kenyon cell gets. |
| `seed` | 20260913 | **Part of your data format.** Changing it invalidates every stored tag. |

Scaling up is one line:

```js
const hasher = new FlyHash({ inputDim: DIM, cells: 8192, sparsity: 0.05 });
const index = new FlyIndex({ hasher });
```

## Three things that will bite you

**1. Documents and queries must be vectorised identically.**

Indexing with field weights and then querying with a plain string costs about
40 points of recall (89.3% → 50.3%). The two vectors end up in different
spaces. It looks exactly like a broken algorithm. Either use `FlyIndex` with
plain strings on both sides, or pass the same field object to `search()` that
you passed to `add()`.

**2. Heavy weights on short fields amplify noise.**

Weighting a three-word `name` field at 3x means corrupting the name corrupts
most of the vector. Field-weighted matching scored *below* flat matching for
this reason (76.3% vs 89.3%). Keep weights at 2x or less, or repeat the field
in the text instead — `wyvern-search.js` does the latter.

**3. The noise floor is not zero, and it depends on your corpus.**

Two unrelated records share bits by chance. With the defaults that's around
0.05 for genuinely independent text, but on a homogeneous corpus where every
row uses the same vocabulary it measured 0.12 mean and 0.55 at p99. A score of
0.1 means *nothing*, not "a weak match".

```js
index.calibrate();
// { mean: 0.123, p50: 0.108, p99: 0.549, max: 0.71, suggestedMinScore: 0.60 }
```

Run this once on real data and set `minScore` above the p99 it reports.

## Tag length varies with input length

`k` is a ceiling, not a guarantee. A tag only sets bits for cells that actually
fired, and short text doesn't reach many cells:

| Input | Bits set (k = 102) |
|---|---|
| 1 word | 18 |
| 5 words | 92 |
| 12 words | 102 |
| 30 words | 102 |

This matters. An earlier version padded short tags up to `k` with
zero-activation cells, which picks them in index order — so every short input
shared the same block of low-index bits. Unrelated one-word queries scored 0.28
against each other and searching "riopar" returned "Containerdienst Süd" first.
Similarity is therefore cosine over the bits actually present,
`overlap / sqrt(|a| · |b|)`, not `overlap / k`.

## API

### `new FlyHash({ inputDim, cells, sparsity, fanIn, seed })`

- `hash(vec)` → `Uint32Array` tag
- `activate(vec)` → raw cell activations, for visualisation
- `winners(vec)` → indices of firing cells
- `compare(a, b)` → 0..1
- `config()` / `fingerprint()` → store alongside your tags to detect drift
- `FlyHash.encode(tag)` / `FlyHash.decode(b64)` → 344-char base64
- `FlyHash.count(tag)` → bits set

### `new FlyIndex({ hasher, weights, capacity })`

- `add(id, text | fields, meta?)` → tag
- `addTag(id, tag, meta?)` — the path used when loading from a database
- `search(query, { limit, minScore })` → `[{ id, score, meta }]`
- `export()` → `[{ id, tag }]` with base64 tags
- `load(rows, metaOf?)` — rehydrate without re-vectorising
- `calibrate(samples?)` → noise floor statistics

### `featurize.js`

- `vectorize(text, { dim, gram, gramWeight })`
- `vectorizeFields(fields, weights, opts)`
- `tokens(text)`, `normalise(text)`

## Using it behind an HTTP API

A worked Express + Supabase integration ships separately (it belongs in the
app repo, not here, since it knows your schema). The shape:

```sql
alter table contacts add column fly_tag text;
alter table contacts add column fly_fp  text;
create index on contacts (fly_fp);
```

```js
import { createSearch } from './lib/search.js';

const search = createSearch(supabase);
app.use('/api/crm', requireRole('crm'), search.router);
await search.warm();          // loads stored tags, no re-hashing
await search.reindexAll();    // one-off migration
```

Endpoints: `GET /search?q=` and `POST /duplicates` (pre-insert dedupe check).

`fly_fp` holds the hasher fingerprint. If you change any hash parameter the
fingerprint changes, `warm()` skips and counts the stale rows, and you know to
reindex. Without it you get silent garbage after a config change, which is a
miserable bug to chase.

## What it is not

It has no idea what words mean. "Skip hire" and "waste disposal" share no
characters and score at the noise floor. This is fast lexical matching over
shared vocabulary, spelling and typos.

The sensible use is as a cheap first pass: narrow thousands of rows to fifty in
a few milliseconds with no external call, then send those fifty to an embedding
model or an LLM if you need actual meaning.

## Files

```
flyhash.js        core — projection, winner-take-all, encode/decode
featurize.js      text → vector; tokenisation, n-grams, field weighting
index.js          searchable index, export/load, calibrate
bench.js          test suite and 100k benchmark
sweep.js          parameter grid
demo/index.html   self-contained interactive demo, no build step
```

Zero dependencies. ESM. Runs in Node 18+ and in the browser.

## Licence

MIT. The algorithm is from the Dasgupta et al. paper cited above; this is an
independent implementation.
