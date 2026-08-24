import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { access, copyFile, mkdir, rename, rm, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";


const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, "..");
const rawDir = resolve(projectRoot, "data", "phase2", "raw");
const chunkDir = resolve(rawDir, ".grch38_chunks");
const target = resolve(rawDir, "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz");
const legacyPart = `${target}.part`;

const URL = "https://encode-public.s3.amazonaws.com/2015/12/03/a7fea375-057d-4cdc-8ccd-0b0f930823df/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz";
const EXPECTED_BYTES = 872_949_833;
const EXPECTED_MD5 = "a08035b6a6e31780e96a34008ff21bd6";
const CHUNK_BYTES = 8 * 1024 * 1024;
const CONCURRENCY = 12;
const RETRIES = 6;


function safePath(path) {
  if (!path.startsWith(rawDir + "\\")) throw new Error(`unsafe path: ${path}`);
  return path;
}


async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}


async function md5File(path) {
  const hash = createHash("md5");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}


function chunkSpec(index) {
  const start = index * CHUNK_BYTES;
  const end = Math.min(EXPECTED_BYTES - 1, start + CHUNK_BYTES - 1);
  const length = end - start + 1;
  const name = String(index).padStart(5, "0");
  return {
    index,
    start,
    end,
    length,
    done: safePath(resolve(chunkDir, `${name}.chunk`)),
    part: safePath(resolve(chunkDir, `${name}.part`)),
  };
}


async function downloadChunk(spec) {
  if (await exists(spec.done)) {
    if ((await stat(spec.done)).size === spec.length) return "cached";
    await rm(spec.done, { force: true });
  }

  let offset = 0;
  if (await exists(spec.part)) {
    offset = (await stat(spec.part)).size;
    if (offset > spec.length) {
      await rm(spec.part, { force: true });
      offset = 0;
    }
  }

  for (let attempt = 1; attempt <= RETRIES; attempt += 1) {
    try {
      const rangeStart = spec.start + offset;
      const response = await fetch(URL, {
        headers: {
          Range: `bytes=${rangeStart}-${spec.end}`,
          "User-Agent": "RC-Attribution-Phase2/0.2",
        },
        signal: AbortSignal.timeout(300_000),
      });
      if (response.status !== 206 || !response.body) {
        throw new Error(`HTTP ${response.status}; expected 206`);
      }
      const contentRange = response.headers.get("content-range") || "";
      if (!contentRange.startsWith(`bytes ${rangeStart}-${spec.end}/`)) {
        throw new Error(`unexpected content-range: ${contentRange}`);
      }
      await pipeline(
        Readable.fromWeb(response.body),
        createWriteStream(spec.part, { flags: offset > 0 ? "a" : "w" }),
      );
      const observed = (await stat(spec.part)).size;
      if (observed !== spec.length) {
        offset = observed;
        throw new Error(`short chunk ${observed}/${spec.length}`);
      }
      await rename(spec.part, spec.done);
      console.log(`OK chunk=${spec.index} bytes=${spec.length}`);
      return "downloaded";
    } catch (error) {
      if (await exists(spec.part)) offset = (await stat(spec.part)).size;
      console.error(`RETRY chunk=${spec.index} attempt=${attempt} offset=${offset} error=${error.message}`);
      if (attempt === RETRIES) throw error;
      await new Promise((resolvePromise) => setTimeout(resolvePromise, attempt * 1000));
    }
  }
}


async function runPool(specs) {
  let cursor = 0;
  async function worker() {
    while (cursor < specs.length) {
      const spec = specs[cursor];
      cursor += 1;
      await downloadChunk(spec);
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, specs.length) }, worker));
}


async function assemble(specs) {
  const assembled = safePath(resolve(rawDir, ".GRCh38.assembled.part"));
  await rm(assembled, { force: true });
  const output = createWriteStream(assembled, { flags: "wx" });
  for (const spec of specs) {
    for await (const chunk of createReadStream(spec.done)) {
      if (!output.write(chunk)) await new Promise((resolvePromise) => output.once("drain", resolvePromise));
    }
  }
  await new Promise((resolvePromise, rejectPromise) => {
    output.end(resolvePromise);
    output.on("error", rejectPromise);
  });
  const observedBytes = (await stat(assembled)).size;
  if (observedBytes !== EXPECTED_BYTES) {
    throw new Error(`assembled size mismatch expected=${EXPECTED_BYTES} actual=${observedBytes}`);
  }
  const observedMd5 = await md5File(assembled);
  if (observedMd5 !== EXPECTED_MD5) {
    throw new Error(`assembled MD5 mismatch expected=${EXPECTED_MD5} actual=${observedMd5}`);
  }
  await rm(target, { force: true });
  await rename(assembled, target);
  console.log(`COMPLETE bytes=${observedBytes} md5=${observedMd5}`);
}


await mkdir(chunkDir, { recursive: true });
if (await exists(target)) {
  const currentMd5 = await md5File(target);
  if (currentMd5 === EXPECTED_MD5) {
    console.log(`SKIP complete reference md5=${currentMd5}`);
    process.exit(0);
  }
  throw new Error(`existing target has unexpected MD5: ${currentMd5}`);
}

const specs = Array.from(
  { length: Math.ceil(EXPECTED_BYTES / CHUNK_BYTES) },
  (_, index) => chunkSpec(index),
);

// Reuse the bytes already obtained by the older sequential downloader as the
// beginning of chunk 0. The original partial remains intact until copied.
if (await exists(legacyPart) && !(await exists(specs[0].done)) && !(await exists(specs[0].part))) {
  const legacyBytes = (await stat(legacyPart)).size;
  if (legacyBytes > 0 && legacyBytes < specs[0].length) {
    await copyFile(legacyPart, specs[0].part);
    console.log(`REUSED legacy_partial_bytes=${legacyBytes}`);
  }
}

await runPool(specs);
await assemble(specs);
