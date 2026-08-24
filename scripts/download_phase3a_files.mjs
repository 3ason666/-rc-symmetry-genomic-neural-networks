import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "node:stream/promises";


const scriptDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(scriptDir, "..");
const registryPath = join(root, "metadata", "phase3a_candidate_registry.json");
const rawDir = join(root, "data", "phase3", "raw");
const resultPath = join(root, "results", "phase3a_external_selection", "download_qc.json");


async function md5File(path) {
  const hash = createHash("md5");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}


async function downloadOne(item) {
  const extension = item.url.endsWith(".bed.gz") ? ".bed.gz" : "";
  const destination = join(rawDir, `${item.accession}${extension}`);
  let reused = false;
  try {
    const existingMd5 = await md5File(destination);
    if (existingMd5 === item.md5sum) reused = true;
  } catch {
    // Missing files are downloaded below.
  }
  if (!reused) {
    const partial = `${destination}.part`;
    let lastError;
    for (let attempt = 1; attempt <= 8; attempt += 1) {
      try {
        let partialBytes = 0;
        try {
          partialBytes = (await stat(partial)).size;
        } catch {
          // Start a new partial file.
        }
        if (partialBytes > item.file_size) {
          await rm(partial, { force: true });
          partialBytes = 0;
        }
        console.log(`${item.accession}: download attempt ${attempt}/8 from byte ${partialBytes}`);
        const headers = { "user-agent": "rc-attribution-phase3a/1.1" };
        if (partialBytes > 0) headers.range = `bytes=${partialBytes}-`;
        const response = await fetch(item.url, {
          headers,
          signal: AbortSignal.timeout(600000),
        });
        if (!response.ok || !response.body) throw new Error(`${item.accession}: HTTP ${response.status}`);
        const resumed = partialBytes > 0 && response.status === 206;
        if (partialBytes > 0 && !resumed) {
          partialBytes = 0;
          console.log(`${item.accession}: server did not honor Range; restarting this file`);
        }
        await pipeline(response.body, createWriteStream(partial, { flags: resumed ? "a" : "w" }));
        const partialStat = await stat(partial);
        if (partialStat.size !== item.file_size) {
          throw new Error(`${item.accession}: incomplete ${partialStat.size}/${item.file_size} bytes`);
        }
        const observedMd5 = await md5File(partial);
        if (observedMd5 !== item.md5sum) {
          await rm(partial, { force: true });
          throw new Error(`${item.accession}: MD5 mismatch ${observedMd5}; partial reset`);
        }
        await rm(destination, { force: true });
        await rename(partial, destination);
        lastError = undefined;
        break;
      } catch (error) {
        lastError = error;
        console.warn(`${item.accession}: attempt ${attempt} failed: ${error.message}`);
      }
    }
    if (lastError) throw lastError;
  }
  const observedMd5 = await md5File(destination);
  const fileStat = await stat(destination);
  return {
    accession: item.accession,
    role: item.role,
    path: destination.slice(root.length + 1).replaceAll("\\", "/"),
    bytes: fileStat.size,
    expected_bytes: item.file_size,
    md5: observedMd5,
    expected_md5: item.md5sum,
    md5_passed: observedMd5 === item.md5sum,
    size_passed: fileStat.size === item.file_size,
    reused,
  };
}


await mkdir(rawDir, { recursive: true });
await mkdir(dirname(resultPath), { recursive: true });
const registry = JSON.parse(await readFile(registryPath, "utf8"));
const selected = registry.confirmatory_tasks.flatMap((task) => task.files);
const rows = [];
for (const item of selected) rows.push(await downloadOne(item));
const report = {
  status: rows.every((row) => row.md5_passed && row.size_passed) ? "passed" : "failed",
  selected_file_count: rows.length,
  downloaded_bytes: rows.reduce((total, row) => total + row.bytes, 0),
  rows,
};
await writeFile(resultPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
