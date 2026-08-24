import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";


const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, "..");
const rawDir = resolve(projectRoot, "data", "phase2", "raw");
const includeReference = process.argv.includes("--include-reference");

const records = [
  {
    role: "positive_peaks",
    name: "ENCFF148JKK.bed.gz",
    md5: "d5183b4c65853c9dea2b299fd17f2562",
    bytes: 262784,
    url: "https://encode-public.s3.amazonaws.com/2017/07/20/e21418dd-7e81-4317-97f6-e6aad5e5cea4/ENCFF148JKK.bed.gz",
  },
  {
    role: "positive_peaks_conservative_sensitivity",
    name: "ENCFF875JHB.bed.gz",
    md5: "8e03c2d46eded620a080b6b726df9c6e",
    bytes: 197526,
    url: "https://encode-public.s3.amazonaws.com/2017/07/20/1ff8f628-c133-48c6-a194-aef1cf873ab5/ENCFF875JHB.bed.gz",
  },
  {
    role: "accessible_negative_pool",
    name: "ENCFF333TAT.bed.gz",
    md5: "0f7a6c13e23c2e3fc8716153a89ed481",
    bytes: 7067871,
    url: "https://encode-public.s3.amazonaws.com/2021/03/16/f9c5229c-df01-48d5-b8df-10d060edd52f/ENCFF333TAT.bed.gz",
  },
  {
    role: "ctcf_positive_control",
    name: "ENCFF769AUF.bed.gz",
    md5: "7d086cac19c5311a77b7e21e3d931435",
    bytes: 919491,
    url: "https://encode-public.s3.amazonaws.com/2020/09/26/e2f6f331-f487-451e-ad67-cf3d22f50b31/ENCFF769AUF.bed.gz",
  },
  {
    role: "blacklist",
    name: "ENCFF356LFX.bed.gz",
    md5: "393688b4f06c9ce26165d47433dd8c37",
    bytes: 8211,
    url: "https://encode-public.s3.amazonaws.com/2020/05/05/bc5dcc02-eafb-4471-aba0-4ebc7ee8c3e6/ENCFF356LFX.bed.gz",
  },
  {
    role: "reference_fasta",
    name: "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz",
    md5: "a08035b6a6e31780e96a34008ff21bd6",
    bytes: 872949833,
    url: "https://encode-public.s3.amazonaws.com/2015/12/03/a7fea375-057d-4cdc-8ccd-0b0f930823df/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz",
    reference: true,
  },
];


async function md5File(path) {
  const hash = createHash("md5");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}


async function download(record) {
  const target = resolve(rawDir, record.name);
  if (!target.startsWith(rawDir + "\\") && target !== rawDir) {
    throw new Error(`unsafe target path: ${target}`);
  }
  try {
    const existing = await md5File(target);
    if (existing === record.md5) {
      const info = await stat(target);
      console.log(`SKIP ${record.role} ${record.name} bytes=${info.size} md5=${existing}`);
      return;
    }
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }

  const part = `${target}.part`;
  let offset = 0;
  try {
    offset = (await stat(part)).size;
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }

  const headers = { "User-Agent": "RC-Attribution-Phase2/0.1" };
  if (offset > 0) headers.Range = `bytes=${offset}-`;
  const response = await fetch(record.url, {
    redirect: "follow",
    headers,
    signal: AbortSignal.timeout(300_000),
  });
  if (!response.ok || !response.body) {
    throw new Error(`${record.role}: HTTP ${response.status}`);
  }

  const canResume = offset > 0 && response.status === 206;
  if (offset > 0 && !canResume) {
    console.log(`RESTART ${record.role} ${record.name}: server ignored Range request`);
    await rm(part, { force: true });
    offset = 0;
  } else if (canResume) {
    console.log(`RESUME ${record.role} ${record.name} from=${offset}`);
  }
  await pipeline(
    Readable.fromWeb(response.body),
    createWriteStream(part, { flags: canResume ? "a" : "w" }),
  );
  const downloadedSize = (await stat(part)).size;
  if (downloadedSize !== record.bytes) {
    throw new Error(`${record.role}: size mismatch expected=${record.bytes} actual=${downloadedSize}`);
  }
  const actual = await md5File(part);
  if (actual !== record.md5) {
    await rm(part, { force: true });
    throw new Error(`${record.role}: MD5 mismatch expected=${record.md5} actual=${actual}`);
  }
  await rename(part, target);
  const info = await stat(target);
  console.log(`OK ${record.role} ${record.name} bytes=${info.size} md5=${actual}`);
}


await mkdir(rawDir, { recursive: true });
const selected = records.filter((record) => includeReference || !record.reference);
for (const record of selected) await download(record);
console.log(`downloaded_or_verified=${selected.length}`);
