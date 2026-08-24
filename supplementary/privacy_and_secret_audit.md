# Privacy and secret audit

Audit date: 2026-08-24

## Scope

The audit searched text-like files in project configuration, protocols, source code, scripts, metadata, selected results, README files, requirements, and Methods drafts. Large genomic data, binary model checkpoints, virtual environments, caches, and document binaries were not parsed as credential stores; they were excluded from the public release candidate.

Search patterns included case-insensitive variants of `password`, `token`, `api_key`, `secret`, and `credential`, plus email-like strings and Windows/macOS/Linux absolute user paths.

## Findings

| Finding | Status | Action |
|---|---|---|
| Password assignment or literal | Not found | None |
| API key or token assignment | Not found | None |
| Secret/credential assignment | Not found | None |
| Email address | Not found in scanned text | None |
| Windows user path | One hit | `methods_metadata_extraction_v1.md` contains the original local Windows home path; the file was excluded from `github_release` |
| Other absolute home paths | No verified release hit | Rechecked after packaging |

No credential or secret was identified in the scanned source material. This is a pattern-based audit, not a guarantee against secrets hidden in unparsed binary files.

## Files and categories excluded from the release

- `.venv/`, `.pytest_cache/`, `__pycache__/`, compiled Python files, Matplotlib caches, IDE settings, and temporary work directories.
- Raw GRCh38 FASTA files, compressed reference archives, download chunks, source BED files, and other large raw data.
- Model checkpoints and checkpoint directories.
- Large prediction/attribution intermediates that are not required to rebuild the main figures.
- Experimental reports in DOCX, QA render images, local scratch folders, temporary ZIPs, and unrelated logs.
- The unsanitized Methods metadata extraction file containing the local Windows path.

## Release scan result

The assembled `github_release` contains no detected password, API-key, secret, credential, email-address, or absolute Windows user-home string. Placeholder repository and author fields remain clearly marked and are not secrets.
