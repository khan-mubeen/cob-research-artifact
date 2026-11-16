# Anonymity audit

This is the separate review copy. Changes were confined to this directory; the original repository was not accessed or modified. No commits, pushes or history rewrites were performed.

## Removed or withheld

- Nine private meeting notes, the planning PDF and the earlier internal repository audit.
- Noncanonical pilot outputs, identifying operational ledgers and two local debugging files.
- The raw DataCo customer CSV. It was a qualitative reference rather than an input to synthesis; its public source citation remains in the methodology.
- Known contributor names, emails, usernames, user-specific directories and project account identifiers.
- Hostnames, scheduler identifiers, evaluation/recording timestamps and identifying Git references in the retained measurement metadata.
- Original result and selection hashes where they would incorrectly certify files changed by anonymization. Such fields are marked unknown.

Research source, synthetic inputs, model revisions, per-row decisions, metrics, selected epochs and measured timings remain available. The metadata changes are not new measurements. The original execution provenance is not claimed to be byte-identical to this copy.

Third-party source records, scientific observation dates, model references, the MIT notice and the Source Sans 3 licence and font files were retained. Public source attribution was not treated as an author identifier.

## Verification

- All 162 official result and selection records passed payload-equality checks during redaction, including their selected epochs.
- The report rebuilt successfully from this copy: 48 chart files were generated.
- All 19 comparable tables matched the retained published tables.
- All 24 rebuilt PNG figures matched the published figures pixel for pixel.
- Published and rebuilt latency tables passed all 14 consistency checks.
- Auction tests: 11 passed. Schema tests: 18 passed. Reporting regressions: four passed.
- All Python files parsed successfully.
- A scan of 1,236 retained text files, approximately 5 GB, found no matches for the known contributor identity strings or project account identifier.
- All 24 scientific PDFs had only generic Matplotlib creator/producer information, with no author fields or XMP metadata. The 106 PNGs had only software and resolution metadata. The SVG metadata identified Matplotlib only. No binary rewrite was needed.
- The font licence and bundled font files were unchanged.

A broad binary metadata rewrite was rejected by automatic approval review because it could damage figures or attribution. Read-only inspection established that the existing figure metadata did not require that operation; it was not performed.

The full historical corpus, GPU training, live services and cluster campaigns were not rerun. They still require undistributed inputs or site-specific tools. Redacted provenance fields must not be used as original authorization or integrity evidence. Historical freeze/hash checks that depend on withheld ledgers are not supported by this review copy; their source is retained rather than weakened to make them pass.

## Packaging

Use the supplied command from this directory:

    python package_review.py

It builds submission/chain-of-bid-review.zip from the current files, without a commit. It excludes Git metadata, private notes, customer source data, local credentials, environments, runtime logs and generated output. ZIP entry timestamps and permissions are normalized, the archive comment is empty, and SHA256SUMS.txt records the packaged file hashes.

The packager is a distribution tool, not a general anonymizer for arbitrary future edits. Recheck content if new files or records are added. It refuses to overwrite an existing archive; use --output with another filename under submission/ for a new version.

Do not use git archive of HEAD for the review ZIP: its archive comment contains the commit identifier. Do not submit the working directory or a Git clone.

## Remaining risks

- The local .git directory still contains identifying history. It is intentionally unchanged and excluded from the review archive.
- The project name, source code, scientific results and retained hashes may be matched to an already-public project. Removing direct identifiers cannot guarantee unlinkability to public material.
- Public provider names, geographical dataset details, execution-profile labels and third-party model revisions remain where needed for attribution or reproducibility. They are not statements of author affiliation.
- Scientific source dates and some historical filenames are retained. They describe data or version lineage rather than claiming anonymous original provenance.
- Third-party licences and redistribution terms still apply. Anonymization does not grant new distribution rights.
- The audit targets the distributed files, not the paper manuscript, external websites, author accounts or venue-specific anonymity policies. Those need a final author review.

The supplied ZIP is the submission artifact. Keep the working Git directory and checking environment local.
