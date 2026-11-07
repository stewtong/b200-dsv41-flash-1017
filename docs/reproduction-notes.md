<!-- register: engineering notes | reader: benchmark reproducer | consumed: setup and result interpretation -->
# Reproduce with bounded ownership and explicit evidence

Run fixtures and new measurements into fresh directories outside the checkout. The launcher uses one owner for the node, records exact process and container identities, and keeps a safety collector active through teardown. A timed-out cleanup needs separate fresh evidence of process death and idle GPUs before another launch.

Verify the complete checksum manifest and file set before using a capture. Retain interrupted attempts outside this repository. Repair evidence packaging against the original captured bytes where possible; repeating inference changes the experiment and its request accounting.

Report first nonempty output, visible output and terminal response boundaries as defined by each experiment. Stream chunks can contain several tokens, so stream-window decode time is a proxy for token timing. A graph dispatch decision alone does not establish GPU execution.

Qualification fixtures, finite performance bursts, executable task checks and sustained-load tests establish different things. Preserve the stated scope of each result. Changes to logging, sampling or placement need explicit configuration identities and comparable controls before they support performance conclusions.

## Live node execution requires operator assembly

The commands in the README exercise fixtures and exported results. This release does not provide an end-to-end fresh-node deployment command, a deployment configuration example or a source-archive builder. The fixed-block measured path is the Python API `node_fixed_block.run(config)`; it requires prepared inputs and qualification records. The plan CLI does not dispatch a completed fixed comparison.

Before calling that API, an operator must assemble and hash a source toolkit with a root `MANIFEST.json` accepted by `toolkit_integrity.verify`, stage and qualify the pinned model, image and router, prepare tokenizer-bound inputs and registered profiles, and bind their receipts to the operational manifest. The toolkit manifest maps every included relative file path to its SHA-256; the verifier rejects unlisted files and symlinks. The public checkout alone does not supply the historical qualification receipts or this deployed toolkit manifest.

The inherited node preflight also requires four explicitly named prior containers to exist and be stopped, a specified prior collector timer to be inactive, and the excluded prior weight path to be absent. A clean node without those historical resources fails that preflight. Adapting this requirement to a clean node needs an explicit source change and new qualification; creating dummy resources or bypassing the check is not a validated reproduction path. The historical retirement requirement remains unchanged in this source snapshot.

The portable capture code separates the first nonempty output chunk from the first chunk containing non-whitespace text. An all-whitespace response has no first-visible-text timestamp. These definitions apply to new captures with this source version. Original result provenance still identifies each historical producer; later source fixes do not recalculate or reaccept historical measurements.
