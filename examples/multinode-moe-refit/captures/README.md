# Captured evaluation

- [before.request.json](before.request.json): exact original request.
- [before.response.json](before.response.json): all returned classifications and token usage.
- [before.report.json](before.report.json): findings produced by the unchanged extension pipeline.
- [provenance.json](provenance.json): source revisions, file hashes, deployment context, settings and recorded request timestamps.

This example was selected from a known upstream optimization PR. The original response was saved before the merged implementation was fetched. No PR explanation, expected classification, optimized code or previous answer was included in the original request. The existing context budget clipped an unrelated constructor reference; the export method and attached mapping helper were complete.

The preserved [after.request.json](after.request.json) received HTTP 503 responses. It has no captured classification and is not required for this demonstration. Recorded failures are retained in provenance; provider headers and identifiers are omitted from the exported response.

To retry that exact request, if needed:

```sh
curl --fail-with-body https://ai-gateway.vercel.sh/v1/evaluate \
  -H "Authorization: Bearer $AI_GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @examples/multinode-moe-refit/captures/after.request.json
```
