# Day 3 border refinement: triangle IoU over 90%

Scope: segmentation and Counting only; classifier weights are unchanged. Windows CPU, four threads, three interleaved complete CLI repeats for both models. These are validation selection results, not Pi timings or an official overall competition score.

| Model | Set | IoU | Counting | Temporary score /15 | CLI seconds/image | Median per-image seconds |
|---|---|---:|---:|---:|---:|---:|
| accuracy | legacy | 79.08% | 95.48% | 13.0921 | 0.0749 | 0.0412 |
| accuracy | triangle | 88.29% | 97.14% | 13.9071 | 0.3518 | 0.2346 |
| border | legacy | 79.08% | 95.48% | 13.0921 | 0.0739 | 0.0410 |
| border | triangle | 90.84% | 100.00% | 14.3129 | 0.3575 | 0.2308 |

Temporary score = 7.5 * IoU + 7.5 * Counting. Triangle improves from 13.9071 to 14.3129 /15 (+0.4058). The equally weighted two-validation-set proxy improves from 13.4996 to 13.7025 /15 (+0.2029). The official Day 3 scoring formula and ranked Pi latency are not known, so no official total is claimed.

## Cause and change

Four triangle images already exceeded 91% IoU. Mixed_triangle3 had a false-positive wooden tabletop strip at the upper image border. The extra instance reduced that image to 76.61% IoU and 85.71% Counting. Rejecting the strip raises it to about 89.38% IoU and 100% Counting; the five-image average reaches 90.84%.

The optional watershed rule rejects a connected foreground component only if it touches an image border, spans at least 70% of the parallel image dimension and occupies at most 20% of the perpendicular dimension. It applies symmetrically to horizontal and vertical borders. It does not use image names, ground truth, classes or an expected bean count. Filtering is disabled for every model without the new settings.

Five neighbouring geometry settings were compared on both validation sets and produced the same metrics. A conservative 70% span was selected rather than the more aggressive 50%/60% options. The weights, input size 512, and classifier are identical to A0_day3_search_accuracy. No extra neural-network pass was introduced.

Triangle complete-CLI timing changed from 0.3518 to 0.3575 s/image in the repeated run (+5.6 ms, within the overlapping run ranges). Within-process median changed from 0.2346 to 0.2308 s/image. Startup variation prevents claiming a reliable speed increase or a material slowdown.

## Verification and limitations

- All three repeats gave exactly the same IoU and Counting for each candidate/set.
- The existing Day 3 output validator passed on the final 33-image legacy and five-image triangle outputs; PNG IDs, JSON objects and polygons agree.
- A synthetic check confirms broad border strips are removed while compact beans touching the border remain. The disabled rule preserves the original behavior.
- Legacy Valid scores did not change: IoU 79.0781%, Counting 95.4837%. This is not a 90% result on piles.
- Triangle class-aware F1 rose incidentally from 59.02% to 60.00% after removing the false positive; classification was not trained or optimized.
- All validation images have been used for tuning. A very large, shallow real object along an image edge could satisfy the filter; validate on new arrangements/backgrounds before deployment. Disable the two border_strip settings if that input type matters.
- No Test data was used to choose the filter. No Pi, HTTP or MQTT benchmark was run. Confirm the <=1 second target on the actual Pi.

## Commands

```bash
python benchmark_all.py --split valid --model A0_day3_search_accuracy A0_day3_border_refined --data-root datasets/day3_triangle --out outputs_day3_border_pi
python benchmark_all.py --split valid --model A0_day3_search_accuracy A0_day3_border_refined --out outputs_day3_border_legacy_pi
```

Reproduce Windows search with optimize_day3_border.py. Repeated timing: benchmark_day3_final.py --out <folder> --repeats 3 --candidate accuracy border. No commit or push was performed.
