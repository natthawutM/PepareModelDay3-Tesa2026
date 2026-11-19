# Day 3 segmentation and latency search - Windows

Scope: classification training was explicitly deferred. Optimize segmentation/counting and reduce latency on Windows; Pi measurement was deferred by the user. This is a local search, not proof of a global optimum.

## Results

Median of three interleaved complete CLI runs, CPU, four threads, no warm-up. CLI time includes process/model loading, image decoding, inference, native-resolution uint16 PNG masks, JSON and overlay PNGs (preview longest side 1280). It excludes ground-truth evaluation and HTTP/MQTT. Compare candidates on the same dataset: startup cost is divided by 33 or 5 images respectively.

| Candidate | Validation set | IoU | Counting | CLI seconds/image | Median within-process seconds/image |
|---|---|---:|---:|---:|---:|
| original | legacy | 77.85% | 94.80% | 0.0767 | 0.0414 |
| original | triangle | 84.93% | 93.33% | 0.4005 | 0.2667 |
| accuracy | legacy | 79.08% | 95.48% | 0.0748 | 0.0412 |
| accuracy | triangle | 88.29% | 97.14% | 0.3490 | 0.2266 |
| speed | legacy | 76.87% | 94.36% | 0.0617 | 0.0282 |
| speed | triangle | 86.07% | 97.14% | 0.3379 | 0.2082 |

Recommended starting candidate: `A0_day3_search_accuracy`. On triangle it has appreciably better masks than the fast candidate for about 11 ms/image additional Windows CLI time. Keep `A0_day3_search_speed` for Pi comparison if latency dominates. `A0_day3_search_balanced` is an additional intermediate candidate, not part of the three-candidate repeated timing comparison.

The classifier file is unchanged in all packages. Mask changes still change classifier crops and matching: triangle class-aware F1 changed from 63.49% to 59.02% for both finalists. This search does not claim a better classification score or a best total official Day 3 score.

## Search and stopping

- 72 measured candidate/epoch configurations: original full-frame, three ROI margins with full/compact locators; coordinate watershed search; nine fine-tune epochs over three learning-rate/boundary-loss recipes; input sizes 384/448/576/640; two selective INT8 recipes; further coordinate search for the fast 384 model.
- Both validation groups have equal weight in selection, so the 33-image set does not drown out the five-image triangle group.
- Temporary selection proxy: mean over validation groups of `7.5 * (IoU + Counting)`. This is not a verified official scoring formula. Classification, ranked Pi latency and MQTT are not assigned invented scores.
- Candidate guard: neither IoU nor Counting may drop more than 0.02 absolute against the original baseline on either group.
- Full-size coordinate search stopped after four rounds, with the final three rounds each improving less than 0.05 proxy points. The 384 search stopped after three such rounds. The best candidate is retained even if later trials regress.
- Fine-tuning used only Train: 151 reviewed original images plus 11 Day 3 grid/square images, sampled 75/25. Three recipes used learning rates 1e-5, 3e-6, 1e-6 and boundary weights 1.0, 1.5, 0.75. Encoder parameters and encoder BN statistics were frozen. Each recipe stopped after three epochs without beating the retained champion; none was selected. No classifier training was performed.
- INT8 encoder-only and all Conv except the output head both reduced IoU and did not accelerate this Windows run. They were rejected. They are not assumed to have the same behavior on ARM.
- The fast model uses original A0 weights exported at 384; the accuracy model uses original A0 at 512 with improved watershed. Do not describe either selected model as a newly trained or INT8 model.

## Selected settings

- `A0_day3_search_accuracy`: input 512, watershed `{"boundary": 0.3, "foreground": 0.4, "min_area": 8, "min_marker": 22}`.
- `A0_day3_search_balanced`: input 512, watershed `{"boundary": 0.375, "foreground": 0.45, "min_area": 12, "min_marker": 16}`.
- `A0_day3_search_speed`: input 384, watershed `{"boundary": 0.275, "foreground": 0.35, "min_area": 4, "min_marker": 13}`.

## Runtime changes and verification

`post.py` compacts instance IDs on the small watershed grid before nearest-neighbour upscaling, avoiding full-resolution scans and float32 intermediates. Downscaling retains the previous path. Verified exact ID-mask parity on upscale, downscale, empty and touching-bean cases.

`infer.py` supports optional compact ROI localization through `roi_locator_side` in a manifest; the default remains unchanged. Actual five-image default ROI PNG masks and JSON were compared against the pre-change outputs and were identical. The selected finalists use full-frame inference.

All three packaged candidates passed the existing Day 3 client output validator on both validation sets (38 images each), checking PNG/JSON IDs, class names and polygons. All 18 repeated benchmark runs produced exactly repeatable IoU/Counting. Python compilation and Git whitespace checks passed.

## Data and limitations

Valid legacy is 33 images / 419 instances, including 15 pile images; it is not entirely piles. Triangle is five images / 30 instances. Train/Valid byte hashes were checked for overlap and none were found. Different arrangements may contain the same physical beans, which byte hashes cannot rule out. Both validation sets have been used repeatedly for tuning, so these are selection results, not independent final-test estimates. Test was not used for model selection.

These are Windows timings only. Neither the <=1 second Pi target nor an official latency ranking is verified. No organizer server or MQTT broker was contacted. Final deployment should use the actual Pi measurements and the announced official scoring/timing definition.

## Reproduction

The original benchmark reference is pinned to Git commit `8f82104209d7c6c6f215152685948db49e678caa`. The scripts require the existing training environment, `../overnight_01/src`, reviewed Train and Day 3 Train. Set PYTHONPATH to the installed runtime dependency directory as appropriate.

```powershell
python optimize_day3.py --out C:/DAY3/runs/day3_score_latency_search_v1
python optimize_day3_speed.py --out C:/DAY3/runs/day3_score_latency_search_v1
python optimize_day3_lowres.py --out C:/DAY3/runs/day3_score_latency_search_v1
python benchmark_day3_final.py --out C:/DAY3/runs/day3_score_latency_search_v1/repeated --repeats 3 --candidate original accuracy speed
```

The first script starts a new search history; use a new output folder for a separate experiment. Raw full outputs and calibration/model intermediates are in `C:/DAY3/runs/day3_score_latency_search_v1`. Tracked CSV/JSON files here preserve the comparison and selected model hashes. No commit or push was performed by this search.
