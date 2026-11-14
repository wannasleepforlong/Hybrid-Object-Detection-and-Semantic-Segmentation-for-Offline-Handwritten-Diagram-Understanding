# Hybrid Object Detection and Semantic Segmentation for Offline Handwritten Diagram Understanding

A reimplementation of Costa & Mello, *"Hybrid Object Detection and Semantic Segmentation for
Offline Handwritten Diagram Understanding"* (IEEE Access, vol. 14, 2026). The paper PDF is
included in this repository.

It turns an image of a hand-drawn flowchart (or finite automaton) into a graph, and exports it as
Mermaid, draw.io, a clean SVG redraw and an HTML report.

```
image ─► YOLOv7 (symbols, arrows, arrow head/tail, text)
      ─► DeepLabv3 (pixel mask of arrow strokes)
      ─► post-processing (duplicates, arrow ends, text → symbols)   paper §III-C
      ─► TrOCR (handwritten text)                                   paper §III-D
      ─► graph: arrow tail/head linked to symbols by overlap        paper §III-E
      ─► Mermaid · draw.io · SVG · HTML report
```

## Results (test sets)

| | Paper | This implementation |
|---|---|---|
| Symbol detection, FC_A (mean AP, IoU 0.8) | 99.4% | 98.4% |
| Symbol detection, FC_B | 99.7% | 98.9% |
| Arrow precision, FC_A (IoU 0.8) | 98.4% | 92.0% |
| Arrow precision, FA | 99.9% | 95.5% |
| Text CER, FC_A (lower is better) | 0.26 | **0.21** |
| Text WER, FC_A (lower is better) | 84% | **46%** |
| Full diagrams recognized, FC_A | 79.5% | 71.9% |
| Full diagrams recognized, FC_B | 90.3% | 89.3% |
| Full diagrams recognized, FA | 89.2% | 65.5% |

Full test sets: FC_A 171, FC_B 196, FA 84 images. Symbol detection is the mean over the real
symbol classes including arrow and text (without the arrow head/tail boxes, whose size the paper
does not specify). A diagram counts as recognized (paper criterion) when the number of symbols
matches, every symbol has IoU ≥ 0.8 with the ground truth, and every arrow's source and target
are correct.

## Our results in detail

**Detection and graph** (test sets, after post-processing, IoU 0.8 unless noted)

| | FC_A | FC_B | FA |
|---|---|---|---|
| Shapes (terminator, process, decision, data, connection / states), mean AP | 99.8% | 99.2% | 100.0% |
| Arrow AP | 95.9% | 98.8% | 94.1% |
| Text box AP | 94.7% | 97.4% | 93.5% |
| Arrow head / tail AP | 36.0% / 58.6% | 7.3% / 12.0% | 24.3% / 23.8% |
| mAP@0.5, all classes | 98.5% | 98.4% | 97.0% |
| Arrows linked to the right two symbols | 95.6% | 99.0% | 96.0% |
| Full diagrams recognized | 123/171 (71.9%) | 175/196 (89.3%) | 55/84 (65.5%) |

**Text recognition (TrOCR-base-handwritten, FC_A test, 1290 text crops)**

| Training data | WER | CER |
|---|---|---|
| FC_A text only | 50.0% | 0.247 |
| FC_A + 5,000 synthetic lines | **46.1%** | **0.214** |

**Arrow segmentation (DeepLabv3):** best validation IoU on arrow pixels 0.50 (FC_B val, 95 epochs,
early stopping).

**Training cost:** about 6 h on Kaggle with 2× T4 (YOLOv7 100 epochs per dataset on one GPU,
DeepLabv3 and TrOCR on the other). **Inference on CPU:** about 10 s for detection and segmentation,
plus about 7 s per text box for TrOCR (1.5–3.5 min per diagram).

## Things to notice

- **Works on clean data only.** All three datasets are pen strokes on a white background (FC_B also
  exists as clean flatbed scans). On a real phone photo of a notebook page (dim light, ruled lines,
  pencil) the pipeline found 1 of about 6 boxes, 5 of about 12 arrows and no connections. Ruled
  lines look like arrows to the detector. Simple image clean-up (lighting correction, binarization,
  ruled-line removal) did not fix it. Real-world use needs training data with that kind of noise
  (e.g. augmentation with ruled lines, lighting and perspective) or a different model for photos.
- **Word error rate is high (46%).** One wrong character makes the whole word wrong, and flowchart
  text is short and symbol-heavy (`v>0`, `f(V)`), so WER punishes small slips. Part of the error is
  **memorization**: FC_A uses the same 14 templates in training and test, so the model sometimes
  writes a training string instead of reading the image (`Compute f(V)` → `calculate g(x)`,
  `v>0` → `No`). Validation CER was 0.046 but test CER 0.214, because validation held out writers,
  not templates. Holding out templates and adding more varied synthetic text should help.
- **Only flowcharts and finite automata.** The detector knows the paper's symbol sets (terminator,
  process, decision, data, connection; state, final state). Architecture diagrams, sequence
  diagrams, BPMN or UML need new labelled data.
- **Full-diagram recognition is all-or-nothing.** One wrong arrow fails the whole diagram, so with
  about 9–10 arrows per FA diagram, 96% correct arrows still gives 65.5% recognized diagrams.
- **Arrow head / tail AP is not comparable with the paper.** These boxes are 3% squares around the
  arrow ends, chosen here because the paper does not say how it built them. At IoU 0.8 such a small
  box must land within a few pixels, so AP is low even when the end is found correctly (FC_B links
  are 99% correct with 7–12% head/tail AP).
- **Arrow mask IoU is modest (0.50).** Strokes are about 2 px wide at 700×700, where a one-pixel
  shift costs half the overlap. The mask is only used to separate arrows and check their ends, for
  which this is enough.
- **Fewer YOLOv7 epochs than the paper** (100 instead of 300). Results are already close; 300
  epochs and larger head/tail boxes for FA are the most promising next steps.
- **Code generation (paper §V) is not implemented.** The output is a graph (Mermaid, draw.io, SVG),
  not Python.
- **Slow on CPU.** Text recognition dominates (a 1.3 GB model run per text box).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

# YOLOv7's code is needed to load the detector weights (they are pickled YOLOv7 models)
git clone --depth 1 https://github.com/WongKinYiu/yolov7 third_party/yolov7

# datasets (COCO format, from the Arrow R-CNN / DiagramNet authors)
git clone --depth 1 https://github.com/bernhardschaefer/handwritten-diagram-datasets
```

Trained weights are not in git (about 1.6 GB). Train them with the Kaggle notebook (below) and
put the run's `hdr/` output folder at `weights/hdr/`:

```
weights/hdr/
  yolov7_runs/{fca,fcb,fa}/weights/best.pt
  deeplab/best.pt
  trocr/<best experiment>/best/
  STATE.json
```

## Usage

```bash
# one image -> graph.json, diagram.mmd, diagram.drawio, redraw.svg, report.html, overlay.png
python -m hdr.pipeline handwritten-diagram-datasets/datasets/fcb/test/writer017_fc_001.png --dataset fcb

# re-export after editing graph.json by hand (e.g. fixing a misread text)
python -m hdr.export out/pipeline/writer017_fc_001/graph.json
```

`--dataset` picks the detector: `fca` / `fcb` for flowcharts, `fa` for finite automata. Dataset
images are also scored against their ground truth.

## Training (Kaggle, 2× T4)

All GPU training runs from [notebooks/train_kaggle.ipynb](notebooks/train_kaggle.ipynb), a thin
wrapper around [hdr/tools/cloud_run.py](hdr/tools/cloud_run.py):

```bash
python -m hdr.tools.pack_code --datasets   # out/hdr_code.zip and out/hdr_datasets.zip
```

1. Upload both zips as Kaggle datasets (`hdr_code`, `hdr_datasets`) and add them as notebook inputs.
2. Settings: GPU T4 ×2, Internet on.
3. Run once with `SMOKE = True` (~10 min check), then `SMOKE = False` with
   **Save Version → Save & Run All** (~6 h).

YOLOv7 trains on GPU 0 while DeepLabv3 and then TrOCR train on GPU 1. The run stops itself
before Kaggle's 12 h limit and resumes when its previous output is attached as an input. Only
final weights, logs and metrics are kept.

## Evaluation

```bash
python -m hdr.eval.run_eval predict --datasets fca fcb fa   # runs the models once, caches outputs
python -m hdr.eval.run_eval score   --datasets fca fcb fa   # metrics -> out/eval/results.json
```

## Tests

```bash
pytest tests
```

## Repository layout

```
hdr/
  config.py              classes, thresholds, paths
  data/                  dataset loading, YOLO labels, arrow masks, text crops, synthetic text, CROHME
  train/                 DeepLabv3 and TrOCR training
  infer/models.py        YOLOv7 / DeepLabv3 / TrOCR inference wrappers
  postprocess.py         paper §III-C post-processing
  graph.py               graph building (paper §III-E)
  export.py              Mermaid, draw.io, SVG, HTML report
  pipeline.py            image -> outputs
  eval/                  test-set evaluation (paper metrics)
  tools/                 Kaggle runner, notebook builder, packaging
  assets/fonts/          handwriting fonts for synthetic text (OFL / Apache, licenses included)
notebooks/train_kaggle.ipynb
tests/
docs/PLAN.md             implementation plan and notes
```

## Differences from the paper

- **Arrow keypoints:** the dataset metadata lists arrow keypoints as `[head, tail]`, but keypoint 0 is
  the tail and keypoint 1 the head (checked on 3092/3092 FC_A arrows).
- **Arrow masks:** the offline FC_B waypoints miss arrowhead barbs, so training masks use the real ink
  near the waypoints with symbol shapes removed, instead of 6 px polylines.
- **DeepLabv3:** learning rate 1e-4 with early stopping instead of 1e-5 for 10,000 epochs.
- **YOLOv7:** trained 100 epochs (paper: 300). Head/tail boxes are squares of 3% of the image side.
- **TrOCR:** FC_A + synthetic text (CROHME optional). The loss is computed explicitly because some
  transformers versions shift VisionEncoderDecoder labels twice.
- **Post-processing additions:** head and tail are assigned jointly per arrow (using the arrow mask);
  in flowcharts, self-links are re-linked and duplicate links merged.

## Data and licenses

- Datasets: FC_A / OHFCD (CC BY-NC-SA), FC_B and FA (Bresler et al.), via
  [handwritten-diagram-datasets](https://github.com/bernhardschaefer/handwritten-diagram-datasets).
  Not included here.
- Fonts in `hdr/assets/fonts`: SIL OFL 1.1 and Apache 2.0, license files included.
- YOLOv7: https://github.com/WongKinYiu/yolov7 (cloned separately).

## Citation

```bibtex
@article{costa2026hybrid,
  title   = {Hybrid Object Detection and Semantic Segmentation for Offline Handwritten Diagram Understanding},
  author  = {Costa, Daniela de Sousa and Mello, Carlos Alexandre Barros de},
  journal = {IEEE Access},
  volume  = {14},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3721793}
}
```
