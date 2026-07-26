# Scoring pipeline coverage — mm-eval org audit

Survey of every dataset repo on https://huggingface.co/mm-eval: the atomic
`score_pipeline` each repo declares per subset in its `metadata.json` (see
`docs/en/SCORING.md`, "Metadata schema"). Every supported pipeline below was
validated against the stage registry; REFUSED rows raise an explicit,
actionable error instead of silently mis-grading.

**169 subsets across 128 dataset repos: 158 supported, 11 refused.** (`mm-eval/VLMEvalKit` carries no metadata.json — it is a raw TSV mirror, not an mm-eval-format dataset — and is excluded.)

## Stage usage census

Official-protocol demand per atomic stage (datasets / subsets):

- `exact-match`: 61 datasets / 80 subsets
- `rule-match`: 91 datasets / 127 subsets
- `llm-match`: 30 datasets / 47 subsets
- `llm-judge`: 30 datasets / 39 subsets
- `vqa-accuracy`: 4 datasets / 4 subsets
- `anls`: 4 datasets / 4 subsets

Narrowest stages: `vqa-accuracy` (VQAv2, OK-VQA, TextVQA, VizWiz-VQA) and
`anls` (DocVQA, InfographicVQA, ST-VQA, ChartNet; the ANLS comparator also
backs `string_match=anls`, e.g. ChartQAPro) — 4 datasets each, above the
1–2-dataset prune threshold, and both are the ecosystem's canonical
fractional metrics.

| Dataset | Subset | score_pipeline | Status | Notes |
|---|---|---|---|---|
| 3DSRBench | main | rule-match,llm-match | supported |  |
| A-OKVQA | main | exact-match,rule-match | supported |  |
| AI2D | main | rule-match,llm-match | supported |  |
| AlgoPuzzleVQA | default | exact-match,rule-match | supported |  |
| ArxivQA | main | rule-match,llm-match | supported |  |
| BLINK | Art_Style | rule-match,llm-match | supported |  |
| BLINK | Counting | rule-match,llm-match | supported |  |
| BLINK | Forensic_Detection | rule-match,llm-match | supported |  |
| BLINK | Functional_Correspondence | rule-match,llm-match | supported |  |
| BLINK | IQ_Test | rule-match,llm-match | supported |  |
| BLINK | Jigsaw | rule-match,llm-match | supported |  |
| BLINK | Multi_view_Reasoning | rule-match,llm-match | supported |  |
| BLINK | Object_Localization | rule-match,llm-match | supported |  |
| BLINK | Relative_Depth | rule-match,llm-match | supported |  |
| BLINK | Relative_Reflectance | rule-match,llm-match | supported |  |
| BLINK | Semantic_Correspondence | rule-match,llm-match | supported |  |
| BLINK | Spatial_Relation | rule-match,llm-match | supported |  |
| BLINK | Visual_Correspondence | rule-match,llm-match | supported |  |
| BLINK | Visual_Similarity | rule-match,llm-match | supported |  |
| BenchLMM | main | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 1], 'official_judge_model': 'gpt-4-0613'} |
| CMMMU | art_and_design | exact-match,rule-match | supported |  |
| CMMMU | business | exact-match,rule-match | supported |  |
| CMMMU | health_and_medicine | exact-match,rule-match | supported |  |
| CMMMU | humanities_and_social_sciences | exact-match,rule-match | supported |  |
| CMMMU | science | exact-match,rule-match | supported |  |
| CMMMU | technology_and_engineering | exact-match,rule-match | supported |  |
| CRPE | exist | exact-match,rule-match | supported |  |
| CRPE | relation | exact-match,rule-match | supported |  |
| CV-Bench | main | exact-match,rule-match | supported |  |
| CVQA | main | rule-match,llm-match | supported |  |
| CharXiv | reasoning | exact-match,rule-match,llm-judge | supported |  |
| ChartNet | main | anls | supported | score_params: {'threshold': 0.0} |
| ChartQA | main | exact-match,rule-match | supported | score_params: {'numeric_rel_tol': 0.05} |
| ChartQAPro | main | exact-match,rule-match | supported | score_params: {'numeric_rel_tol': 0.05, 'string_match': 'anls', 'anls_threshold': 0.5} |
| DocVQA | main | anls | supported | score_params: {'threshold': 0.5} |
| DynaMath | main | rule-match,llm-match | supported | score_params: {'numeric_abs_tol': 0.001} |
| EMMA | main | exact-match,rule-match,llm-judge | supported |  |
| ENEM | main | exact-match,rule-match | supported |  |
| EXAMS-V | default | exact-match,rule-match | supported |  |
| EndoBench | main | rule-match,llm-match | supported |  |
| Ferret-Bench | default | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 10], 'official_judge_model': 'gpt-4-0314'} |
| Flickr8k | main | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l']}) |
| GMAI-MMBench | default | rule-match,llm-match | supported |  |
| GQA | main | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| GQA-Spatial | main | exact-match,rule-match | supported |  |
| Geometry3K | main | [] (no official protocol) | supported | default pipeline, stamped non-official |
| HRBench4K | main | rule-match,llm-match | supported |  |
| HRBench8K | main | rule-match,llm-match | supported |  |
| HallusionBench | main | exact-match,rule-match,llm-judge | supported | score_params: {'judge_inputs': ['gt_answer_details']} |
| HallusionBench | non_image | exact-match,rule-match,llm-judge | supported | score_params: {'judge_inputs': ['gt_answer_details']} |
| HumanEval-V | main | unsupported: code-execution | REFUSED | requires official test-case execution (pass@1); use the benchmark's own evaluator (score_params: {'metric': 'pass@1'}) |
| IconQA | choose_img | exact-match,rule-match | supported |  |
| IconQA | choose_txt | exact-match,rule-match | supported |  |
| IconQA | fill_in_blank | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| InfographicVQA | main | anls | supported | score_params: {'threshold': 0.5} |
| LEGO-Puzzles | default | rule-match,llm-match | supported |  |
| LLaVA-Bench-COCO | default | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 10], 'official_judge_model': 'gpt-4-0314'} |
| LLaVA-Bench-Wilder | main | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 10], 'official_judge_model': 'gpt-4-vision-preview'} |
| LLaVA-Bench-in-the-Wild | main | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 10], 'official_judge_model': 'gpt-4-0314', 'judge_inputs': ['caption']} |
| LiveBench | 2024-05 | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o'} |
| LiveBench | 2024-06 | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o', 'judge_inputs': ['criteria']} |
| LiveBench | 2024-07 | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o', 'judge_inputs': ['criteria']} |
| LiveBench | 2024-08 | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o', 'judge_inputs': ['criteria']} |
| LiveBench | 2024-09 | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o', 'judge_inputs': ['criteria']} |
| LogicVista | main | rule-match,llm-match | supported |  |
| M3CoT | main | exact-match,rule-match | supported |  |
| MEGA-Bench | main | unsupported: per-task-metric | REFUSED | requires the benchmark's own evaluator (per-row metric dispatch) (score_params: {'metric_field': 'task_name'}) |
| MM-IFEval | main | llm-judge | supported | score_params: {'rubric': 'partial_credit', 'judge_inputs': ['constraints']} |
| MM-Vet-v2 | default | llm-judge | supported | score_params: {'rubric': 'partial_credit', 'scale': [0.0, 1.0], 'official_judge_model': 'gpt-4-0613', 'reference_syntax': 'and_or_composition'} |
| MMBench | cc | rule-match,llm-match | supported |  |
| MMBench | cn | rule-match,llm-match | supported |  |
| MMBench | en | rule-match,llm-match | supported |  |
| MMBench-V11 | cn | rule-match,llm-match | supported |  |
| MMBench-V11 | en | rule-match,llm-match | supported |  |
| MME | main | exact-match,rule-match | supported |  |
| MME-RealWorld-Lite | main | exact-match,rule-match | supported |  |
| MMEvalPro | main | exact-match,rule-match | supported |  |
| MMHal-Bench | main | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 6], 'official_judge_model': 'gpt-4', 'judge_inputs': ['image_content']} |
| MMIU | default | rule-match,llm-match | supported |  |
| MMMU | main | exact-match,rule-match | supported |  |
| MMMU-Pro | standard_10_options | exact-match,rule-match | supported |  |
| MMMU-Pro | standard_4_options | exact-match,rule-match | supported |  |
| MMMU-Pro | vision | exact-match,rule-match | supported |  |
| MMStar | main | rule-match,llm-match | supported |  |
| MMT-Bench | main | rule-match,llm-match | supported |  |
| MMVP | main | rule-match,llm-match | supported |  |
| MMVU | main | exact-match,rule-match,llm-judge | supported |  |
| MMVet | main | llm-judge | supported | score_params: {'rubric': 'partial_credit', 'scale': [0.0, 1.0], 'official_judge_model': 'gpt-4-0613', 'reference_syntax': 'and_or_composition'} |
| MMVet-v2 | main | llm-judge | supported | score_params: {'rubric': 'partial_credit', 'scale': [0.0, 1.0], 'official_judge_model': 'gpt-4-0613', 'reference_syntax': 'and_or_composition'} |
| MS-COCO-Captions | main | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l']}) |
| MTVQA | main | exact-match,rule-match | supported | score_params: {'string_match': 'contains'} |
| MaRVL | main | exact-match,rule-match | supported |  |
| Mantis-Eval | main | exact-match,rule-match | supported |  |
| MathVerse | testmini | exact-match,rule-match,llm-judge | supported |  |
| MathVerse | testmini_text_only | exact-match,rule-match,llm-judge | supported |  |
| MathVision | main | exact-match,rule-match | supported |  |
| MathVista | main | rule-match,llm-match | supported |  |
| MedXpertQA | MM | exact-match,rule-match | supported |  |
| Mementos | default | unsupported: keyword-set-f1 | REFUSED | requires keyword extraction + set-F1 grading (Mementos); use the benchmark's own evaluator (score_params: {'compare': 'keyword_set_f1'}) |
| MuirBench | main | exact-match,rule-match | supported |  |
| NExT-QA | mc | exact-match,rule-match | supported |  |
| NLVR2 | main | exact-match,rule-match | supported |  |
| NaturalBench | default | exact-match,rule-match | supported |  |
| NoCaps | main | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l']}) |
| OCR-VQA | main | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| OCRBench | main | exact-match,rule-match | supported | score_params: {'string_match': 'contains'} |
| OCRBench-v2 | default | unsupported: per-task-metric | REFUSED | requires the benchmark's own evaluator (per-row metric dispatch) (score_params: {'metric_field': 'type'}) |
| OK-VQA | main | vqa-accuracy | supported |  |
| OlympiadBench | main | exact-match,rule-match | supported |  |
| PMC-VQA | main | exact-match,rule-match | supported |  |
| POPE | main | exact-match,rule-match | supported |  |
| PathVQA | main | exact-match,rule-match,llm-judge | supported |  |
| PhyX | mc | exact-match,rule-match,llm-judge | supported |  |
| PuzzleVQA | default | exact-match,rule-match | supported |  |
| Q-Bench-Plus | default | rule-match,llm-match | supported |  |
| R-Bench-V | main | llm-judge | supported | score_params: {'rubric': 'binary', 'official_judge_model': 'gpt-4o'} |
| RealWorldQA | main | rule-match,llm-match | supported |  |
| SAT | real | exact-match,rule-match | supported |  |
| SAT | synthetic | exact-match,rule-match | supported |  |
| SEED-Bench-2-Plus | main | rule-match,llm-match | supported |  |
| SEED-Bench-H | main | rule-match,llm-match | supported |  |
| SEEDBench | main | rule-match,llm-match | supported |  |
| SLAKE | main | exact-match,rule-match,llm-judge | supported |  |
| ST-VQA | main | anls | supported | score_params: {'threshold': 0.5} |
| SciFIBench | caption2figure | rule-match,llm-match | supported |  |
| SciFIBench | figure2caption | rule-match,llm-match | supported |  |
| SciVQA | default | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['rouge1', 'rouge_l', 'bertscore']}) |
| ScienceQA | main | exact-match,rule-match | supported |  |
| ScienceQA-IMG | default | exact-match,rule-match | supported |  |
| ScreenQA | main | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| SeePhys | default | exact-match,rule-match,llm-judge | supported |  |
| SpatialEval | vqa | exact-match,rule-match | supported |  |
| SpatialEval | vtqa | exact-match,rule-match | supported |  |
| TableVQA-Bench | fintabnetqa | exact-match,rule-match | supported |  |
| TableVQA-Bench | vtabfact | exact-match,rule-match | supported |  |
| TableVQA-Bench | vwtq | exact-match,rule-match | supported |  |
| TableVQA-Bench | vwtq_syn | exact-match,rule-match | supported |  |
| TallyQA | main | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| TempCompass | caption_matching | exact-match,rule-match,llm-judge | supported |  |
| TempCompass | captioning | llm-judge | supported | score_params: {'rubric': 'binary', 'official_judge_model': 'gpt-3.5-turbo-1106'} |
| TempCompass | multi_choice | exact-match,rule-match,llm-judge | supported |  |
| TempCompass | yes_no | exact-match,rule-match,llm-judge | supported |  |
| TextCaps | main | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l']}) |
| TextVQA | main | vqa-accuracy | supported |  |
| TheoremQA | image | exact-match,rule-match | supported | score_params: {'numeric_rel_tol': 0.04} |
| VLMsAreBlind | default | exact-match,rule-match | supported |  |
| VQA-RAD | main | exact-match,rule-match,llm-judge | supported |  |
| VQAv2 | main | vqa-accuracy | supported |  |
| VSI-Bench | mc | exact-match,rule-match | supported |  |
| VSR | zeroshot | rule-match,llm-match | supported |  |
| VStarBench | main | rule-match,llm-match | supported |  |
| VibeEval | main | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 5], 'official_judge_model': 'reka-core-20240501'} |
| VisIT-Bench | default | llm-judge | supported | score_params: {'rubric': 'binary', 'official_judge_model': 'gpt-4', 'judge_inputs': ['instruction_conditioned_caption']} |
| VisOnlyQA | main | rule-match,llm-match | supported |  |
| Visual7W | main | rule-match,llm-match | supported |  |
| VisualMRC | main | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['rouge_l', 'bleu4', 'meteor', 'cider']}) |
| VisualSimpleQA | main | llm-judge | supported | score_params: {'rubric': 'three_way', 'official_judge_model': 'SimpleQA-style ChatGPT grader (GPT-4o class)'} |
| VizWiz-Captions | main | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l', 'spice']}) |
| VizWiz-VQA | main | vqa-accuracy | supported |  |
| WHOOPS | main | exact-match,rule-match,llm-judge | supported |  |
| We-Math | main | exact-match,rule-match | supported |  |
| WebSRC | main | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| WildVision-Bench | main | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [-2, 2], 'official_judge_model': 'gpt-4o'} |
| Winoground | main | exact-match,rule-match | supported |  |
| WorldMedQA-V | default | rule-match,llm-match | supported |  |
| WorldVQA | main | llm-judge | supported | score_params: {'rubric': 'three_way', 'official_judge_model': 'gpt-4o-1120'} |
| ZeroBench | default | exact-match,rule-match | supported |  |
| ZoomBench | main | exact-match,rule-match,llm-judge | supported |  |
| xGQA | main | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |

## Refused protocols

- `caption_metrics` (7 subsets) → corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator.
- `per_task_metric` (2) → per-row metric dispatch; use the benchmark's own evaluator.
- `execution` (1) → official test-case execution (pass@1).
- `llm_extract` + `compare=keyword_set_f1` (1, Mementos) → keyword extraction + set-F1 grading.

An explicit `--score_pipeline` can force approximate scoring for these; the
output is then stamped `official_protocol: false` with a protocol note.
