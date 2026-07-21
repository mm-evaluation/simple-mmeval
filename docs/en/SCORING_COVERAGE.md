# Scoring pipeline coverage — mm-eval org audit

Survey of every dataset repo on https://huggingface.co/mm-eval: per subset,
the pre-migration composite `score_type` (audited 2026-07-15, retired) and
the atomic `score_pipeline` **now published natively in each repo's
metadata.json** (org-wide migration completed 2026-07-19; the loader's
translation boundary is deleted — see `docs/en/SCORING.md`, "Published
metadata"). Every supported pipeline below was validated against the stage
registry; REFUSED rows raise an explicit, actionable error instead of
silently mis-grading.

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

| Dataset | Subset | Retired score_type (pre-2026-07-19) | Published score_pipeline | Status | Notes |
|---|---|---|---|---|---|
| 3DSRBench | main | llm_extract | rule-match,llm-match | supported |  |
| A-OKVQA | main | rule | exact-match,rule-match | supported |  |
| AI2D | main | llm_extract | rule-match,llm-match | supported |  |
| AlgoPuzzleVQA | default | rule | exact-match,rule-match | supported |  |
| ArxivQA | main | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Art_Style | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Counting | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Forensic_Detection | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Functional_Correspondence | llm_extract | rule-match,llm-match | supported |  |
| BLINK | IQ_Test | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Jigsaw | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Multi_view_Reasoning | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Object_Localization | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Relative_Depth | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Relative_Reflectance | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Semantic_Correspondence | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Spatial_Relation | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Visual_Correspondence | llm_extract | rule-match,llm-match | supported |  |
| BLINK | Visual_Similarity | llm_extract | rule-match,llm-match | supported |  |
| BenchLMM | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 1], 'official_judge_model': 'gpt-4-0613'} |
| CMMMU | art_and_design | rule | exact-match,rule-match | supported |  |
| CMMMU | business | rule | exact-match,rule-match | supported |  |
| CMMMU | health_and_medicine | rule | exact-match,rule-match | supported |  |
| CMMMU | humanities_and_social_sciences | rule | exact-match,rule-match | supported |  |
| CMMMU | science | rule | exact-match,rule-match | supported |  |
| CMMMU | technology_and_engineering | rule | exact-match,rule-match | supported |  |
| CRPE | exist | rule | exact-match,rule-match | supported |  |
| CRPE | relation | rule | exact-match,rule-match | supported |  |
| CV-Bench | main | rule | exact-match,rule-match | supported |  |
| CVQA | main | llm_extract | rule-match,llm-match | supported |  |
| CharXiv | reasoning | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| ChartNet | main | anls | anls | supported | score_params: {'threshold': 0.0} |
| ChartQA | main | rule | exact-match,rule-match | supported | score_params: {'numeric_rel_tol': 0.05} |
| ChartQAPro | main | rule | exact-match,rule-match | supported | score_params: {'numeric_rel_tol': 0.05, 'string_match': 'anls', 'anls_threshold': 0.5} |
| DocVQA | main | anls | anls | supported | score_params: {'threshold': 0.5} |
| DynaMath | main | llm_extract | rule-match,llm-match | supported | score_params: {'numeric_abs_tol': 0.001} |
| EMMA | main | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| ENEM | main | rule | exact-match,rule-match | supported |  |
| EXAMS-V | default | rule | exact-match,rule-match | supported |  |
| EndoBench | main | llm_extract | rule-match,llm-match | supported |  |
| Ferret-Bench | default | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 10], 'official_judge_model': 'gpt-4-0314'} |
| Flickr8k | main | caption_metrics | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l']}) |
| GMAI-MMBench | default | llm_extract | rule-match,llm-match | supported |  |
| GQA | main | rule | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| GQA-Spatial | main | rule | exact-match,rule-match | supported |  |
| Geometry3K | main | none | [] (no official protocol) | supported | default pipeline, stamped non-official |
| HRBench4K | main | llm_extract | rule-match,llm-match | supported |  |
| HRBench8K | main | llm_extract | rule-match,llm-match | supported |  |
| HallusionBench | main | rule_llm_judge | exact-match,rule-match,llm-judge | supported | score_params: {'judge_inputs': ['gt_answer_details']} |
| HallusionBench | non_image | rule_llm_judge | exact-match,rule-match,llm-judge | supported | score_params: {'judge_inputs': ['gt_answer_details']} |
| HumanEval-V | main | execution | unsupported: code-execution | REFUSED | requires official test-case execution (pass@1); use the benchmark's own evaluator (score_params: {'metric': 'pass@1'}) |
| IconQA | choose_img | rule | exact-match,rule-match | supported |  |
| IconQA | choose_txt | rule | exact-match,rule-match | supported |  |
| IconQA | fill_in_blank | rule | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| InfographicVQA | main | anls | anls | supported | score_params: {'threshold': 0.5} |
| LEGO-Puzzles | default | llm_extract | rule-match,llm-match | supported |  |
| LLaVA-Bench-COCO | default | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 10], 'official_judge_model': 'gpt-4-0314'} |
| LLaVA-Bench-Wilder | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 10], 'official_judge_model': 'gpt-4-vision-preview'} |
| LLaVA-Bench-in-the-Wild | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 10], 'official_judge_model': 'gpt-4-0314', 'judge_inputs': ['caption']} |
| LiveBench | 2024-05 | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o'} |
| LiveBench | 2024-06 | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o', 'judge_inputs': ['criteria']} |
| LiveBench | 2024-07 | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o', 'judge_inputs': ['criteria']} |
| LiveBench | 2024-08 | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o', 'judge_inputs': ['criteria']} |
| LiveBench | 2024-09 | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 10], 'official_judge_model': 'gpt-4o', 'judge_inputs': ['criteria']} |
| LogicVista | main | llm_extract | rule-match,llm-match | supported |  |
| M3CoT | main | rule | exact-match,rule-match | supported |  |
| MEGA-Bench | main | per_task_metric | unsupported: per-task-metric | REFUSED | requires the benchmark's own evaluator (per-row metric dispatch) (score_params: {'metric_field': 'task_name'}) |
| MM-IFEval | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'partial_credit', 'judge_inputs': ['constraints']} |
| MM-Vet-v2 | default | llm_judge | llm-judge | supported | score_params: {'rubric': 'partial_credit', 'scale': [0.0, 1.0], 'official_judge_model': 'gpt-4-0613', 'reference_syntax': 'and_or_composition'} |
| MMBench | cc | llm_extract | rule-match,llm-match | supported |  |
| MMBench | cn | llm_extract | rule-match,llm-match | supported |  |
| MMBench | en | llm_extract | rule-match,llm-match | supported |  |
| MMBench-V11 | cn | llm_extract | rule-match,llm-match | supported |  |
| MMBench-V11 | en | llm_extract | rule-match,llm-match | supported |  |
| MME | main | rule | exact-match,rule-match | supported |  |
| MME-RealWorld-Lite | main | rule | exact-match,rule-match | supported |  |
| MMEvalPro | main | rule | exact-match,rule-match | supported |  |
| MMHal-Bench | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [0, 6], 'official_judge_model': 'gpt-4', 'judge_inputs': ['image_content']} |
| MMIU | default | llm_extract | rule-match,llm-match | supported |  |
| MMMU | main | rule | exact-match,rule-match | supported |  |
| MMMU-Pro | standard_10_options | rule | exact-match,rule-match | supported |  |
| MMMU-Pro | standard_4_options | rule | exact-match,rule-match | supported |  |
| MMMU-Pro | vision | rule | exact-match,rule-match | supported |  |
| MMStar | main | llm_extract | rule-match,llm-match | supported |  |
| MMT-Bench | main | llm_extract | rule-match,llm-match | supported |  |
| MMVP | main | llm_extract | rule-match,llm-match | supported |  |
| MMVU | main | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| MMVet | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'partial_credit', 'scale': [0.0, 1.0], 'official_judge_model': 'gpt-4-0613', 'reference_syntax': 'and_or_composition'} |
| MMVet-v2 | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'partial_credit', 'scale': [0.0, 1.0], 'official_judge_model': 'gpt-4-0613', 'reference_syntax': 'and_or_composition'} |
| MS-COCO-Captions | main | caption_metrics | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l']}) |
| MTVQA | main | rule | exact-match,rule-match | supported | score_params: {'string_match': 'contains'} |
| MaRVL | main | rule | exact-match,rule-match | supported |  |
| Mantis-Eval | main | rule | exact-match,rule-match | supported |  |
| MathVerse | testmini | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| MathVerse | testmini_text_only | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| MathVision | main | rule | exact-match,rule-match | supported |  |
| MathVista | main | llm_extract | rule-match,llm-match | supported |  |
| MedXpertQA | MM | rule | exact-match,rule-match | supported |  |
| Mementos | default | llm_extract | unsupported: keyword-set-f1 | REFUSED | requires keyword extraction + set-F1 grading (Mementos); use the benchmark's own evaluator (score_params: {'compare': 'keyword_set_f1'}) |
| MuirBench | main | rule | exact-match,rule-match | supported |  |
| NExT-QA | mc | rule | exact-match,rule-match | supported |  |
| NLVR2 | main | rule | exact-match,rule-match | supported |  |
| NaturalBench | default | rule | exact-match,rule-match | supported |  |
| NoCaps | main | caption_metrics | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l']}) |
| OCR-VQA | main | rule | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| OCRBench | main | rule | exact-match,rule-match | supported | score_params: {'string_match': 'contains'} |
| OCRBench-v2 | default | per_task_metric | unsupported: per-task-metric | REFUSED | requires the benchmark's own evaluator (per-row metric dispatch) (score_params: {'metric_field': 'type'}) |
| OK-VQA | main | vqa_accuracy | vqa-accuracy | supported |  |
| OlympiadBench | main | rule | exact-match,rule-match | supported |  |
| PMC-VQA | main | rule | exact-match,rule-match | supported |  |
| POPE | main | rule | exact-match,rule-match | supported |  |
| PathVQA | main | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| PhyX | mc | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| PuzzleVQA | default | rule | exact-match,rule-match | supported |  |
| Q-Bench-Plus | default | llm_extract | rule-match,llm-match | supported |  |
| R-Bench-V | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'binary', 'official_judge_model': 'gpt-4o'} |
| RealWorldQA | main | llm_extract | rule-match,llm-match | supported |  |
| SAT | real | rule | exact-match,rule-match | supported |  |
| SAT | synthetic | rule | exact-match,rule-match | supported |  |
| SEED-Bench-2-Plus | main | llm_extract | rule-match,llm-match | supported |  |
| SEED-Bench-H | main | llm_extract | rule-match,llm-match | supported |  |
| SEEDBench | main | llm_extract | rule-match,llm-match | supported |  |
| SLAKE | main | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| ST-VQA | main | anls | anls | supported | score_params: {'threshold': 0.5} |
| SciFIBench | caption2figure | llm_extract | rule-match,llm-match | supported |  |
| SciFIBench | figure2caption | llm_extract | rule-match,llm-match | supported |  |
| SciVQA | default | caption_metrics | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['rouge1', 'rouge_l', 'bertscore']}) |
| ScienceQA | main | rule | exact-match,rule-match | supported |  |
| ScienceQA-IMG | default | rule | exact-match,rule-match | supported |  |
| ScreenQA | main | rule | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| SeePhys | default | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| SpatialEval | vqa | rule | exact-match,rule-match | supported |  |
| SpatialEval | vtqa | rule | exact-match,rule-match | supported |  |
| TableVQA-Bench | fintabnetqa | rule | exact-match,rule-match | supported |  |
| TableVQA-Bench | vtabfact | rule | exact-match,rule-match | supported |  |
| TableVQA-Bench | vwtq | rule | exact-match,rule-match | supported |  |
| TableVQA-Bench | vwtq_syn | rule | exact-match,rule-match | supported |  |
| TallyQA | main | rule | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| TempCompass | caption_matching | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| TempCompass | captioning | llm_judge | llm-judge | supported | score_params: {'rubric': 'binary', 'official_judge_model': 'gpt-3.5-turbo-1106'} |
| TempCompass | multi_choice | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| TempCompass | yes_no | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| TextCaps | main | caption_metrics | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l']}) |
| TextVQA | main | vqa_accuracy | vqa-accuracy | supported |  |
| TheoremQA | image | rule | exact-match,rule-match | supported | score_params: {'numeric_rel_tol': 0.04} |
| VLMsAreBlind | default | rule | exact-match,rule-match | supported |  |
| VQA-RAD | main | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| VQAv2 | main | vqa_accuracy | vqa-accuracy | supported |  |
| VSI-Bench | mc | rule | exact-match,rule-match | supported |  |
| VSR | zeroshot | llm_extract | rule-match,llm-match | supported |  |
| VStarBench | main | llm_extract | rule-match,llm-match | supported |  |
| VibeEval | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [1, 5], 'official_judge_model': 'reka-core-20240501'} |
| VisIT-Bench | default | llm_judge | llm-judge | supported | score_params: {'rubric': 'binary', 'official_judge_model': 'gpt-4', 'judge_inputs': ['instruction_conditioned_caption']} |
| VisOnlyQA | main | llm_extract | rule-match,llm-match | supported |  |
| Visual7W | main | llm_extract | rule-match,llm-match | supported |  |
| VisualMRC | main | caption_metrics | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['rouge_l', 'bleu4', 'meteor', 'cider']}) |
| VisualSimpleQA | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'three_way', 'official_judge_model': 'SimpleQA-style ChatGPT grader (GPT-4o class)'} |
| VizWiz-Captions | main | caption_metrics | unsupported: caption-metrics | REFUSED | requires corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator (score_params: {'metrics': ['cider', 'bleu4', 'meteor', 'rouge_l', 'spice']}) |
| VizWiz-VQA | main | vqa_accuracy | vqa-accuracy | supported |  |
| WHOOPS | main | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| We-Math | main | rule | exact-match,rule-match | supported |  |
| WebSRC | main | rule | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |
| WildVision-Bench | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'scale', 'scale': [-2, 2], 'official_judge_model': 'gpt-4o'} |
| Winoground | main | rule | exact-match,rule-match | supported |  |
| WorldMedQA-V | default | llm_extract | rule-match,llm-match | supported |  |
| WorldVQA | main | llm_judge | llm-judge | supported | score_params: {'rubric': 'three_way', 'official_judge_model': 'gpt-4o-1120'} |
| ZeroBench | default | rule | exact-match,rule-match | supported |  |
| ZoomBench | main | rule_llm_judge | exact-match,rule-match,llm-judge | supported |  |
| xGQA | main | rule | exact-match,rule-match | supported | score_params: {'string_match': 'exact'} |

## Retired score_type distribution (pre-migration, historical)

- `rule`: 63
- `llm_extract`: 48
- `llm_judge`: 22
- `rule_llm_judge`: 17
- `caption_metrics`: 7
- `anls`: 4
- `vqa_accuracy`: 4
- `per_task_metric`: 2
- `none`: 1
- `execution`: 1

## Refused protocols

- `caption_metrics` (7 subsets) → corpus-level caption metrics (CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator.
- `per_task_metric` (2) → per-row metric dispatch; use the benchmark's own evaluator.
- `execution` (1) → official test-case execution (pass@1).
- `llm_extract` + `compare=keyword_set_f1` (1, Mementos) → keyword extraction + set-F1 grading.

An explicit `--score_pipeline` can force approximate scoring for these; the
output is then stamped `official_protocol: false` with a protocol note.
