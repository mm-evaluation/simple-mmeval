export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-2B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-2B-Thinking-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-2B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-2B-Instruct-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct-FP8 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-2B-Thinking-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-2B-Thinking-FP8 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-4B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-4B-Thinking-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-4B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-4B-Instruct-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct-FP8 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-4B-Thinking-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-4B-Thinking-FP8 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-8B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-8B-Thinking-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-8B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-8B-Instruct-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct-FP8 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-8B-Thinking-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-8B-Thinking-FP8 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-30B-A3B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-30B-A3B-Thinking-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Thinking \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-30B-A3B-Instruct-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-30B-A3B-Thinking-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-32B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-32B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-32B-Thinking-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-32B-Thinking \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-32B-Instruct-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-32B-Thinking-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-32B-Thinking-FP8 \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-235B-A22B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-235B-A22B-Instruct \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-235B-A22B-Thinking-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-235B-A22B-Thinking \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-235B-A22B-Instruct-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 128

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-235B-A22B-Thinking-FP8-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 128
