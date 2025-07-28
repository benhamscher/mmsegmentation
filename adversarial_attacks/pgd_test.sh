epsilon=1
alpha=2.55
name='pgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/adversarial_attack_configs/segformer_mit-b1_8xb2-240k_cityscapes-1024x1024_structured.py work_dirs/segformer/trainings/segformer_mit-b1_8xb1-240k_cityscapes-1024x1024_batch_size_2/iter_240000.pth  --perform_attack --attack 'pgd' --iterations 5 --epsilon 4 --alpha 2.55 --norm 'linf' --work-dir work_dirs/segformer/evaluation/segformer_mit-b1_8xb1-240k_cityscapes-1024x1024_batch_size_2/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out work_dirs/segformer/evaluation/segformer_mit-b1_8xb1-240k_cityscapes-1024x1024_batch_size_2/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir work_dirs/segformer/evaluation/segformer_mit-b1_8xb1-240k_cityscapes-1024x1024_batch_size_2/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
