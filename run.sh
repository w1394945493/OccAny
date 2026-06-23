conda create --prefix /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/occany python=3.12 pip openssl -y
conda create -n occany python=3.12 pip openssl -y
# pip install /c20250502/wangyushen/whl/nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_x86_64.whl
pip install /c20250502/wangyushen/whl/nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl
pip install /c20250502/wangyushen/whl/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl
pip install /c20250502/wangyushen/whl/torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl
pip install /c20250502/wangyushen/whl/torchvision-0.21.0+cu124-cp312-cp312-linux_x86_64.whl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.29.post2
pip install torch-scatter --no-cache-dir --no-build-isolation
cd third_party/croco/models/curope
python setup.py install
cd /vepfs-mlp2/c20250502/haoce/wangyushen/OccAny/third_party/Grounded-SAM-2/grounding_dino
python setup.py build_ext --inplace 

pip install roma decord transformers
pip install numpy==1.26.4 scipy==1.11.4
pip install opencv-python==4.11.0.86
pip install huggingface-hub[torch]>=0.22
pip install matplotlib tqdm einops
pip install hydra-core 
pip install iopath
pip install timm pycocotools
pip install psutil ftfy supervision 
pip install addict yapf



python inference.py \
  --batch_gen_view 2 \
  --view_batch_size 2 \
  --semantic distill@SAM2_large \
  --compute_segmentation_masks \
  --gen \
  -rot 30 \
  -vpi 2 \
  -fwd 5 \
  --seed_translation_distance 2 \
  --recon_conf_thres 2.0 \
  --gen_conf_thres 2.0 \
  --apply_majority_pooling \
  --model occany_must3r