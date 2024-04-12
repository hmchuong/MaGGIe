SUBSET=$2
CUDA_VISIBLE_DEVICES=$1 python infer_vim.py --config config/InstMatt-stage2.toml --checkpoint /sensei-fs/users/chuongh/InstMatt/checkpoints/InstMatt-ft-vim-1029-stage2/best_model.pth \
                 --image-pattern '/home/chuongh/vm2m/data/syn/benchmark/'$SUBSET'/fgr/*/*.jpg' --mask-dir /home/chuongh/vm2m/data/syn/benchmark/$SUBSET/mask --output results/vim_ft_1029/$SUBSET