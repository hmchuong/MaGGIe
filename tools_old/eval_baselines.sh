for SUBSET in comp_easy comp_medium comp_hard real
do
    python eval_baseline.py $1 $SUBSET
done
# echo 'FTM-VM finetune'
# for SUBSET in comp_easy comp_medium comp_hard real
# do
#     python eval_baseline.py /home/chuongh/FTP-VM/output/ft_22k_rgb $SUBSET
# done

# echo 'FTM-VM public'
# for SUBSET in comp_easy comp_medium comp_hard real
# do
#     python eval_baseline.py /home/chuongh/FTP-VM/output/public $SUBSET
# done

# echo 'OTVM finetune'
# for SUBSET in comp_easy comp_medium comp_hard real
# do
#     python eval_baseline.py /home/chuongh/OTVM/output/xmem_finetuned $SUBSET
# done

# echo 'OTVM public'
# for SUBSET in comp_easy comp_medium comp_hard real
# do
#     python eval_baseline.py /home/chuongh/OTVM/output/benchmark $SUBSET
# done