datasets="harmless-base helpful-base helpful-online helpful-rejection-sampled"
for dataset in $datasets
do
    CUDA_VISIBLE_DEVICES=0 python3 ./tools/diagnose_rlhf.py  --config ./config/hh_rlhf_$dataset\.py
    CUDA_VISIBLE_DEVICES=0 python3 ./tools/cure_hh_rlhf.py  --config_key $dataset
done

CUDA_VISIBLE_DEVICES=0 python3 ./tools/diagnose_rlhf.py  --config ./config/hh_rlhf_red-team-attempts_QA.py    
CUDA_VISIBLE_DEVICES=0 python3 ./tools/diagnose_rlhf.py  --config ./config/hh_rlhf_red-team-attempts_raw.py
CUDA_VISIBLE_DEVICES=0 python3 ./tools/cure_red_team_rlhf.py