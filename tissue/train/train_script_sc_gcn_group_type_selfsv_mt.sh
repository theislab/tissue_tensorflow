#!/bin/bash

#CODE_PATH="/home/iterm/mayar.ali/phd/projects/"
#OUT_PATH_BASE="/home/iterm/mayar.ali/phd/projects/tissue"
#DATA_PATH_BASE="/home/iterm/mayar.ali/phd/projects/tissue/data"
CODE_PATH="/home/icb/sabrina.richter/git"
OUT_PATH_BASE="/home/icb/sabrina.richter/git/testruns"
DATA_PATH_BASE="/storage/groups/ml01/workspace/david.fischer/tissue/data"

PARTITION="gpu_p"

MODEL_CLASS="GCN"
DATA_SET=("schuerch")
OPTIMIZER=("ADAM")
LEARNING_RATE_KEYS=("2+3")
DROPOUT_RATE_KEYS=("1")
L2_KEYS=("1")
DEPTH_FEATURE_EMBEDDING_KEYS=("1" "2")
DEPTH_KEYS=("2" "3")
WIDTH_KEYS=("2")
LOSS_WEIGHT=("1")
LOSS_WEIGHT_TYPE=("1")
MMD_KEYS=("1")
BATCH_SIZE=("2")
NODE_FRACTION=("4")
MAX_DIST=("1")
RADIUS_STEP=("3")
TRANSFORM_KEY=("1")
COVAR_KEY=("1")
FEATURE_SPACE=("TYPE")
TARGET_LABELS=("GROUP")
AGGREGATION="SPECTRAL"
FINAL_POOLING=("MEAN")
MULTITASK=("TARGET")
N_CLUSTER=("1") # 5 and 10 clusters
ENTROPY_WEIGHT=("1")
NUMBER_HEADS=("1")
ADJ_TYPE=("SPECTRAL")
ADJ_BASE="RADIUS"
K_NEIGHBORS=("1")
SELF_SUPERVISION_MODE=("MULTITASK")

GS_KEY="210527_${MODEL_CLASS}_${AGGREGATION}_${TARGET_LABELS}_${FEATURE_SPACE}_${DATA_SET}_${ADJ_BASE}_${SELF_SUPERVISION_MODE}"

OUT_PATH=${OUT_PATH_BASE}/grid_searches/${GS_KEY}

rm -rf ${OUT_PATH}/jobs
rm -rf ${OUT_PATH}/logs
rm -rf ${OUT_PATH}/results
mkdir -p ${OUT_PATH}/jobs
mkdir -p ${OUT_PATH}/logs
mkdir -p ${OUT_PATH}/results

for ds in ${DATA_SET[@]}; do
    for o in ${OPTIMIZER[@]}; do
        for lr in ${LEARNING_RATE_KEYS[@]}; do
            for dr in ${DROPOUT_RATE_KEYS[@]}; do
                for l2 in ${L2_KEYS[@]}; do
                    for d in ${DEPTH_KEYS[@]}; do
                        for w in ${WIDTH_KEYS[@]}; do
                            for lw in ${LOSS_WEIGHT[@]}; do
                                for lt in ${LOSS_WEIGHT_TYPE[@]}; do
                                    for bs in ${BATCH_SIZE[@]}; do
                                        for nf in ${NODE_FRACTION[@]}; do
                                            for md in ${MAX_DIST[@]}; do
                                                for rs in ${RADIUS_STEP[@]}; do
                                                    for tk in ${TRANSFORM_KEY[@]}; do
                                                        for ck in ${COVAR_KEY[@]}; do
                                                            for fp in ${FINAL_POOLING[@]}; do
                                                                for tl in ${TARGET_LABELS[@]}; do
                                                                    for fs in ${FEATURE_SPACE[@]}; do
                                                                        for fe in ${DEPTH_FEATURE_EMBEDDING_KEYS[@]}; do
                                                                            for mk in ${MMD_KEYS[@]}; do
                                                                                for mt in ${MULTITASK[@]}; do
                                                                                    for nc in ${N_CLUSTER[@]}; do
                                                                                        for ew in ${ENTROPY_WEIGHT[@]}; do
                                                                                            for nh in ${NUMBER_HEADS[@]}; do
                                                                                                for ad in ${ADJ_TYPE[@]}; do
                                                                                                    for kn in ${K_NEIGHBORS[@]}; do
                                                                                                        sleep 0.1
                                                                                                        job_file="${OUT_PATH}/jobs/run_${MODEL_CLASS}_${ds}_${o}_${ADJ_BASE}_${SELF_SUPERVISION_MODE}_${lr}_${dr}_${l2}_${d}_${w}_${lw}_${lt}_${mk}_${bs}_${nf}_${md}_${rs}_${tk}_${ck}_${tl}_${fs}_${fe}_${fp}_${mt}_${nc}_${ew}_${nh}_${ad}_${kn}.cmd"
                                                                                                        echo "#!/bin/bash
#SBATCH -J ${MODEL_CLASS}_${ds}_${o}_${ADJ_BASE}_${SELF_SUPERVISION_MODE}_${lr}_${dr}_${l2}_${d}_${w}_${lw}_${lt}_${mk}_${bs}_${nf}_${md}_${rs}_${tk}_${ck}_${tl}_${fs}_${fe}_${fp}_${mt}_${nc}_${ew}_${nh}_${ad}_${kn}_${GS_KEY}
#SBATCH -o ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${ds}_${o}_${ADJ_BASE}_${SELF_SUPERVISION_MODE}_${lr}_${dr}_${l2}_${d}_${w}_${lw}_${lt}_${mk}_${bs}_${nf}_${md}_${rs}_${tk}_${ck}_${tl}_${fs}_${fe}_${fp}_${mt}_${nc}_${ew}_${nh}_${ad}_${kn}.out
#SBATCH -e ${OUT_PATH}/jobs/run_${MODEL_CLASS}_${ds}_${o}_${ADJ_BASE}_${SELF_SUPERVISION_MODE}_${lr}_${dr}_${l2}_${d}_${w}_${lw}_${lt}_${mk}_${bs}_${nf}_${md}_${rs}_${tk}_${ck}_${tl}_${fs}_${fe}_${fp}_${mt}_${nc}_${ew}_${nh}_${ad}_${kn}.err
#SBATCH -p ${PARTITION}
#SBATCH -t 2-00:00:00
#SBATCH -c 6
#SBATCH --qos=gpu
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#SBATCH --nodelist=supergpu03pxe
source ~/.bash_profile
conda activate tissue
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/
python ${CODE_PATH}/tissue/tissue/train/train_script_gnn.py ${ds} ${o} ${lr} ${dr} ${l2} ${d} ${w} ${lw} ${lt} ${mk} ${bs} ${nf} ${md} ${rs} ${kn} ${tk} ${ck} ${tl} ${fs} ${fe} ${mt} ${nc} ${ew} ${nh} ${SELF_SUPERVISION_MODE} ${MODEL_CLASS} ${AGGREGATION} ${fp} ${ad} ${ADJ_BASE} ${GS_KEY} ${DATA_PATH_BASE} ${OUT_PATH}
" > ${job_file}
                                                                                                       sbatch $job_file
                                                                                                    done
                                                                                                done
                                                                                            done
                                                                                        done
                                                                                    done
                                                                                done
                                                                            done
                                                                        done
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
