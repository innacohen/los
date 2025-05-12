#!/bin/bash
#SBATCH -J los
#SBATCH -p day
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH --mail-type=ALL


module purge
module load miniconda
conda activate aitc
papermill los_features_all_with_quintiles_07-25-2024.ipynb los_features_all_with_quintiles_07-25-2024.ipynb
papermill los_features_all_without_quintiles_07-25-2024.ipynb los_features_all_without_quintiles_07-25-2024.ipynb
papermill los_features_discharged_home_with_quintiles_07-25-2024.ipynb los_features_discharged_home_with_quintiles_07-25-2024.ipynb
papermill los_features_discharged_home_wo_quintiles_07-25-2024.ipynb los_features_discharged_home_wo_quintiles_07-25-2024.ipynb
papermill los_features_hf_with_quintiles_07-25-2024.ipynb los_features_hf_with_quintiles_07-25-2024.ipynb
papermill los_features_hf_wo_quintiles_07-25-2024.ipynb los_features_hf_wo_quintiles_07-25-2024.ipynb
