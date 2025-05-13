

# LIBRARIES ---------------------------------------------------------------


library(tidyverse)
library(readxl)
library(sjlabelled)
library(naniar)
library(dplyr)
library(stringr)


# FUNCTIONS ---------------------------------------------------------------




get_codebook <- function(df) {
  # Extract labels
  cb <- enframe(get_label(df)) %>%
    rename(variable = name, description = value)
  
  # Summary of missing data
  miss_df <- df %>%
    miss_var_summary() %>%
    mutate(pct_miss = as.numeric(pct_miss))  # Convert pillar_num to numeric
  
  # Extract column classes
  column_classes <- sapply(df, function(x) class(x)[1])  # Ensure single class per column
  
  class_df <- data.frame(
    variable = names(df),
    r_class = column_classes,
    stringsAsFactors = FALSE
  )
  
  # Extract unique values and counts
  unique_values_df <- df %>%
    summarise(across(everything(), ~list(unique(.)), .names = "unique_{.col}")) %>%
    pivot_longer(cols = everything(), names_to = "variable", values_to = "unique_values") %>%
    mutate(variable = gsub("^unique_", "", variable))  # Clean variable names
  
  # Get the number of unique values
  num_unique_df <- df %>%
    summarise(across(everything(), ~length(unique(.)), .names = "num_unique_{.col}")) %>%
    pivot_longer(cols = everything(), names_to = "variable", values_to = "num_unique") %>%
    mutate(variable = gsub("^num_unique_", "", variable))  # Clean variable names
  
  # Convert unique_values from list to character safely
  unique_values_df <- unique_values_df %>%
    mutate(unique_values = sapply(unique_values, function(x) {
      if (inherits(x, "pillar_num")) {
        x <- as.numeric(x)  # Convert pillar_num to numeric first
      }
      paste0(x, collapse = ",")
    }))
  
  # Merge all dataframes
  cb <- cb %>%
    left_join(miss_df, by = "variable") %>%
    left_join(unique_values_df, by = "variable") %>%
    left_join(num_unique_df, by = "variable") %>%
    left_join(class_df, by = "variable")
  
  return(cb)
}



map_label <- function(var) {
  case_when(
    var == "viz_disp_collapsed" ~ "Disposition",
    var == "viz_ethnicity" ~ "Ethnicity",
    var == "viz_race_collapsed" ~ "Race",
    var == "viz_language" ~ "Preferred language",
    var == "viz_insurance" ~ "Insurance type",
    var == "viz_service_collapsed" ~ "ED service",
    var == "viz_ynhhs_sg2_service" ~ "SG2 service line",
    var == "viz_observed_mortality_yn" ~ "In-hospital mortality (yes/no)",
    var == "viz_drg" ~ "DRG (Diagnosis Related Group)",
    var == "viz_admission_day" ~ "Admission day of week",
    var == "viz_discharged_day" ~ "Discharge day of week",
    var == "viz_outcome_prolonged_los_yn" ~ "Prolonged LOS (yes/no)",
    var == "thro_boarding_yn" ~ "ED boarding (yes/no)",
    var == "thro_ed_arrival_time" ~ "ED arrival time",
    var == "thro_ed_arrival_day" ~ "ED arrival day of week",
    var == "thro_admit_or_obs_order_time" ~ "Admission/Obs order time",
    var == "thro_admit_or_obs_order_day" ~ "Admission/Obs order day",
    var == "thro_first_bed_assigned_time" ~ "First bed assigned time",
    var == "thro_first_bed_assigned_day" ~ "First bed assigned day",
    var == "thro_last_bed_assigned_time" ~ "Last bed assigned time",
    var == "thro_last_bed_assigned_day" ~ "Last bed assigned day",
    var == "thro_ed_departure_time" ~ "ED departure time",
    var == "thro_ed_departure_day" ~ "ED departure day of week",
    var == "summary_consult_count_all" ~ "Total consults",
    var == "summary_consult_count_unique_services" ~ "Unique consult services",
    var == "summary_pt_consult_order_yn" ~ "PT consult ordered (yes/no)",
    var == "summary_pt_consult_order_time" ~ "PT consult order time",
    var == "summary_pt_consult_order_day" ~ "PT consult order day",
    var == "summary_sw_consult_order_yn" ~ "SW consult ordered (yes/no)",
    var == "summary_sw_consult_order_time" ~ "SW consult order time",
    var == "summary_sw_consult_order_day" ~ "SW consult order day",
    var == "summary_first_edd_time" ~ "First EDD (Estimated Discharge Date) time",
    var == "summary_first_edd_day" ~ "First EDD day",
    var == "summary_first_edd_doc_time" ~ "First documented EDD time",
    var == "summary_first_edd_doc_day" ~ "First documented EDD day",
    var == "summary_last_edd_time" ~ "Last EDD time",
    var == "summary_last_edd_day" ~ "Last EDD day",
    var == "summary_last_edd_doc_time" ~ "Last documented EDD time",
    var == "summary_last_edd_doc_day" ~ "Last documented EDD day",
    var == "summary_first_rfd_status" ~ "First RFD (Ready for Discharge) status",
    var == "summary_first_rfd_time" ~ "First RFD time",
    var == "summary_first_rfd_day" ~ "First RFD day",
    var == "summary_last_rfd_status" ~ "Last RFD status",
    var == "summary_last_rfd_time" ~ "Last RFD time",
    var == "summary_last_rfd_day" ~ "Last RFD day",
    var == "con_signer_ym_provider_count" ~ "Yale Medicine signer count",
    var == "con_signer_nemg_provider_count" ~ "NEMG signer count",
    var == "con_signer_community_provider_count" ~ "Community signer count",
    var == "con_service_addiction_medicine_count" ~ "Addiction medicine consults",
    var == "con_service_cardiology_count" ~ "Cardiology consults",
    var == "con_service_cardiothoracic_surgery_count" ~ "Cardiothoracic surgery consults",
    var == "con_service_colon_and_rectal_count" ~ "Colon and rectal surgery consults",
    var == "con_service_dermatology_count" ~ "Dermatology consults",
    var == "con_service_diabetes_count" ~ "Diabetes consults",
    var == "con_service_gastroenterology_count" ~ "Gastroenterology consults",
    var == "con_service_geriatrics_count" ~ "Geriatrics consults",
    var == "con_service_hematology_count" ~ "Hematology consults",
    var == "con_service_hepatology_count" ~ "Hepatology consults",
    var == "con_service_hospitalist_service_count" ~ "Hospitalist consults",
    var == "con_service_icu_sdu_count" ~ "ICU/SDU consults",
    var == "con_service_infectious_disease_count" ~ "Infectious disease consults",
    var == "con_service_internal_medicine_count" ~ "Internal medicine consults",
    var == "con_service_lab_medicine_count" ~ "Lab medicine consults",
    var == "con_service_nephrology_count" ~ "Nephrology consults",
    var == "con_service_neuro_oncology_count" ~ "Neuro-oncology consults",
    var == "con_service_neurology_count" ~ "Neurology consults",
    var == "con_service_neurosurgery_count" ~ "Neurosurgery consults",
    var == "con_service_oncology_count" ~ "Oncology consults",
    var == "con_service_ophthalmology_count" ~ "Ophthalmology consults",
    var == "con_service_orthopedics_count" ~ "Orthopedic consults",
    var == "con_service_other_count" ~ "Other consults",
    var == "con_service_otolaryngology_ent_count" ~ "ENT (Otolaryngology) consults",
    var == "con_service_palliative_count" ~ "Palliative care consults",
    var == "con_service_pharmacy_count" ~ "Pharmacy consults",
    var == "con_service_physical_medicine_and_rehabilitation_count" ~ "PM&R consults",
    var == "con_service_picc_count" ~ "PICC consults",
    var == "con_service_plastic_surgery_count" ~ "Plastic surgery consults",
    var == "con_service_podiatry_count" ~ "Podiatry consults",
    var == "con_service_psychiatry_count" ~ "Psychiatry consults",
    var == "con_service_pulmonology_count" ~ "Pulmonology consults",
    var == "con_service_radiation_oncology_count" ~ "Radiation oncology consults",
    var == "con_service_radiology_count" ~ "Radiology consults",
    var == "con_service_rheumatology_count" ~ "Rheumatology consults",
    var == "con_service_surgery_count" ~ "Surgery consults",
    var == "con_service_sw_count" ~ "Social work consults",
    var == "con_service_trauma_count" ~ "Trauma consults",
    var == "con_service_urology_count" ~ "Urology consults",
    var == "con_service_vascular_surgery_count" ~ "Vascular surgery consults",
    var == "con_max_consult_order_to_sign_is_signer_ym_provider_yn" ~ "Longest delay from order to sign (YM provider)",
    var == "con_max_consult_order_to_sign_is_signer_nemg_provider_yn" ~ "Longest delay from order to sign (NEMG provider)",
    var == "con_max_consult_order_to_sign_is_signer_community_provider_yn" ~ "Longest delay from order to sign (Community provider)",
    var == "con_max_consult_note_to_sign_is_signer_ym_provider_yn" ~ "Longest note signing delay (YM provider)",
    var == "con_max_consult_note_to_sign_is_signer_nemg_provider_yn" ~ "Longest note signing delay (NEMG provider)",
    var == "con_max_consult_note_to_sign_is_signer_community_provider_yn" ~ "Longest note signing delay (Community provider)",
    var == "con_max_consult_note_creation_time" ~ "Time of last consult note created",
    var == "con_max_consult_note_creation_day" ~ "Day of last consult note created",
    var == "con_max_date_note_signed_time2" ~ "Time of last consult note signed",
    var == "con_max_date_note_signed_day2" ~ "Day of last consult note signed",
    var == "img_count_any" ~ "Any imaging (yes/no)",
    var == "img_count_ct" ~ "CT scan count",
    var == "img_count_mr" ~ "MRI count",
    var == "img_count_us" ~ "Ultrasound count",
    var == "icu_any_icu_yn" ~ "Any ICU stay (yes/no)",
    var == "census_daily_inpt_count" ~ "Inpatient census (daily)",
    var == "census_daily_ed_count" ~ "ED census (daily)",
    var == "viz_gender" ~ "Gender",
    var == "viz_right_service_hf_yn" ~ "Right patient right service (yes/no)",
    TRUE ~ var  # default: return original if not matched
  )
}


process_df <- function(raw_df, data_df) {
  cb <- get_codebook(data_df)
  
  raw_df %>%
    filter(!type %in% c("drop", "continuous", "unchanged")) %>%
    rename(variable = col_name) %>%
    mutate(source = map_source(variable),
           label = map_label(variable)) %>%
    left_join(cb, by = "variable") %>%
    select(variable, label, type, source, n_miss, pct_miss) 
}


View(df3_simple)
# PROCESS AND EXPORT ------------------------------------------------------


raw_df_simple = read_excel("/gpfs/milgram/project/rtaylor/imc33/LOS/data/features_los_simple.xlsx")
raw_df_complex = read_excel("/gpfs/milgram/project/rtaylor/imc33/LOS/data/features_los.xlsx")
data_df = read_csv("/gpfs/milgram/project/rtaylor/imc33/LOS/data/master_los.csv")


df3_simple <- process_df(raw_df_simple, data_df)
df3_complex <- process_df(raw_df_complex, data_df)

View(df3_complex)

write.csv(df3_simple, "/gpfs/milgram/project/rtaylor/imc33/LOS/output/supplement_los_features_2a.csv")
write.csv(df3_complex, "/gpfs/milgram/project/rtaylor/imc33/LOS/output/supplement_los_features_2b.csv")
