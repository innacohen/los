library(tidyverse)
library(readxl)
library(sjlabelled)
library(naniar)

raw_df = read_excel("/gpfs/milgram/project/rtaylor/imc33/LOS/data/features_los.xlsx")
data_df = read_csv("/gpfs/milgram/project/rtaylor/imc33/LOS/data/master_los.csv")

map_source <- function(value) {
  case_when(
    str_starts(value, "viz") ~ "Vizient",
    str_starts(value, "con") | str_detect(value, "consult") ~ "Consult dashboard",
    str_starts(value, "census") ~ "Census",
    str_starts(value, "rfd") | str_detect(value, "edd") | str_detect(value, "rfd") ~ "Discharge Data",
    str_starts(value, "thro") ~ "Essential Services",
    str_starts(value, "img") ~ "Imaging dashboard",
    value == "icu_any_icu_yn" ~ "Quality dashboard",
    TRUE ~ NA_character_
  )
}


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


df2 <- raw_df %>%
  filter(!type %in% c("drop", "continuous","unchanged")) %>%
  rename(variable = col_name) %>%
  mutate(source = map_source(variable))

cb = get_codebook(data_df)


df3 = df2 %>%
      left_join(cb, by="variable") %>%
      select(-r_class)

#sapply(df3, class)


write.csv(df3, "/gpfs/milgram/project/rtaylor/imc33/LOS/output/supplement_los_features.csv")
