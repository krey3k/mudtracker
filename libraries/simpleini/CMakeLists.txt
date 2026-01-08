##########################################
# SimpleIni
##########################################

add_library(
  SimpleIni
  ConvertUTF.c 
  )

target_include_directories(SimpleIni PUBLIC ./)