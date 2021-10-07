LANGUAGE := eng
DATASET := celex
MODEL := lstm
MONOMORPHEMIC := True
DROP_POS := False
N_BOOTSTRAP := 1000

MONO_ARG := $(if $(filter-out $(MONOMORPHEMIC), True),,--mono)
MONO_PATH := $(if $(filter-out $(MONOMORPHEMIC), True),multi,mono)
POS_ARG := $(if $(filter-out $(DROP_POS), True),,--drop-pos)
POS_PATH := $(if $(filter-out $(DROP_POS), True),pos_keep,pos_drop)

DATA_DIR_BASE := ./data
DATA_DIR := $(DATA_DIR_BASE)/$(DATASET)
DATA_DIR_LANG := $(DATA_DIR)/$(LANGUAGE)/$(MONO_PATH)/$(POS_PATH)
CHECKPOINT_DIR_BASE := ./checkpoint
CHECKPOINT_DIR := $(CHECKPOINT_DIR_BASE)/$(DATASET)/$(POS_DIR)
CHECKPOINT_DIR_LANG := $(CHECKPOINT_DIR)/$(LANGUAGE)/$(MONO_PATH)/$(POS_PATH)
RESULTS_DIR_BASE := ./results
RESULTS_DIR := $(RESULTS_DIR_BASE)/$(DATASET)/$(POS_DIR)
RESULTS_DIR_LANG := $(RESULTS_DIR)/$(LANGUAGE)/$(MONO_PATH)/$(POS_PATH)

CELEX_RAW_DIR := $(DATA_DIR_BASE)/celex/raw/
CELEX_RAW_FILE_COMPRESSED := $(CELEX_RAW_DIR)/LDC96L14.tar.gz
CELEX_RAW_DIR_UNCOMPRESSED := $(CELEX_RAW_DIR)/LDC96L14/
CELEX_RAW_FILE := $(CELEX_RAW_DIR_UNCOMPRESSED)/extracted.txt

CELEX_EXTRACTED_DIR := $(CELEX_RAW_DIR)/extracted/
CELEX_EXTRACTED_FILE := $(CELEX_EXTRACTED_DIR)/$(LANGUAGE)_lemma_$(MONOMORPHEMIC)_False_$(DROP_POS)_0_inf.tsv
PROCESSED_DATA_FILE := $(DATA_DIR_LANG)/processed.pckl

CHECKPOINT_FILE := $(CHECKPOINT_DIR_LANG)/$(MODEL)/model.tch
LOSSES_FILE := $(CHECKPOINT_DIR_LANG)/$(MODEL)/losses.pckl
RENYI_FILE := $(CHECKPOINT_DIR_LANG)/$(MODEL)/renyi.pckl
RENYI_SAMPLES := $(CHECKPOINT_DIR_LANG)/$(MODEL)/samples.pckl

all: get_data train eval get_renyi sample_renyi

analysis: $(LOSSES_FILE) $(RENYI_FILE) $(RENYI_SAMPLES)
	echo $(RENYI_SAMPLES)
	python src/h04_analysis/renyi.py --raw-file $(CELEX_EXTRACTED_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type $(MODEL)

sample_renyi: $(RENYI_SAMPLES)
	echo "Finished getting model's renyi" $(LANGUAGE)

get_renyi: $(RENYI_FILE)
	echo "Finished getting model's renyi" $(LANGUAGE)

eval: $(LOSSES_FILE)
	echo "Finished evaluating model" $(LANGUAGE)

train: $(CHECKPOINT_FILE)
	echo "Finished training model" $(LANGUAGE)

get_data: $(PROCESSED_DATA_FILE)
	echo "Finished processing data" $(LANGUAGE)

get_celex: $(CELEX_RAW_FILE)

clean:
	rm $(PROCESSED_DATA_FILE)

$(RENYI_SAMPLES): | $(CHECKPOINT_FILE)
	echo "Sample renyi from model" $(LOSSES_FILE)
	python src/h03_eval/sample_renyis.py --raw-file $(CELEX_EXTRACTED_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type $(MODEL) --batch-size 5000 --n-samples $(N_BOOTSTRAP)

$(RENYI_FILE): | $(CHECKPOINT_FILE)
	echo "Eval models" $(LOSSES_FILE)
	python src/h03_eval/get_renyi.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type $(MODEL) --batch-size 1024

# Eval language models
$(LOSSES_FILE): | $(CHECKPOINT_FILE)
	echo "Eval models" $(LOSSES_FILE)
	python src/h03_eval/eval.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type $(MODEL)

# Train types Model
$(CHECKPOINT_FILE): | $(PROCESSED_DATA_FILE)
	echo "Train types model" $(CHECKPOINT_FILE)
	mkdir -p $(CHECKPOINT_DIR_LANG)/$(MODEL)/
	python src/h02_learn/train.py --data-file $(PROCESSED_DATA_FILE) --checkpoints-path $(CHECKPOINT_DIR_LANG) --model-type $(MODEL)

$(PROCESSED_DATA_FILE): | $(CELEX_EXTRACTED_FILE)
	mkdir -p $(DATA_DIR_LANG)
	python src/h01_data/process_data.py --dataset $(DATASET) --src-file $(CELEX_EXTRACTED_FILE) --data-path $(DATA_DIR_LANG)

# Get celex
$(CELEX_EXTRACTED_FILE): | $(CELEX_RAW_FILE)
	mkdir -p $(CELEX_EXTRACTED_DIR)
	echo "Get celex data" $(CELEX_EXTRACTED_FILE)
	python src/h01_data/extract_lex_celex.py --language $(LANGUAGE) --src-path $(CELEX_RAW_DIR_UNCOMPRESSED) --tgt-path $(CELEX_EXTRACTED_DIR) $(MONO_ARG) $(POS_ARG)

$(CELEX_RAW_FILE): | $(CELEX_RAW_FILE_COMPRESSED)
	echo "Get celex data"
	tar -C $(CELEX_RAW_DIR) -zxvf $(CELEX_RAW_FILE_COMPRESSED)
	touch $(CELEX_RAW_FILE)
