#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

paragraph_extraction ()
{
    SOURCE_DIR=$1
    TARGET_DIR=$2
    echo "Now run sh run.sh --para_extraction.Start paragraph extraction, this may take a few hours"
    echo "Source dir: $SOURCE_DIR"
    echo "Target dir: $TARGET_DIR"

    echo "Processing testset"
    cat $SOURCE_DIR/testset/small_train_data.test1.json | python paragraph_extraction.py test \
            > $TARGET_DIR/testset/small_train_data.test1.json

    echo "Processing trainset"
    cat $SOURCE_DIR/trainset/small_train_data.json | python paragraph_extraction.py train \
            > $TARGET_DIR/trainset/small_train_data.json


    echo "Processing devset"
    cat $SOURCE_DIR/devset/small_train_data.dev.json | python paragraph_extraction.py test \
            > $TARGET_DIR/devset/small_train_data.dev.json

    echo "Paragraph extraction done!"
}


PROCESS_NAME="$1"
case $PROCESS_NAME in
    --para_extraction)
    # Start paragraph extraction 
    if [ ! -d ../data ]; then
        echo "Please download the preprocessed data first (See README - Preprocess)"
        exit 1
    fi
    paragraph_extraction  ../data  ../data/extracted
    ;;
    --prepare|--train|--evaluate|--predict)
        # Start Paddle baseline
        python run.py $@
    ;;
    *)
        echo $"Usage: $0 {--para_extraction|--prepare|--train|--evaluate|--predict}"
esac
