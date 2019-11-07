#!/usr/bin/env bash
moses-tokenizer en | python3 -c "import sys; print(sys.stdin.read().lower())" | ../lib/tools/apply_bpe.py --bpe_rules $1

#cat $out | moses-tokenizer en | python3 -c "import sys; print(sys.stdin.read().lower())" > .tmp
#mv .tmp $out
#
#cat $out | ../lib/tools/apply_bpe.py --bpe_rules $2

