#!/usr/bin/env bash

# this script is copied from egs/librispeech/s5/local/prepare_dict.sh

# Copyright 2021 Xiaomi Corporation Yongqing Wang
# Apache 2.0

# Prepares the dictionary and auto-generates the pronunciations for the words,
# that are in our vocabulary but not in CMUdict

stage=0
nj=4 # number of parallel Sequitur G2P jobs, we would like to use
cmd=run.pl
train_set=
test_sets=

. ./path.sh || exit 1
. parse_options.sh || exit 1;


if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <g2p-model> <dst-dir>"
  echo "e.g.: $0 ~/GigaSpeech_data/dict/g2p/g2p.model.4 data/local/dict"
  echo "Options:"
  echo "  --cmd '<command>'    # script to launch jobs with, default: run.pl"
  echo "  --nj <nj>            # number of jobs to run, default: 4."
  exit 1
fi

g2p_model=$1
dst_dir=$2

lexicon_raw_nosil=$dst_dir/lexicon_raw_nosil.txt
cmudict_dir=$dst_dir/cmudict
cmudict_plain=$dst_dir/cmudict.0.7a.plain
vocab=

[ -d $dst_dir ] || mkdir -p $dst_dir || exit 1;

if [ $stage -le 0 ]; then
  echo "Downloading and preparing CMUdict"
  if [ ! -s $cmudict_dir/cmudict.0.7a ]; then
    svn co -r 12440 https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict $cmudict_dir || exit 1;
  fi
  echo "Removing the pronunciation variant markers ..."
  grep -v ';;;' $cmudict_dir/cmudict.0.7a | \
    perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' \
    > $cmudict_plain || exit 1;
fi

if [ $stage -le 1 ]; then
  # check if we have Sequitur G2P is installed
  if [ ! -f  "$sequitur" ]; then
    if ! which swig >&/dev/null; then
      echo "Please install 'swig' and then run $KALDI_ROOT/tools/extra/install_sequitur.sh"
      exit 1
    else
      echo "Sequitur G2P not found- running $KALDI_ROOT/tools/extra/install_sequitur.sh"
      pushd $KALDI_ROOT/tools
      extras/install_sequitur.sh || exit 1
      popd
    fi
  fi
  [[ -f "$sequitur" ]] || { echo "Still can't find Sequitur G2P- check your path.sh"; exit 1; }
fi

if [ $stage -le 2 ]; then
  g2p_dir=$dst_dir/g2p
  auto_vocab_prefix="$g2p_dir/vocab_autogen"
  auto_lexicon_prefix="$g2p_dir/lexicon_autogen"
  vocab=$dst_dir/words.txt
  [ -s "$train_set" ] && echo "$train_set is empty" && exit 1
  [ -f $dst_dir/words-raw.txt ] && rm $dst_dir/words-raw.txt
  for dataset in $train_set $test_sets; do
    cat data/$dataset/text | sed 's|\t| |g' | cut -d " " -f 2- | sed 's| |\n|g' >>$dst_dir/words-raw.txt
  done
  sort -u $dst_dir/words-raw.txt | sed '/^$/d' > $vocab

  mkdir -p $g2p_dir/log
  auto_vocab_splits=$(eval "echo $auto_vocab_prefix.{$(seq -s',' $nj | sed 's/,$//')}")
  awk 'NR==FNR{a[$1] = 1; next} !($1 in a)' $cmudict_plain $vocab |\
    sort | tee $g2p_dir/vocab_autogen.full |\
    utils/split_scp.pl /dev/stdin $auto_vocab_splits || exit 1
  echo "Autogenerating pronunciations for the words in $auto_vocab_prefix.* ..."
  $cmd JOB=1:$nj $g2p_dir/log/g2p.JOB.log \
    local/g2p.sh  $auto_vocab_prefix.JOB $g2p_model $auto_lexicon_prefix.JOB || exit 1
  g2p_vocab_size=$(wc -l <$g2p_dir/vocab_autogen.full)
  g2p_lex_size=$(wc -l < <(cat $auto_lexicon_prefix.*))
  [[ "$g2p_vocab_size" -eq "$g2p_lex_size" ]] || { echo "Unexpected G2P error"; exit 1; }
  sort <(cat $auto_vocab_prefix.*) >$dst_dir/vocab_autogen.txt
  sort <(cat $auto_lexicon_prefix.*) >$dst_dir/lexicon_autogen.txt
  echo "$(wc -l <$g2p_dir/vocab_autogen.full) pronunciations autogenerated OK"
fi

if [ $stage -le 3 ]; then
  echo "Combining the CMUdict pronunciations with the autogenerated ones ..."
  awk 'NR==FNR{a[$1]=1; next} ($1 in a)' $vocab $cmudict_plain |\
    cat - $dst_dir/lexicon_autogen.txt | sort >$lexicon_raw_nosil || exit 1
  raw_lex_size=$(cat $lexicon_raw_nosil | awk '{print $1}' | sort -u | wc -l)
  vocab_size=$(wc -l <$vocab)
  [[ "$vocab_size" -eq "$raw_lex_size" ]] || {
    echo "Inconsistent lexicon($raw_lex_size) vs vocabulary($vocab_size) size!";
    exit 1; }
  echo "Combined lexicon saved to '$lexicon_raw_nosil'"
fi

# The copy operation below is necessary, if we skip the g2p stages(e.g. using --stage 3)
[ ! -s "$lexicon_raw_nosil" ] || (echo "$lexicon_raw_nosil is empty" && exit 1)

if [ $stage -le 4 ]; then
  silence_phones=$dst_dir/silence_phones.txt
  optional_silence=$dst_dir/optional_silence.txt
  nonsil_phones=$dst_dir/nonsilence_phones.txt
  extra_questions=$dst_dir/extra_questions.txt

  echo "Preparing phone lists and clustering questions"
  (echo SIL; echo SPN;) > $silence_phones
  echo SIL > $optional_silence
  # nonsilence phones; on each line is a list of phones that correspond
  # really to the same base phone.
  awk '{for (i=2; i<=NF; ++i) { print $i; gsub(/[0-9]/, "", $i); print $i}}' $lexicon_raw_nosil |\
    sort -u |\
    perl -e 'while(<>){
      chop; m:^([^\d]+)(\d*)$: || die "Bad phone $_";
      $phones_of{$1} .= "$_ "; }
      foreach $list (values %phones_of) {print $list . "\n"; } ' | sort \
      > $nonsil_phones || exit 1;
  # A few extra questions that will be added to those obtained by automatically clustering
  # the "real" phones.  These ask about stress; there's also one for silence.
  cat $silence_phones| awk '{printf("%s ", $1);} END{printf "\n";}' > $extra_questions || exit 1;
  cat $nonsil_phones | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
    $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
    >> $extra_questions || exit 1;
  echo "$(wc -l <$silence_phones) silence phones saved to: $silence_phones"
  echo "$(wc -l <$optional_silence) optional silence saved to: $optional_silence"
  echo "$(wc -l <$nonsil_phones) non-silence phones saved to: $nonsil_phones"
  echo "$(wc -l <$extra_questions) extra triphone clustering-related questions saved to: $extra_questions"
fi

if [ $stage -le 4 ]; then
  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |\
  cat - $lexicon_raw_nosil | sort | uniq >$dst_dir/lexicon.txt
  echo "Lexicon text file saved as: $dst_dir/lexicon.txt"
fi

exit 0
