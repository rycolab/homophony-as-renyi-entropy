
for model in lstm ngram
do
    for lang in nld deu eng
    do
        make MONOMORPHEMIC=True LANGUAGE=${lang} MODEL=${model}
    done
done

for model in lstm ngram
do
    for lang in nld deu eng
    do
        make MONOMORPHEMIC=False LANGUAGE=${lang} MODEL=${model}
    done
done
