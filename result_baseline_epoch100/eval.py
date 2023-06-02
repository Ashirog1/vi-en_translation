import sacrebleu

hypothesis = []
reference = []
with open('hypothesis.txt') as h, open('reference.txt') as r:
    for i, line in enumerate(h): 
        line = line.replace('\n', '')
        hypothesis.append(line.replace('<s>', '').replace('<unk>', '').replace('</s>', '') )
    for i, line in enumerate(r): 
        line = line.replace('\n', '')
        reference.append([line.replace('<s>', '').replace('<unk>', '').replace('</s>', '')])

# print(hypothesis)
# print(reference)
for i in range(0, 10):
    score = sacrebleu.corpus_bleu([hypothesis[i]], reference[i])
    print(score.score)

score = sacrebleu.corpus_bleu(hypothesis, reference)
print(f"The BLEU score is: {score.score}")
print(f"BLEU details: {score}")