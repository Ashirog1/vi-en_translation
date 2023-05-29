import sacrebleu

with open('hypothesis.txt') as h, open('reference.txt') as r:
    hypothesis = h.readlines()  
    reference = r.readlines()

score = sacrebleu.corpus_bleu(hypothesis, reference)
print(f"The BLEU score is: {score.score}")
print(f"BLEU details: {score}")